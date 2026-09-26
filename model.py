import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical



class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]



class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.backbone = nn.Sequential(
                                nn.Linear(state_dim, 64),
                                nn.Tanh(),
                                nn.Linear(64, 64),
                                nn.Tanh(),
                            )
        
        self.movement_head = nn.Linear(64, 5)
        self.rotation_mean_head = nn.Linear(64, 1)
        self.rotation_log_std_head = nn.Linear(64, 1)
        self.combat_head = nn.Linear(64, 3)

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                        )
        
    def actor(self, state):
        x = self.backbone(state)

        return (
            self.movement_head(x),
            self.rotation_mean_head(x),
            self.rotation_log_std_head(x),
            self.combat_head(x)
        )
    
    def act(self, state):

        movement_logits, rotation_mean, rotation_log_std, combat_logits = self.actor(state)

        action_movement = Categorical(logits=movement_logits)
        action_rotation_std = torch.exp(torch.clamp(rotation_log_std, -5.0, 1.0))
        action_rotation = Normal(rotation_mean, action_rotation_std)
        action_combat = Categorical(logits=combat_logits)
        action_rotation_sample = action_rotation.sample()
        rotation = torch.tanh(action_rotation_sample).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # tanh inverse
        raw_rotation = torch.atanh(rotation)

        rotation_logprob = (
            action_rotation.log_prob(raw_rotation)
            - torch.log(1 - rotation.pow(2) + 1e-6)
        ).sum(dim=-1)

        actions = [
            action_movement.sample().detach(),
            rotation.detach(),
            action_combat.sample().detach()
        ]

        action_logprobs = (
            action_movement.log_prob(actions[0])
            + rotation_logprob
            + action_combat.log_prob(actions[2])
        )

        state_val = self.critic(state).squeeze(-1)

        return actions, action_logprobs.detach(), state_val.detach()
            

    def evaluate(self, state, action):

        movement_logits, rotation_mean, rotation_log_std, combat_logits = self.actor(state)
        action_movement = Categorical(logits=movement_logits)
        action_rotation_std = torch.exp(torch.clamp(rotation_log_std, -5.0, 1.0))
        action_rotation = Normal(rotation_mean, action_rotation_std)
        action_combat = Categorical(logits=combat_logits)

        movement = action[:, 0].long()
        rotation = action[:, 1:2]
        combat = action[:, 2].long()

        # tanh inverse
        raw_rotation = torch.atanh(rotation.clamp(-1.0 + 1e-6, 1.0 - 1e-6))

        rotation_logprob = (
            action_rotation.log_prob(raw_rotation)
            - torch.log(1 - rotation.pow(2) + 1e-6)
        ).sum(dim=-1)

        action_logprobs = (
            action_movement.log_prob(movement)
            + rotation_logprob
            + action_combat.log_prob(combat)
        )

        state_values = self.critic(state)
        
        return action_logprobs, state_values


class PPO:
    def __init__(self, state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):

        self.device = device
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim).to(self.device)
        self.actor_params = (
            list(self.policy.backbone.parameters())
            + list(self.policy.movement_head.parameters())
            + list(self.policy.rotation_mean_head.parameters())
            + list(self.policy.rotation_log_std_head.parameters())
            + list(self.policy.combat_head.parameters())
        )

        self.optimizer = torch.optim.Adam([
            {'params': self.actor_params, 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)

            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return [
            int(action[0].item()),
            float(action[1].item()),
            int(action[2].item())
        ]


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-7)

        # convert list to tensor
        old_states = torch.stack(self.buffer.states, dim=0).reshape(-1, self.state_dim).detach().to(self.device)
        old_actions = torch.stack([
            torch.cat([
                action[0].reshape(1).float(),
                action[1].reshape(1),
                action[2].reshape(1).float()
            ])
            for action in self.buffer.actions
        ], dim=0).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).reshape(-1).detach().to(self.device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).reshape(-1).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values= self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = state_values.reshape(-1)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))