import os
import time
import json
import torch
import numpy as np

from datetime import datetime

from model import PPO
from unity_env import UnityTCPEnv


print("============================================================================================")

################################### Training ###################################

####### initialize environment hyperparameters ######

env_name = "Unity"

max_ep_len = 1000  
max_training_timesteps = int(1e5)

print_freq = max_ep_len * 4

random_seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################

################ Hyperparameters ################

K_epochs = 40
eps_clip = 0.2
gamma = 0.99

lr_actor = 0.0003
lr_critic = 0.001

#####################################################


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(BASE_DIR, "PPO_preTrained", env_name, "latest.pth")
trajectory_path = os.path.join(BASE_DIR, "PPO_logs", env_name, "latest_ppo_trajectory.json")


def save_latest_model(ppo_agent):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temporary_path = checkpoint_path + ".tmp"
    torch.save(ppo_agent.policy_old.state_dict(), temporary_path)
    os.replace(temporary_path, checkpoint_path)


def _json_value(value):
    """Keep the exact rollout samples readable without changing PPO tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_episode_trajectory(trajectory, episode_number, episode_steps):

    step_count = len(trajectory.rewards)
    lengths = {
        "states": len(trajectory.states),
        "actions": len(trajectory.actions),
        "logprobs": len(trajectory.logprobs),
        "state_values": len(trajectory.state_values),
        "rewards": step_count,
        "is_terminals": len(trajectory.is_terminals),
    }
    if step_count == 0 or any(length != step_count for length in lengths.values()):
        raise RuntimeError(f"Incomplete PPO trajectory: {lengths}")
    if not trajectory.is_terminals[-1]:
        raise RuntimeError("This Monte Carlo update requires a terminal trajectory")

    payload = {
        "episode": episode_number,
        "episode_steps": episode_steps,
        "training_step_start": episode_steps - step_count + 1,
        "training_step_end": episode_steps,
        "num_steps": step_count,
        "total_reward": float(sum(trajectory.rewards)),
        "steps": [
            {
                "state": _json_value(trajectory.states[i]),
                "action": _json_value(trajectory.actions[i]),
                "old_logprob": _json_value(trajectory.logprobs[i]),
                "old_state_value": _json_value(trajectory.state_values[i]),
                "reward": float(trajectory.rewards[i]),
                "done": bool(trajectory.is_terminals[i]),
            }
            for i in range(step_count)
        ],
    }

    os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
    temp_path = trajectory_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, trajectory_path)
    return step_count


def main():
    print("[MODE] TRAIN SERVER (action inference + end-of-episode PPO update)")
    print("training environment name :", env_name)
    print("state space dimension : 31")
    print("action space : movement(5), rotation(1), combat(3)")
    print("PPO update frequency : every finished episode")
    print("Training samples : last", max_ep_len, "steps of each episode")
    print("PPO K epochs :", K_epochs)
    print("PPO epsilon clip :", eps_clip)
    print("discount factor (gamma) :", gamma)
    print("checkpoint :", checkpoint_path)
    print("latest PPO trajectory :", trajectory_path)

    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    ###################### logging ######################
    log_dir = os.path.join(BASE_DIR, "PPO_logs", env_name)
    os.makedirs(log_dir, exist_ok=True)
    run_num = len([n for n in os.listdir(log_dir) if n.endswith(".csv")])
    log_f_name = os.path.join(log_dir, f"PPO_{env_name}_log_{run_num}.csv")
    print("logging at :", log_f_name)

    ################# training procedure ################
    state_dim = 31
    ppo_agent = PPO(
        state_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        device,
    )

    if os.path.isfile(checkpoint_path):
        weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
        ppo_agent.policy.load_state_dict(weights)
        ppo_agent.policy_old.load_state_dict(weights)
        print("Loaded latest model for continued training:", checkpoint_path)
    else:
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())
        save_latest_model(ppo_agent)
        print("Created initial checkpoint:", checkpoint_path)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at :", start_time)

    print_running_reward = 0.0
    print_running_episodes = 0
    time_step = 0
    i_episode = 0
    last_print_timestep = 0

    env = None
    with open(log_f_name, "w", encoding="utf-8") as log_f:
        log_f.write("episode,timestep,reward\n")
        try:
            env = UnityTCPEnv(host="127.0.0.1", port=5005)

            ################# training loop #################
            while True:
                # The next reset is sent only after Unity receives episode_end.
                state = env.reset()
                current_ep_reward = 0.0
                done = False

                # One episode = one trajectory. select_action() records
                trajectory = ppo_agent.buffer
                if trajectory.rewards or trajectory.states:
                    raise RuntimeError("Previous episode remains in the PPO trajectory buffer")

                episode_steps = 0

                # While episode doesn't end
                while not done:
                    action = ppo_agent.select_action(state)
                    state, reward, done, _ = env.step(action)

                    trajectory.rewards.append(reward)
                    trajectory.is_terminals.append(done)

                    episode_steps += 1
                    time_step += 1
                    current_ep_reward += reward

                    if len(trajectory.rewards) > max_ep_len:
                        trajectory.states.pop(0)
                        trajectory.actions.pop(0)
                        trajectory.logprobs.pop(0)
                        trajectory.state_values.pop(0)
                        trajectory.rewards.pop(0)
                        trajectory.is_terminals.pop(0)

                #####################################################
                # UPDATE: Unity is still waiting for episode_end.
                #####################################################
                i_episode += 1
                training_steps = save_episode_trajectory(trajectory, i_episode, episode_steps)
                print(f"[TRAIN] Episode {i_episode} ended | "
                      f"episode steps={episode_steps} | training steps={training_steps} "
                      f"({episode_steps - training_steps + 1}~{episode_steps}) | "
                      f"episode reward={current_ep_reward:.3f}")
                print("[TRAIN] PPO trajectory saved:", trajectory_path)

                update_start = time.monotonic()
                ppo_agent.update() 
                update_seconds = time.monotonic() - update_start
                ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())

                # Save first, then release Unity to respawn.
                save_latest_model(ppo_agent)
                print(f"[TRAIN] PPO update complete ({K_epochs} epochs, {update_seconds:.2f}s)")
                print("[TRAIN] Latest model saved:", checkpoint_path)
                env.finish_episode()  
                print("[TRAIN] episode_end sent; Unity may respawn.")

                #####################################################
                # Logging, retained from the original training loop.
                #####################################################
                print_running_reward += current_ep_reward
                print_running_episodes += 1 
                log_f.write(f"{i_episode},{time_step},{current_ep_reward:.4f}\n")
                log_f.flush()

                if time_step - last_print_timestep >= print_freq:
                    print_avg_reward = round(print_running_reward / print_running_episodes, 2)
                    print(f"Episode : {i_episode} \t Timestep : {time_step} "
                          f"\t Average Reward : {print_avg_reward}")
                    print_running_reward = 0.0
                    print_running_episodes = 0
                    last_print_timestep = time_step

        except KeyboardInterrupt:
            print("\n[TRAIN] Stopped. The last fully updated episode is in latest.pth.")
        finally:
            if env is not None:
                env.close()

    end_time = datetime.now().replace(microsecond=0)
    print("============================================================================================")
    print("Started training at :", start_time)
    print("Finished training at :", end_time)
    print("Total training time :", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    main()
