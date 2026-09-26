import json
from pathlib import Path

import torch
import torch.nn.functional as F



STATE_DIM = 31
MAX_DISTANCE = 10.0
MAX_ROTATION = 30.0
MOVEMENT_DIM = 5  # Idle, W, A, S, D
COMBAT_DIM = 3    # Idle, Attack, Parry


def one_hot(value: int, num_classes: int) -> torch.Tensor:
    index = torch.tensor(int(value), dtype=torch.long)

    return F.one_hot(index, num_classes=num_classes).to(torch.float32)


def normalize_position(value: float) -> torch.Tensor:
    value = torch.tensor(value, dtype=torch.float32)

    return torch.clamp(value / MAX_DISTANCE,-1.0, 1.0)


def normalize_health(value: float) -> torch.Tensor:
    value = torch.tensor(value, dtype=torch.float32)

    return torch.clamp(value,0.0,1.0)


def normalize_rotation(value: float) -> torch.Tensor:
    value = torch.tensor(value, dtype=torch.float32)

    return torch.clamp(value / MAX_ROTATION,-1.0,1.0)

def preprocess_state(state: dict, device: torch.device | str = "cpu") -> torch.Tensor:
    relative_position = state["enemyRelativePosition"]

    relative_x = normalize_position(relative_position["x"])
    relative_z = normalize_position(relative_position["z"])



    distance = normalize_position(state["distance"])


    angle_degrees = torch.tensor(state["angleToEnemy"],dtype=torch.float32)

    angle_radians = torch.deg2rad(angle_degrees)
    angle_sin = torch.sin(angle_radians)
    angle_cos = torch.cos(angle_radians)

    my_health = normalize_health(state["myHealth"])
    enemy_health = normalize_health(state["enemyHealth"])

    my_action = one_hot(state["myAction"],COMBAT_DIM)
    enemy_action = one_hot(state["enemyAction"],COMBAT_DIM)
    my_previous_movement = one_hot(state["myPreviousMovement"],MOVEMENT_DIM)
    my_previous_rotation = normalize_rotation(state["myPreviousRotation"] )
    my_previous_combat = one_hot(state["myPreviousCombat"],COMBAT_DIM)

    enemy_previous_movement = one_hot(state["enemyPreviousMovement"],MOVEMENT_DIM)

    enemy_previous_rotation = normalize_rotation(state["enemyPreviousRotation"])

    enemy_previous_combat = one_hot(state["enemyPreviousCombat"],COMBAT_DIM)


    result = torch.cat(
        [
            relative_x.reshape(1),
            relative_z.reshape(1),

            distance.reshape(1),

            angle_sin.reshape(1),
            angle_cos.reshape(1),

            my_health.reshape(1),
            enemy_health.reshape(1),

            my_action,
            enemy_action,

            my_previous_movement,
            my_previous_rotation.reshape(1),
            my_previous_combat,

            enemy_previous_movement,
            enemy_previous_rotation.reshape(1),
            enemy_previous_combat,
        ],
        dim=0
    )

    if result.shape != (STATE_DIM,):
        raise ValueError(f"State dim Error: ")

    return result.to(device)


def load_trajectory(json_path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:

    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as file:
        trajectory = json.load(file)

    states = trajectory["states"]

    if not states:
        return torch.empty(
            (0, STATE_DIM),
            dtype=torch.float32,
            device=device
        )

    processed_states = [
        preprocess_state(state, device=device)
        for state in states
    ]

    return torch.stack(processed_states, dim=0)
