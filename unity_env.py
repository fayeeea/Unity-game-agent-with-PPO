import json
import socket

import numpy as np
import torch

from state_preprocessor import preprocess_state


class UnityTCPEnv:
    def __init__(self, host="127.0.0.1", port=5005):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.conn = None
        self.reader = None
        self.terminal_pending = False

        try:
            self.server.bind((host, port))
            self.server.listen(1)
            print(f"[TRAIN] Waiting for Unity on {host}:{port}")
            self.conn, addr = self.server.accept()
            self.reader = self.conn.makefile("r", encoding="utf-8")
            print("[TRAIN] Unity connected:", addr)
        except BaseException:
            self.close()
            raise

    def receive(self):
        line = self.reader.readline()
        if not line:
            raise ConnectionError("Unity disconnected")
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError("Unity message must be a JSON object")
        return data

    def send(self, message):
        payload = json.dumps(message, ensure_ascii=False) + "\n"
        self.conn.sendall(payload.encode("utf-8"))

    @staticmethod
    def _state_from_message(data):
        raw_state = data.get("state")
        if not isinstance(raw_state, dict) or "enemyRelativePosition" not in raw_state:
            raise ValueError(
                "Missing nested state.enemyRelativePosition; "
                f"type={data.get('type')!r}, keys={list(data)}"
            )
        state = preprocess_state(raw_state)
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (31,):
            raise ValueError(f"Expected state shape (31,), got {state.shape}")
        return state

    def reset(self):
        if self.terminal_pending:
            raise RuntimeError("Call finish_episode() before the next reset()")
        data = self.receive()
        if data.get("type") != "reset" or data.get("done", False):
            raise ValueError(f"Expected reset message, got: {data.get('type')!r}")
        return self._state_from_message(data)

    def step(self, action):
        if self.terminal_pending:
            raise RuntimeError("Episode ended; call finish_episode() before step()")

        # Keep rotation normalized inside PPO; convert only on the wire.
        self.send({
            "type": "action",
            "movement": int(action[0]),
            "rotation": float(action[1]) * 30.0,
            "combat": int(action[2]),
        })

        data = self.receive()
        if data.get("type") != "step":
            raise ValueError(f"Expected step message, got: {data.get('type')!r}")

        next_state = self._state_from_message(data)
        reward = float(data["reward"])
        done = bool(data["done"])
        self.terminal_pending = done

        # IMPORTANT: do NOT send episode_end here. Training is not finished yet.
        return next_state, reward, done, {}

    def finish_episode(self):
        """Release Unity to respawn only after train.py finishes PPO update/save."""
        if not self.terminal_pending:
            raise RuntimeError("No terminal step to acknowledge")
        self.send({"type": "episode_end"})
        self.terminal_pending = False

    def close(self):
        if self.conn is not None:
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.reader is not None:
            try:
                self.reader.close()
            except OSError:
                pass
        if self.conn is not None:
            self.conn.close()
        self.server.close()
