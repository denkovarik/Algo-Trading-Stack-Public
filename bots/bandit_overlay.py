# bots/bandit_overlay.py
# LinUCB contextual bandit with per-action ridge regression and UCB exploration.
from __future__ import annotations
import numpy as np

class LinUCB:
    def __init__(self, d: int, n_actions: int, alpha: float = 0.6, l2: float = 1.0):
        self.d = int(d); self.k = int(n_actions)
        self.alpha = float(alpha); self.l2 = float(l2)
        self.A = [np.eye(self.d) * self.l2 for _ in range(self.k)]
        self.b = [np.zeros(self.d, dtype=np.float64) for _ in range(self.k)]
        self._updates = 0

    def choose_ucb(self, x: np.ndarray) -> int:
        x = self._vec(x)
        scores = []
        for a in range(self.k):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(theta @ x)
            bonus = float(self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0)))
            scores.append(mean + bonus)
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, a: int, r: float):
        x = self._vec(x); a = int(a); r = float(r)
        self.A[a] += np.outer(x, x)
        self.b[a] += x * r
        self._updates += 1

    def total_updates(self) -> int:
        return int(self._updates)

    def _vec(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"Context dim mismatch: expected {self.d}, got {x.shape[0]}")
        return x

    def save(self, path: str):
        import numpy as _np
        _np.savez_compressed(path, d=self.d, k=self.k, alpha=self.alpha, l2=self.l2, updates=self._updates,
                             **{f"A_{i}": self.A[i] for i in range(self.k)},
                             **{f"b_{i}": self.b[i] for i in range(self.k)})

    @staticmethod
    def load(path: str) -> "LinUCB":
        import numpy as _np, os
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = _np.load(path, allow_pickle=False)
        obj = LinUCB(int(data["d"]), int(data["k"]), float(data["alpha"]), float(data["l2"]))
        for i in range(obj.k):
            obj.A[i] = data[f"A_{i}"]
            obj.b[i] = data[f"b_{i}"]
        obj._updates = int(data["updates"])
        return obj

