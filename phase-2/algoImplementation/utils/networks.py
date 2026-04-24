"""
utils/networks.py
-----------------
Pure-NumPy neural network building blocks.
Identical to Phase 1 — no changes needed; the MLP is dimension-agnostic.

Implements:
  - Linear layer  (weights + bias, He initialisation)
  - ReLU / Softmax / Tanh activations
  - MLP (multi-layer perceptron) forward + backward pass
  - Adam optimiser (Kingma & Ba 2015)
  - Circular ReplayBuffer for DQN / DDPG
"""

import numpy as np


# ── Activations ───────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)   # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


# ── Linear Layer ──────────────────────────────────────────────────────────────

class Linear:
    """Fully-connected layer: out = x @ W + b  (He initialisation)."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng   = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W  = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self._x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x.ndim == 1:
            self.dW = np.outer(self._x, grad_out)
            self.db = grad_out
            return grad_out @ self.W.T
        else:
            self.dW = self._x.T @ grad_out
            self.db = grad_out.sum(axis=0)
            return grad_out @ self.W.T

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP:
    """Multi-layer perceptron with ReLU hidden layers.

    Architecture: [in_dim] → hidden_sizes → [out_dim]
    Output activation applied externally.
    """

    def __init__(self, in_dim: int, hidden_sizes: list, out_dim: int, seed: int = 0):
        sizes       = [in_dim] + hidden_sizes + [out_dim]
        self.layers = [Linear(sizes[i], sizes[i+1], seed + i)
                       for i in range(len(sizes) - 1)]
        self._pre_activations = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._pre_activations = []
        out = x.astype(np.float32)
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            self._pre_activations.append(out.copy())
            if i < len(self.layers) - 1:
                out = relu(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out.astype(np.float32)
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                grad = grad * relu_grad(self._pre_activations[i])
            grad = self.layers[i].backward(grad)
        return grad

    @property
    def params(self):
        p = []
        for l in self.layers: p.extend(l.params)
        return p

    @property
    def grads(self):
        g = []
        for l in self.layers: g.extend(l.grads)
        return g


# ── Adam Optimiser ────────────────────────────────────────────────────────────

class Adam:
    """Adam optimiser (Kingma & Ba 2015)."""

    def __init__(self, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m     = []
        self.v     = []

    def init(self, params: list):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params: list, grads: list):
        if not self.m:
            self.init(params)
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat     = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat     = self.v[i] / (1 - self.beta2 ** self.t)
            p        -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular replay buffer for DQN / DDPG."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity    = capacity
        self.ptr         = 0
        self.size        = 0
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity,              dtype=np.int32)
        self.rewards     = np.zeros(capacity,              dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity,              dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size
