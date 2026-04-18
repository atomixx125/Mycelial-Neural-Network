# =============================================================================
# MYCELIAL SNAKE v2 — NEUROEVOLUTION + DQN HYBRID
# =============================================================================
# Run:  python mycelium_snake_v2.py
#
# WHY v1 WAS BAD (and what's fixed):
#
#   PROBLEM 1 — Blind snake:
#     v1 only saw whether the immediate next cell was dangerous (binary).
#     v2 adds RAYCASTING: the snake looks in 8 directions and measures
#     the DISTANCE to the nearest wall, body, and food in each direction.
#     That's 24 distance signals instead of 3 binary flags.
#     Now it can see "wall is 3 cells ahead" not just "wall is/isn't right here".
#
#   PROBLEM 2 — Broken fitness:
#     v1's score² formula rewarded the rare lucky snake that ate food,
#     ignoring the 49 snakes that never ate any. Evolution had almost no
#     gradient signal to work with.
#     v2 adds: survival reward, food-approach reward (getting closer = good),
#     death penalty, and a loop-detection penalty (stops hiding in corners).
#
#   PROBLEM 3 — Pure neuroevolution is sample-inefficient:
#     v1 needed thousands of generations because each snake's entire game
#     only produced ONE fitness number. Very little learning signal.
#     v2 MIXES neuroevolution WITH a DQN-style experience replay buffer.
#     The best snakes from evolution get their experiences stored, and we
#     run backprop on those experiences to fine-tune the best brain.
#     Evolution explores → DQN exploits and fine-tunes.
#
#   PROBLEM 4 — No exploration strategy:
#     v1 mutated randomly. v2 adds epsilon-greedy exploration during
#     DQN training: sometimes take a random action to discover new paths.
#
# ARCHITECTURE:
#   Inputs:  24 raycast distances + 4 direction + 4 food direction = 32
#   Network: 32 → 128 → 64 → 3  (with ReLU, BatchNorm, Dropout)
#   Output:  Q-values for [turn_left, straight, turn_right]
#
# CONTROLS:
#   SPACE  — start / pause
#   F      — toggle fast evolution
#   D      — toggle DQN fine-tuning overlay
#   R      — reset
#   +/-    — demo speed
#   Q/ESC  — quit
# =============================================================================

import pygame
import numpy as np
import sys, time, math, random
from collections import deque

# =============================================================================
# CONFIG
# =============================================================================
GRID         = 20
CELL         = 22
GAME_W       = GRID * CELL
GAME_H       = GRID * CELL
SIDE_W       = 340
BOTTOM_H     = 110
WIN_W        = GAME_W + SIDE_W
WIN_H        = GAME_H + BOTTOM_H

POP_SIZE     = 20       # originally 60 
MAX_STEPS    = 400
MUTATION_RATE= 0.10
MUTATION_STR = 0.25
ELITES       = 6

# DQN settings
REPLAY_SIZE  = 2000      # how many experiences to store
BATCH_SIZE   = 32       # how many to sample per DQN update
GAMMA        = 0.95      # discount factor: how much future reward matters
DQN_LR       = 3e-4      # DQN learning rate
EPSILON_START= 0.3       # initial random action probability
EPSILON_END  = 0.02      # minimum random action probability
EPSILON_DECAY= 0.998     # decay per DQN update step

# Network: 32 inputs → 128 → 64 → 3 outputs
# LAYER_SIZES  = [32, 128, 64, 3]
LAYER_SIZES  = [32, 64, 32, 3]

# Colors
BG          = (5, 8, 14)
PANEL_BG    = (9, 14, 20)
BORDER      = (20, 38, 20)
MYCEL       = (57, 255, 133)
MYCEL_DIM   = (26, 92, 58)
MYCEL_DARK  = (15, 45, 25)
FOOD_COL    = (255, 107, 53)
DANGER_COL  = (255, 45, 85)
GOLD        = (255, 215, 0)
BLUE        = (64, 180, 255)
TEXT_COL    = (200, 240, 208)
DIM_COL     = (74, 106, 80)
RAY_COL     = (30, 70, 30)

# =============================================================================
# MATH PRIMITIVES
# =============================================================================

def relu(x):        return np.maximum(0, x)
def relu_g(x):      return (x > 0).astype(float)
def leaky(x, a=0.01): return np.where(x > 0, x, a * x)
def leaky_g(x, a=0.01): return np.where(x > 0, 1.0, a)

def softmax(x):
    if x.ndim == 1:
        e = np.exp(x - x.max()); return e / e.sum()
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def randn_he(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])


# =============================================================================
# NEURAL NETWORK
# Now outputs Q-values (how good is each action?) instead of just probabilities.
# This matters for DQN: Q-values encode learned value, probabilities just rank.
# =============================================================================

class QNetwork:
    """
    Q-Network: maps state → Q-values for each action.

    Q(state, action) = expected total future reward if we take 'action' now.
    We want to take the action with the HIGHEST Q-value.

    This is the core of Deep Q-Learning (DQN):
    1. Observe state
    2. Compute Q-values for all actions
    3. Take action with highest Q-value (or random with prob epsilon)
    4. Observe reward and next state
    5. Target Q = reward + gamma * max(Q(next_state))
    6. Train network to minimize (predicted Q - target Q)²
    """

    def __init__(self, weights=None):
        self.W  = []  # weight matrices
        self.b  = []  # bias vectors
        self.bn_gamma = []  # batchnorm scale
        self.bn_beta  = []  # batchnorm shift
        self.bn_mean  = []  # running mean (inference)
        self.bn_var   = []  # running variance (inference)

        # Adam optimizer state
        self.mW = []; self.vW = []
        self.mb_adam = []; self.vb = []
        self.mg = []; self.vg = []
        self.mb_bn = []; self.vbeta = []
        self.t = 0

        # Cache for backprop
        self._cache = []

        if weights is not None:
            self._from_flat(weights)
        else:
            for i in range(len(LAYER_SIZES) - 1):
                fan_in  = LAYER_SIZES[i]
                fan_out = LAYER_SIZES[i + 1]
                self.W.append(randn_he((fan_out, fan_in)))
                self.b.append(np.zeros(fan_out))
                self.mW.append(np.zeros((fan_out, fan_in)))
                self.vW.append(np.zeros((fan_out, fan_in)))
                self.mb_adam.append(np.zeros(fan_out))
                self.vb.append(np.zeros(fan_out))

                # BatchNorm for hidden layers only
                if i < len(LAYER_SIZES) - 2:
                    self.bn_gamma.append(np.ones(fan_out))
                    self.bn_beta.append(np.zeros(fan_out))
                    self.bn_mean.append(np.zeros(fan_out))
                    self.bn_var.append(np.ones(fan_out))
                    self.mg.append(np.zeros(fan_out))
                    self.vg.append(np.zeros(fan_out))
                    self.mb_bn.append(np.zeros(fan_out))
                    self.vbeta.append(np.zeros(fan_out))

    def forward(self, x, training=False):
        """
        x: (batch, 32) state vector
        Returns: (batch, 3) Q-values — one per action
        """
        x = np.atleast_2d(x).astype(float)
        self._cache = []
        signal = x
        bn_idx = 0

        for i in range(len(self.W)):
            z     = signal @ self.W[i].T + self.b[i]
            is_last = (i == len(self.W) - 1)

            if not is_last:
                # BatchNorm
                if training:
                    mu  = z.mean(axis=0)
                    var = z.var(axis=0) + 1e-5
                    std = np.sqrt(var)
                    z_n = (z - mu) / std
                    # Update running stats
                    self.bn_mean[bn_idx] = 0.9 * self.bn_mean[bn_idx] + 0.1 * mu
                    self.bn_var[bn_idx]  = 0.9 * self.bn_var[bn_idx]  + 0.1 * var
                else:
                    std = np.sqrt(self.bn_var[bn_idx] + 1e-5)
                    z_n = (z - self.bn_mean[bn_idx]) / std

                z_bn = self.bn_gamma[bn_idx] * z_n + self.bn_beta[bn_idx]
                act  = leaky(z_bn)
                self._cache.append((signal.copy(), z.copy(), z_n.copy(),
                                    std, act.copy(), bn_idx))
                signal  = act
                bn_idx += 1
            else:
                # Last layer: linear (raw Q-values)
                self._cache.append((signal.copy(), z.copy(), None, None, z.copy(), -1))
                signal = z

        return signal  # Q-values

    def backward(self, grad_out, lr=DQN_LR):
        """
        Backprop through all layers.
        grad_out: (batch, 3) gradient of loss w.r.t. Q-values
        """
        self.t += 1
        grad   = grad_out
        b1, b2, eps = 0.9, 0.999, 1e-8
        bn_idx = len(self.bn_gamma) - 1

        for i in reversed(range(len(self.W))):
            inp, z, z_n, std, act, cache_bn = self._cache[i]
            is_last = (i == len(self.W) - 1)

            if not is_last:
                # Leaky relu gradient
                d_act = grad * leaky_g(act)
                # BatchNorm backward
                N = d_act.shape[0]
                d_gamma = (d_act * z_n).sum(axis=0)
                d_beta  = d_act.sum(axis=0)
                d_zn    = d_act * self.bn_gamma[bn_idx]
                d_var   = (-0.5 * d_zn * z_n / std).sum(axis=0)
                d_mu    = (-d_zn / std).sum(axis=0) + d_var * (-2 * (z - z.mean(axis=0))).mean(axis=0)
                d_z     = d_zn / std + d_var * 2 * (z - z.mean(axis=0)) / N + d_mu / N

                # Adam for bn params
                self.mg[bn_idx] = b1 * self.mg[bn_idx] + (1-b1) * d_gamma
                self.vg[bn_idx] = b2 * self.vg[bn_idx] + (1-b2) * d_gamma**2
                mg_h = self.mg[bn_idx] / (1 - b1**self.t)
                vg_h = self.vg[bn_idx] / (1 - b2**self.t)
                self.bn_gamma[bn_idx] -= lr * mg_h / (np.sqrt(vg_h) + eps)

                self.mb_bn[bn_idx] = b1 * self.mb_bn[bn_idx] + (1-b1) * d_beta
                self.vbeta[bn_idx] = b2 * self.vbeta[bn_idx] + (1-b2) * d_beta**2
                mb_h = self.mb_bn[bn_idx] / (1 - b1**self.t)
                vb_h = self.vbeta[bn_idx] / (1 - b2**self.t)
                self.bn_beta[bn_idx] -= lr * mb_h / (np.sqrt(vb_h) + eps)

                bn_idx -= 1
                grad = d_z
            else:
                grad = grad  # linear output, no activation gradient

            # Weight gradient
            dW = grad.T @ inp / grad.shape[0]
            db = grad.mean(axis=0)

            # Adam for weights
            self.mW[i] = b1 * self.mW[i] + (1-b1) * dW
            self.vW[i] = b2 * self.vW[i] + (1-b2) * dW**2
            mW_h = self.mW[i] / (1 - b1**self.t)
            vW_h = self.vW[i] / (1 - b2**self.t)
            self.W[i] -= lr * mW_h / (np.sqrt(vW_h) + eps)

            self.mb_adam[i] = b1 * self.mb_adam[i] + (1-b1) * db
            self.vb[i]      = b2 * self.vb[i]      + (1-b2) * db**2
            mb_h = self.mb_adam[i] / (1 - b1**self.t)
            vb_h = self.vb[i]      / (1 - b2**self.t)
            self.b[i] -= lr * mb_h / (np.sqrt(vb_h) + eps)

            # Pass gradient to previous layer
            grad = grad @ self.W[i]

    def to_flat(self):
        parts = []
        for W, b in zip(self.W, self.b):
            parts.extend([W.flatten(), b])
        for g, be in zip(self.bn_gamma, self.bn_beta):
            parts.extend([g, be])
        return np.concatenate(parts)

    def _from_flat(self, flat):
        self.W = []; self.b = []
        self.mW=[]; self.vW=[]; self.mb_adam=[]; self.vb=[]
        self.bn_gamma=[]; self.bn_beta=[]
        self.bn_mean=[]; self.bn_var=[]
        self.mg=[]; self.vg=[]; self.mb_bn=[]; self.vbeta=[]
        self.t = 0
        self._cache = []
        idx = 0
        for i in range(len(LAYER_SIZES)-1):
            fan_in  = LAYER_SIZES[i]
            fan_out = LAYER_SIZES[i+1]
            sz = fan_out * fan_in
            self.W.append(flat[idx:idx+sz].reshape(fan_out, fan_in)); idx+=sz
            self.b.append(flat[idx:idx+fan_out].copy()); idx+=fan_out
            self.mW.append(np.zeros((fan_out,fan_in)))
            self.vW.append(np.zeros((fan_out,fan_in)))
            self.mb_adam.append(np.zeros(fan_out))
            self.vb.append(np.zeros(fan_out))
            if i < len(LAYER_SIZES)-2:
                self.bn_gamma.append(flat[idx:idx+fan_out].copy()); idx+=fan_out
                self.bn_beta.append(flat[idx:idx+fan_out].copy()); idx+=fan_out
                self.bn_mean.append(np.zeros(fan_out))
                self.bn_var.append(np.ones(fan_out))
                self.mg.append(np.zeros(fan_out))
                self.vg.append(np.zeros(fan_out))
                self.mb_bn.append(np.zeros(fan_out))
                self.vbeta.append(np.zeros(fan_out))

    def copy(self):
        return QNetwork(self.to_flat().copy())

    def soft_update(self, other, tau=0.005):
        """
        Soft update: slowly blend this network's weights toward another's.
        Used for DQN target network — stable targets prevent training collapse.
        theta_target = tau * theta_online + (1-tau) * theta_target
        """
        for i in range(len(self.W)):
            self.W[i] = tau * other.W[i] + (1-tau) * self.W[i]
            self.b[i] = tau * other.b[i] + (1-tau) * self.b[i]

    @property
    def total_weights(self):
        return self.to_flat().shape[0]


# =============================================================================
# EXPERIENCE REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """
    Stores past (state, action, reward, next_state, done) tuples.

    WHY THIS MATTERS:
    Neural networks trained on sequential data (one step after another)
    learn badly because consecutive steps are highly correlated.
    If the snake is turning left, the next 10 states are all "turning left" —
    the network just memorizes the last thing it did.

    Replay buffer fixes this by storing thousands of experiences and
    sampling RANDOM mini-batches to train on. The random mixing breaks
    correlations and makes training stable.

    Same principle as shuffling a deck of cards before dealing.
    """

    def __init__(self, capacity=REPLAY_SIZE):
        self.buffer   = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.copy(),
            action,
            reward,
            next_state.copy(),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states      = np.array([e[0] for e in batch])
        actions     = np.array([e[1] for e in batch], dtype=int)
        rewards     = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones       = np.array([e[4] for e in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

def crossover(pa, pb):
    mask = np.random.rand(len(pa)) < 0.5
    return np.where(mask, pa, pb)

def mutate(w):
    mask  = np.random.rand(len(w)) < MUTATION_RATE
    noise = np.random.randn(len(w)) * MUTATION_STR
    return w + mask * noise

def tournament(pop):
    a = random.choice(pop)
    b = random.choice(pop)
    return a if a.fitness >= b.fitness else b

def randn():
    u, v = 0, 0
    while u == 0: u = random.random()
    while v == 0: v = random.random()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


# =============================================================================
# SNAKE AGENT v2
# Richer inputs: raycasting in 8 directions
# Better fitness function
# =============================================================================

DIRS_4 = [(0,-1),(1,0),(0,1),(-1,0)]   # UP RIGHT DOWN LEFT
DIRS_8 = [                              # 8 directions for raycasting
    (0,-1),(1,-1),(1,0),(1,1),
    (0,1),(-1,1),(-1,0),(-1,-1)
]

class SnakeV2:
    """
    Improved snake agent.

    Key improvements over v1:
    1. RAYCAST INPUTS: looks in 8 directions, measures distance to wall,
       body, and food. 8 directions × 3 things = 24 inputs (vs 3 binary).
    2. BETTER FITNESS: survival + food approach + penalties for loops.
    3. LOOP DETECTION: tracks last N positions, penalizes revisiting.
    """

    def __init__(self, brain=None):
        self.brain   = brain or QNetwork()
        self.fitness = 0.0
        self.reset()

    def reset(self):
        cx, cy = GRID // 2, GRID // 2
        self.body           = [(cx, cy), (cx-1, cy), (cx-2, cy)]
        self.dir_idx        = 1   # RIGHT
        self.alive          = True
        self.score          = 0
        self.steps          = 0
        self.steps_to_food  = MAX_STEPS
        self.fitness        = 0.0
        self.food           = self._spawn_food()
        self.last_acts      = None
        self.thoughts       = []
        self.ray_hits       = {}      # for visualization
        # Loop detection: store recent head positions
        self.recent_pos     = deque(maxlen=40)
        self.prev_food_dist = self._food_dist()
        self.total_reward   = 0.0

    def _spawn_food(self):
        while True:
            f = (random.randint(0, GRID-1), random.randint(0, GRID-1))
            if f not in self.body:
                return f

    def _food_dist(self):
        hx, hy = self.body[0]
        fx, fy = self.food
        return abs(hx - fx) + abs(hy - fy)  # Manhattan distance

    # ── RAYCASTING INPUTS ─────────────────────────────────────────────────
    # Cast a ray in each of 8 directions. For each ray, measure:
    #   - distance to wall (normalized 0-1)
    #   - 1 if body segment hit, else 0
    #   - 1 if food hit, else 0
    # Total: 8 × 3 = 24 values
    # Plus: 4 direction one-hot + 4 food direction = 32 total
    def get_inputs(self):
        hx, hy  = self.body[0]
        body_set = set(self.body[1:])  # exclude head
        inputs   = []
        self.ray_hits = {}

        for ddx, ddy in DIRS_8:
            dist_wall  = 0.0
            hit_body   = 0.0
            hit_food   = 0.0
            x, y = hx + ddx, hy + ddy
            steps = 1

            while 0 <= x < GRID and 0 <= y < GRID:
                if (x, y) in body_set and hit_body == 0.0:
                    hit_body = 1.0 / steps  # closer = stronger signal
                if (x, y) == self.food and hit_food == 0.0:
                    hit_food = 1.0
                x += ddx
                y += ddy
                steps += 1

            # Normalize wall distance: 1 = immediately adjacent, 0 = far
            # Wall distance = how many steps until we exit the grid
            wall_dist = 1.0 / max(steps - 1, 1)
            inputs.extend([wall_dist, hit_body, hit_food])
            self.ray_hits[(ddx, ddy)] = (steps - 1, hit_body > 0, hit_food > 0)

        # Direction one-hot (4)
        inputs.extend([
            1.0 if self.dir_idx == 0 else 0.0,
            1.0 if self.dir_idx == 1 else 0.0,
            1.0 if self.dir_idx == 2 else 0.0,
            1.0 if self.dir_idx == 3 else 0.0,
        ])

        # Food direction (4)
        fx, fy = self.food
        inputs.extend([
            1.0 if fy < hy else 0.0,
            1.0 if fy > hy else 0.0,
            1.0 if fx < hx else 0.0,
            1.0 if fx > hx else 0.0,
        ])

        return np.array(inputs, dtype=float)

    def step(self, epsilon=0.0):
        """
        One step of the game.
        epsilon: probability of taking a random action (for DQN exploration)
        Returns: (state, action, reward, next_state, done) for DQN training
        """
        if not self.alive:
            return None

        state  = self.get_inputs()
        q_vals = self.brain.forward(state, training=False)[0]

        # Epsilon-greedy: explore randomly with probability epsilon
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = int(np.argmax(q_vals))

        # Store activations (for viz) — only on deterministic steps
        self.last_acts = self.brain.forward(state, training=False)

        # Build thoughts
        action_names = ['TURN LEFT', 'GO STRAIGHT', 'TURN RIGHT']
        self.thoughts = []
        self.thoughts.append((f'Q: L={q_vals[0]:.2f} S={q_vals[1]:.2f} R={q_vals[2]:.2f}', DIM_COL))
        self.thoughts.append((f'→ {action_names[action]}', MYCEL))

        # Apply action
        if action == 0: self.dir_idx = (self.dir_idx + 3) % 4
        if action == 2: self.dir_idx = (self.dir_idx + 1) % 4

        dx, dy = DIRS_4[self.dir_idx]
        hx, hy = self.body[0]
        nx, ny = hx + dx, hy + dy

        # ── REWARD FUNCTION ──────────────────────────────────────────────
        # This is the heart of DQN. The snake learns to maximize total reward.
        # Reward shaping: give small intermediate rewards to guide learning.
        reward = 0.0

        # Collision → die → big negative reward
        hit_wall = (nx < 0 or ny < 0 or nx >= GRID or ny >= GRID)
        hit_self = (nx, ny) in set(self.body)

        if hit_wall or hit_self:
            self.alive = False
            reward = -15.0  # death penalty
            self._calc_fitness()
            next_state = state  # terminal
            return state, action, reward, next_state, True

        self.body.insert(0, (nx, ny))

        # Ate food → large positive reward
        if (nx, ny) == self.food:
            self.score         += 1
            self.steps_to_food  = MAX_STEPS + self.score * 30
            reward              = 15.0
            self.food           = self._spawn_food()
            self.prev_food_dist = self._food_dist()
        else:
            self.body.pop()
            self.steps_to_food -= 1

            # Getting closer to food → small positive reward
            new_dist = self._food_dist()
            if new_dist < self.prev_food_dist:
                reward = 0.8
            elif new_dist > self.prev_food_dist:
                reward = -0.6   # moving away → small negative
            self.prev_food_dist = new_dist

            # Starvation → death
            if self.steps_to_food <= 0:
                self.alive = False
                reward     = -10.0
                self._calc_fitness()
                next_state = self.get_inputs()
                return state, action, reward, next_state, True

        # Loop penalty: if revisiting recent position too much
        self.recent_pos.append((nx, ny))
        loop_count = sum(1 for p in self.recent_pos if p == (nx, ny))
        if loop_count > 3:
            reward -= 0.5 * loop_count  # escalating penalty for looping

        self.steps        += 1
        self.total_reward += reward
        next_state         = self.get_inputs()

        return state, action, reward, next_state, False

    def _calc_fitness(self):
        """
        Better fitness function:
        - Food is still the main goal (exponential reward)
        - Total reward integrates all the shaped rewards during the game
        - Survival matters but less than food
        """
        food_reward    = (self.score ** 2) * 400 + self.score * 50
        survival_bonus = self.steps * 0.3
        shaped_reward  = max(self.total_reward * 10, 0)
        self.fitness   = food_reward + survival_bonus + shaped_reward

    def run_to_death(self):
        while self.alive:
            self.step()


# =============================================================================
# DQN TRAINER
# Fine-tunes the best evolved brain using experience replay + backprop
# =============================================================================

class DQNTrainer:
    """
    Deep Q-Network training loop.

    Uses the best evolved brain as starting point, then improves it
    with actual gradient descent using stored experiences.

    Two networks:
    - online_net:  the one being trained
    - target_net:  a slowly-updated copy used to compute stable targets

    WHY TWO NETWORKS?
    If we update targets and predictions with the same network simultaneously,
    the targets keep moving — like chasing a moving target. This causes
    training to oscillate or collapse.
    The target network is updated slowly (soft update each step) to provide
    stable training targets.
    """

    def __init__(self, brain=None):
        self.online_net = brain or QNetwork()
        self.target_net = self.online_net.copy()  # starts identical
        self.replay     = ReplayBuffer()
        self.epsilon    = EPSILON_START
        self.steps      = 0
        self.losses     = []

    def collect_experience(self, n_episodes=5):
        """
        Play n_episodes with the current online network (+ epsilon exploration)
        and store experiences in the replay buffer.
        """
        total_score = 0
        for _ in range(n_episodes):
            snake = SnakeV2(self.online_net.copy())
            snake.reset()
            while snake.alive:
                result = snake.step(epsilon=self.epsilon)
                if result:
                    state, action, reward, next_state, done = result
                    self.replay.push(state, action, reward, next_state, done)
            total_score += snake.score
        return total_score / n_episodes

    def train_step(self):
        """
        One DQN update step:
        1. Sample random batch from replay buffer
        2. Compute target Q-values using target network
        3. Compute predicted Q-values using online network
        4. Backprop the difference (TD error)
        5. Soft-update target network
        """
        if len(self.replay) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)

        # Predict Q-values for current states
        q_pred = self.online_net.forward(states, training=True)   # (batch, 3)

        # Compute target Q-values using target network
        # TARGET: r + gamma * max_a'(Q_target(s', a'))    if not done
        #         r                                        if done
        with_target = self.target_net.forward(next_states, training=False)
        max_q_next  = with_target.max(axis=1)                     # (batch,)
        targets     = rewards + GAMMA * max_q_next * (1 - dones)  # (batch,)

        # Build target Q matrix: only update the Q-value for the taken action
        q_targets         = q_pred.copy()
        q_targets[np.arange(BATCH_SIZE), actions] = targets

        # Loss: MSE between predicted and target Q-values
        loss = np.mean((q_pred - q_targets) ** 2)

        # Backprop
        grad = 2.0 * (q_pred - q_targets) / BATCH_SIZE
        self.online_net.backward(grad, lr=DQN_LR)

        # Soft update target network (slowly track online network)
        self.target_net.soft_update(self.online_net, tau=0.005)

        # Decay epsilon (less random over time as we learn)
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        self.steps += 1
        self.losses.append(loss)
        return loss


# =============================================================================
# EVOLUTION ENGINE
# =============================================================================

class Evolution:
    def __init__(self):
        self.generation   = 0
        self.population   = [SnakeV2() for _ in range(POP_SIZE)]
        self.best_ever    = None
        self.best_fitness = 0.0
        self.history      = []

    def run_generation(self):
        for s in self.population: s.reset(); s.run_to_death()
        self.population.sort(key=lambda s: s.fitness, reverse=True)

        best  = self.population[0]
        avg_f = np.mean([s.fitness for s in self.population])

        if best.fitness > self.best_fitness:
            self.best_fitness = best.fitness
            self.best_ever    = SnakeV2(best.brain.copy())

        self.history.append({
            'gen': self.generation, 'best': best.fitness,
            'avg': avg_f, 'score': best.score
        })

        # Breed
        next_pop = [SnakeV2(self.population[i].brain.copy())
                    for i in range(ELITES)]
        while len(next_pop) < POP_SIZE:
            pa = tournament(self.population)
            pb = tournament(self.population)
            child_w = mutate(crossover(pa.brain.to_flat(), pb.brain.to_flat()))
            next_pop.append(SnakeV2(QNetwork(child_w)))

        self.population = next_pop
        self.generation += 1
        return best.fitness, avg_f, best.score


# =============================================================================
# RENDERER
# =============================================================================

class Renderer:
    def __init__(self, screen, fonts):
        self.screen = screen
        self.f_sm, self.f_md = fonts

    def draw_game(self, snake, evo, dqn_trainer):
        pygame.draw.rect(self.screen, BG, (0, 0, GAME_W, GAME_H))
        for i in range(GRID + 1):
            c = (12, 22, 12)
            pygame.draw.line(self.screen, c, (i*CELL, 0), (i*CELL, GAME_H))
            pygame.draw.line(self.screen, c, (0, i*CELL), (GAME_W, i*CELL))

        if not snake: return

        # Draw raycasts
        hx, hy = snake.body[0]
        for (ddx, ddy), (dist, hit_body, hit_food) in snake.ray_hits.items():
            end_x = (hx + ddx * dist) * CELL + CELL // 2
            end_y = (hy + ddy * dist) * CELL + CELL // 2
            col = FOOD_COL if hit_food else (DANGER_COL if hit_body else RAY_COL)
            pygame.draw.line(self.screen, col,
                (hx*CELL+CELL//2, hy*CELL+CELL//2), (int(end_x), int(end_y)), 1)

        # Food
        fx, fy = snake.food
        cx, cy = fx*CELL+CELL//2, fy*CELL+CELL//2
        for r, a in [(CELL//2, 25), (CELL//3, 50)]:
            s = pygame.Surface((r*2,r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*FOOD_COL, a), (r,r), r)
            self.screen.blit(s, (cx-r, cy-r))
        pygame.draw.circle(self.screen, FOOD_COL, (cx, cy), CELL//4)

        # Body
        for i, (bx, by) in enumerate(snake.body):
            t    = 1.0 - i / max(len(snake.body), 1)
            rect = pygame.Rect(bx*CELL+1, by*CELL+1, CELL-2, CELL-2)
            if i == 0:
                pygame.draw.rect(self.screen, MYCEL, rect, border_radius=3)
                ex, ey = bx*CELL+CELL//2, by*CELL+CELL//2
                pygame.draw.circle(self.screen, BG, (ex-3, ey-2), 2)
                pygame.draw.circle(self.screen, BG, (ex+3, ey-2), 2)
            else:
                g = int(t*160+30)
                pygame.draw.rect(self.screen, (0,g,40), rect, border_radius=2)

        # HUD
        eps_str = f'ε={dqn_trainer.epsilon:.3f}' if dqn_trainer else ''
        hud = self.f_sm.render(
            f'GEN {evo.generation}  SCORE {snake.score}  BEST {evo.history[-1]["score"] if evo.history else 0}  {eps_str}',
            True, MYCEL_DIM)
        self.screen.blit(hud, (6, 4))

    def draw_panel(self, snake, evo, dqn_trainer):
        x0 = GAME_W
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, SIDE_W, WIN_H))
        pygame.draw.line(self.screen, BORDER, (x0, 0), (x0, WIN_H), 1)

        # Title
        t = self.f_md.render('MYCELIAL BRAIN v2', True, MYCEL)
        self.screen.blit(t, (x0 + SIDE_W//2 - t.get_width()//2, 6))

        # Network
        self._draw_net(snake, x0, 28, SIDE_W, 220)

        # Heatmap
        self._draw_heat(evo, x0, 258, SIDE_W, 24)

        # Stats
        y = 298
        dqn_loss = f'{dqn_trainer.losses[-1]:.4f}' if dqn_trainer and dqn_trainer.losses else '—'
        stats = [
            ('GENERATION',     str(evo.generation)),
            ('BEST FITNESS',   f'{evo.best_fitness:.0f}'),
            ('REPLAY BUFFER',  str(len(dqn_trainer.replay)) if dqn_trainer else '—'),
            ('DQN LOSS',       dqn_loss),
            ('EPSILON',        f'{dqn_trainer.epsilon:.3f}' if dqn_trainer else '—'),
            ('WEIGHTS',        str(QNetwork().total_weights)),
        ]
        for label, val in stats:
            ls = self.f_sm.render(label, True, DIM_COL)
            vs = self.f_sm.render(val, True, MYCEL)
            self.screen.blit(ls, (x0+8, y))
            self.screen.blit(vs, (x0+SIDE_W-vs.get_width()-8, y))
            y += 16

        # Thoughts
        y += 6
        t = self.f_sm.render('SNAKE THOUGHTS', True, MYCEL_DIM)
        self.screen.blit(t, (x0+8, y)); y += 14
        for text, color in (snake.thoughts if snake else []):
            ts = self.f_sm.render(text, True, color)
            self.screen.blit(ts, (x0+8, y)); y += 14
            if y > WIN_H - BOTTOM_H - 20: break

        # Controls
        ctrl = self.f_sm.render('SPC=pause F=fast D=dqn R=reset +/-=spd Q=quit', True, (35,55,35))
        self.screen.blit(ctrl, (x0+4, WIN_H - BOTTOM_H - 14))

    def _draw_net(self, snake, x0, y0, W, H):
        if not snake or snake.last_acts is None: return
        acts        = snake.last_acts
        layer_count = len(LAYER_SIZES)
        margin      = 12
        positions   = []

        for l, n in enumerate(LAYER_SIZES):
            lx = x0 + margin + (l/(layer_count-1))*(W-2*margin)
            col = []
            for i in range(n):
                ly = y0 + margin + ((i+1)/(n+1))*(H-2*margin)
                col.append((int(lx), int(ly)))
            positions.append(col)

        for l in range(layer_count-1):
            Wmat = snake.brain.W[l]
            for j in range(min(LAYER_SIZES[l+1], 12)):
                for i in range(min(LAYER_SIZES[l], 16)):
                    w = Wmat[j, i]
                    if abs(w) < 0.35: continue
                    s = min(abs(w)/3.0, 1.0)
                    c = tuple(int(x*s*0.5) for x in (MYCEL if w>0 else DANGER_COL))
                    pygame.draw.line(self.screen, c, positions[l][i], positions[l+1][j], 1)

        for l, lpos in enumerate(positions):
            layer_acts = acts[0] if l < len(acts) else []
            for i, (nx, ny) in enumerate(lpos):
                a = float(np.clip(layer_acts[i] if i < len(layer_acts) else 0, 0, 1))
                r = 3 + int(a*4)
                g_val = int(a*200+55)
                pygame.draw.circle(self.screen, (0,g_val,int(a*60+20)), (nx,ny), r)

    def _draw_heat(self, evo, x0, y0, W, H):
        if not evo.best_ever: return
        lbl = self.f_sm.render('WEIGHT HEATMAP', True, DIM_COL)
        self.screen.blit(lbl, (x0+8, y0))
        y0 += 12
        flat  = evo.best_ever.brain.to_flat()
        total = len(flat)
        bw    = W - 16
        cw    = bw / total
        for i, w in enumerate(flat):
            n  = math.tanh(w)
            px = int(x0+8+i*cw)
            pw = max(1, int(cw)+1)
            c  = (0, int(n*200+55), 0) if n>0 else (int(-n*200+55), 0, 0)
            pygame.draw.rect(self.screen, c, (px, y0, pw, H))

    def draw_chart(self, evo, dqn_trainer):
        y0 = GAME_H
        pygame.draw.rect(self.screen, (6,10,6), (0, y0, WIN_W, BOTTOM_H))
        pygame.draw.line(self.screen, BORDER, (0,y0), (WIN_W,y0), 1)

        title = self.f_sm.render(
            f'EVOLUTION FITNESS  |  DQN LOSS (blue)  |  GEN {evo.generation}',
            True, DIM_COL)
        self.screen.blit(title, (6, y0+3))

        h = evo.history
        chart_h = BOTTOM_H - 22
        chart_w = WIN_W - 12

        if len(h) >= 2:
            max_f = max(max(e['best'] for e in h), 1)
            def tp(idx, val):
                px = int(6 + (idx/(len(h)-1)) * chart_w)
                py = int(y0 + 18 + chart_h - (val/max_f) * chart_h)
                return (px, py)
            avg_pts  = [tp(i, e['avg'])  for i, e in enumerate(h)]
            best_pts = [tp(i, e['best']) for i, e in enumerate(h)]
            if len(avg_pts)  > 1: pygame.draw.lines(self.screen, MYCEL_DIM, False, avg_pts,  1)
            if len(best_pts) > 1: pygame.draw.lines(self.screen, MYCEL,     False, best_pts, 2)

        # DQN loss (right side, blue line)
        if dqn_trainer and len(dqn_trainer.losses) >= 2:
            losses  = dqn_trainer.losses[-chart_w:]
            max_l   = max(max(losses), 0.01)
            dqn_pts = []
            for i, l in enumerate(losses):
                px = int(WIN_W//2 + (i/(len(losses)-1))*(WIN_W//2-12))
                py = int(y0 + 18 + chart_h - (l/max_l)*chart_h)
                dqn_pts.append((px, py))
            if len(dqn_pts) > 1:
                pygame.draw.lines(self.screen, BLUE, False, dqn_pts, 1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    pygame.init()
    pygame.display.set_caption('🍄 Mycelial Snake v2 — Neuroevolution + DQN')
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    try:
        f_sm = pygame.font.SysFont('couriernew', 11)
        f_md = pygame.font.SysFont('couriernew', 13, bold=True)
    except:
        f_sm = pygame.font.SysFont('monospace', 11)
        f_md = pygame.font.SysFont('monospace', 13)

    renderer = Renderer(screen, (f_sm, f_md))

    evo         = Evolution()
    dqn         = DQNTrainer()
    demo        = None
    running     = False
    paused      = False
    fast_mode   = True
    dqn_active  = True
    demo_fps    = 12
    best_score  = 0

    last_demo   = time.time()
    last_evo    = time.time()

    total_w = QNetwork().total_weights
    print(f"""
╔══════════════════════════════════════════════════════╗
║    🍄  MYCELIAL SNAKE v2 — NEURO + DQN HYBRID       ║
╠══════════════════════════════════════════════════════╣
║  Inputs:       32 (raycast × 8 directions)           ║
║  Architecture: {' → '.join(str(x) for x in LAYER_SIZES):<38}║
║  Weights:      {total_w:<38}║
║  Evolution:    {POP_SIZE} snakes, {ELITES} elites              ║
║  DQN:          replay={REPLAY_SIZE}, batch={BATCH_SIZE}, γ={GAMMA}      ║
╠══════════════════════════════════════════════════════╣
║  SPACE=start/pause  F=fast  D=toggle DQN             ║
║  R=reset  +/-=speed  Q=quit                          ║
╚══════════════════════════════════════════════════════╝
    """)

    def refresh_demo():
        nonlocal demo
        brain = dqn.online_net if dqn_active else (evo.best_ever.brain if evo.best_ever else None)
        if brain:
            demo = SnakeV2(brain.copy())
            demo.reset()

    while True:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif event.key == pygame.K_SPACE:
                    if not running:
                        running = True
                        print('\n🌱 Evolution + DQN started...\n')
                    else:
                        paused = not paused
                elif event.key == pygame.K_f:
                    fast_mode = not fast_mode
                    print(f'⚡ Fast: {"ON" if fast_mode else "OFF"}')
                elif event.key == pygame.K_d:
                    dqn_active = not dqn_active
                    print(f'🧬 DQN: {"ON" if dqn_active else "OFF"}')
                    refresh_demo()
                elif event.key == pygame.K_r:
                    evo=Evolution(); dqn=DQNTrainer()
                    demo=None; running=False; paused=False; best_score=0
                    print('↺ Reset')
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    demo_fps = min(demo_fps+2, 60)
                elif event.key == pygame.K_MINUS:
                    demo_fps = max(demo_fps-2, 1)

        if running and not paused:
            now = time.time()

            # Evolution
            gens_per_frame = 3 if fast_mode else 1
            if fast_mode or (now - last_evo > 0.3):
                for _ in range(gens_per_frame):
                    best_f, avg_f, score = evo.run_generation()
                    best_score = max(best_score, score)
                    if evo.generation % 5 == 0:
                        print(f'  Gen {evo.generation:>4} | fit={best_f:.0f} | score={score} | ε={dqn.epsilon:.3f}')
                last_evo = now

            # DQN fine-tuning (runs alongside evolution)
            if dqn_active:
                # Seed DQN with best evolved brain periodically
                if evo.best_ever and evo.generation % 15 == 0 and evo.generation > 0:
                    dqn.online_net = evo.best_ever.brain.copy()
                    dqn.target_net = dqn.online_net.copy()

                # Collect experiences and train
                for _ in range(2 if fast_mode else 1):
                    dqn.collect_experience(n_episodes=3)
                    for _ in range(8):
                        dqn.train_step()

            # Refresh demo
            if demo is None and (evo.best_ever or dqn_active):
                refresh_demo()
            if evo.generation % 10 == 0:
                refresh_demo()

        # Demo step
        if demo and running and not paused:
            now = time.time()
            if now - last_demo >= 1.0 / demo_fps:
                result = demo.step(epsilon=0.0)
                last_demo = now
                if not demo.alive:
                    refresh_demo()

        # Draw
        screen.fill(BG)
        renderer.draw_game(demo, evo, dqn)
        renderer.draw_panel(demo, evo, dqn)
        renderer.draw_chart(evo, dqn)

        if not running:
            overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            overlay.fill((5,8,14,210))
            screen.blit(overlay, (0,0))
            lines = [
                '🍄 MYCELIAL SNAKE v2',
                '',
                'Press SPACE to start',
                '',
                'Neuroevolution (60 snakes)',
                '+ DQN experience replay',
                '+ Raycast vision (8 dirs)',
                '+ Better reward shaping',
                '',
                f'Network: {" → ".join(str(x) for x in LAYER_SIZES)}',
                f'Weights: {total_w}',
            ]
            y = GAME_H//2 - len(lines)*12
            for line in lines:
                col = MYCEL if ('🍄' in line or 'SPACE' in line) else DIM_COL
                s = f_md.render(line, True, col)
                screen.blit(s, (GAME_W//2 - s.get_width()//2, y))
                y += 22

        if fast_mode and running:
            fm = f_sm.render('⚡ FAST MODE', True, GOLD)
            screen.blit(fm, (6, GAME_H-18))
        if dqn_active and running:
            dqn_ind = f_sm.render(f'DQN ACTIVE  steps={dqn.steps}', True, BLUE)
            screen.blit(dqn_ind, (80, GAME_H-18))

        pygame.display.flip()


if __name__ == '__main__':
    main()
