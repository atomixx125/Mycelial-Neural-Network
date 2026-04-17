# =============================================================================
# MYCELIAL SNAKE — NEUROEVOLUTION IN PYTHON
# =============================================================================
# Run this with:  python mycelium_snake.py
#
# WHAT THIS IS:
#   A snake game where the snake's brain IS a neural network.
#   The network learns to play using NEUROEVOLUTION — a genetic algorithm.
#   No backpropagation. No calculus at runtime. Just survival of the fittest.
#
# HOW IT WORKS IN PLAIN ENGLISH:
#   1. Spawn 50 snakes, each with a random neural network brain
#   2. Let all 50 play simultaneously (fast, invisible)
#   3. The ones that ate the most food = highest "fitness"
#   4. Those snakes reproduce: copy their weights + add small random mutations
#   5. New generation plays. Repeat. Watch it get good over ~50-100 generations.
#
# WHAT YOU'LL SEE:
#   - Left panel:  the best snake playing in real time
#   - Right panel: the live neural network with glowing activations
#   - Bottom:      fitness chart showing learning over generations
#   - Sidebar:     generation log, thoughts, weight heatmap
#
# CONTROLS:
#   SPACE  — pause / resume
#   F      — toggle fast evolution mode
#   R      — reset everything
#   +/-    — speed up / slow down demo
#   Q/ESC  — quit
# =============================================================================

import pygame
import numpy as np
import sys
import time
import math
import random
from collections import deque

# =============================================================================
# CONFIG
# =============================================================================

GRID       = 20          # grid squares per side
CELL       = 22          # pixels per cell
GAME_W     = GRID * CELL # game area width  (440)
GAME_H     = GRID * CELL # game area height (440)
SIDE_W     = 320         # right panel width
BOTTOM_H   = 100         # bottom chart height
WIN_W      = GAME_W + SIDE_W
WIN_H      = GAME_H + BOTTOM_H

POP_SIZE        = 50     # snakes per generation
MAX_STEPS       = 300    # steps before snake starves
MUTATION_RATE   = 0.12   # probability a weight mutates
MUTATION_STR    = 0.3    # how much weights shift on mutation
ELITES          = 5      # top snakes survive unchanged

# Neural network shape: 11 inputs → 16 hidden → 12 hidden → 3 outputs
LAYER_SIZES = [11, 16, 12, 3]

# Colors (dark mycelium theme)
BG          = (6,   9,  16)
PANEL_BG    = (11,  16,  24)
BORDER      = (26,  42,  26)
MYCEL       = (57, 255, 133)
MYCEL_DIM   = (26,  92,  58)
FOOD_COL    = (255, 107,  53)
DANGER_COL  = (255,  45,  85)
GOLD        = (255, 215,   0)
TEXT_COL    = (200, 240, 208)
DIM_COL     = (74,  106,  80)
WHITE       = (255, 255, 255)

# =============================================================================
# MATH PRIMITIVES
# (same concepts as our previous network — relu, softmax)
# =============================================================================

def relu(x):
    """Pass signal through only if positive. Dead simple, fast."""
    return np.maximum(0, x)

def softmax(x):
    """
    Turn raw output scores into probabilities (sum to 1).
    We subtract max for numerical stability (prevents overflow).
    The highest score gets the highest probability.
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()

def randn_scaled(shape, scale):
    """
    Xavier/He initialization: random weights scaled by sqrt(2/inputs).
    Keeps signals from exploding or vanishing through layers.
    """
    return np.random.randn(*shape) * scale

# =============================================================================
# NEURAL NETWORK (THE SNAKE'S BRAIN)
# =============================================================================

class MycelialBrain:
    """
    A feedforward neural network.
    Architecture: 11 → 16 → 12 → 3

    11 inputs  = what the snake senses (danger, direction, food location)
     3 outputs = turn left | go straight | turn right

    Unlike the previous file, we DON'T use backpropagation here.
    Weights are adjusted by EVOLUTION (mutation + selection), not gradients.
    """

    def __init__(self, weights=None):
        if weights is not None:
            self._from_flat(weights)
        else:
            # Build weight matrices and bias vectors for each layer
            self.W = []  # list of weight matrices
            self.b = []  # list of bias vectors
            for i in range(len(LAYER_SIZES) - 1):
                fan_in  = LAYER_SIZES[i]
                fan_out = LAYER_SIZES[i + 1]
                scale   = math.sqrt(2.0 / fan_in)  # He init for relu
                self.W.append(randn_scaled((fan_out, fan_in), scale))
                self.b.append(np.zeros(fan_out))

    def forward(self, x):
        """
        Pass inputs through all layers.
        Returns final softmax probabilities for [turn_left, straight, turn_right].
        """
        signal = np.array(x, dtype=float)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ signal + b          # weighted sum: (fan_out,) vector
            if i < len(self.W) - 1:
                signal = relu(z)        # relu for hidden layers
            else:
                signal = softmax(z)     # softmax for output layer
        return signal

    def forward_verbose(self, x):
        """
        Same as forward but returns ALL layer activations.
        Used for the live network visualization.
        """
        activations = [np.array(x, dtype=float)]
        signal = activations[0].copy()
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ signal + b
            if i < len(self.W) - 1:
                signal = relu(z)
            else:
                signal = softmax(z)
            activations.append(signal.copy())
        return activations

    def to_flat(self):
        """Flatten all weights and biases into one 1D array."""
        parts = []
        for W, b in zip(self.W, self.b):
            parts.append(W.flatten())
            parts.append(b)
        return np.concatenate(parts)

    def _from_flat(self, flat):
        """Reconstruct weight matrices from a flat array."""
        self.W = []
        self.b = []
        idx = 0
        for i in range(len(LAYER_SIZES) - 1):
            fan_in  = LAYER_SIZES[i]
            fan_out = LAYER_SIZES[i + 1]
            size_W  = fan_out * fan_in
            self.W.append(flat[idx:idx + size_W].reshape(fan_out, fan_in))
            idx += size_W
            self.b.append(flat[idx:idx + fan_out].copy())
            idx += fan_out

    def copy(self):
        return MycelialBrain(self.to_flat().copy())

    @property
    def total_weights(self):
        return self.to_flat().shape[0]


# =============================================================================
# GENETIC ALGORITHM OPERATIONS
# These replace backpropagation as the learning mechanism.
# =============================================================================

def crossover(parent_a, parent_b):
    """
    Combine two parent weight arrays into one child.
    Each weight is inherited from either parent A or parent B.
    Mimics biological sexual reproduction / DNA recombination.
    """
    mask  = np.random.rand(len(parent_a)) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return child

def mutate(weights):
    """
    Randomly perturb some weights.
    Gaussian noise added to MUTATION_RATE fraction of all weights.
    This is the source of NEW behaviors — without it, evolution stagnates.
    """
    mask   = np.random.rand(len(weights)) < MUTATION_RATE
    noise  = np.random.randn(len(weights)) * MUTATION_STR
    return weights + mask * noise

def tournament_select(population):
    """
    Pick two random snakes, return the fitter one.
    Simple but effective selection pressure — good snakes reproduce more.
    """
    a = random.choice(population)
    b = random.choice(population)
    return a if a.fitness >= b.fitness else b


# =============================================================================
# SNAKE AGENT
# One snake = one brain + game state
# =============================================================================

# Cardinal directions: UP, RIGHT, DOWN, LEFT
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class SnakeAgent:
    """
    One snake. Carries a neural network brain.
    On each step: sense → think → act → update state.
    """

    def __init__(self, brain=None):
        self.brain   = brain or MycelialBrain()
        self.fitness = 0.0
        self.reset()

    def reset(self):
        cx, cy = GRID // 2, GRID // 2
        self.body          = [(cx, cy), (cx - 1, cy)]
        self.dir_idx       = 1   # start moving RIGHT
        self.alive         = True
        self.score         = 0
        self.steps         = 0
        self.steps_to_food = MAX_STEPS
        self.fitness       = 0.0
        self.food          = self._spawn_food()
        self.last_acts     = None   # for visualization
        self.thoughts      = []     # for thought display

    def _spawn_food(self):
        while True:
            f = (random.randint(0, GRID - 1), random.randint(0, GRID - 1))
            if f not in self.body:
                return f

    # ── 11 SENSORY INPUTS ──────────────────────────────────────────────────
    # This is what the snake "sees". We encode the world as 11 numbers (0 or 1).
    def get_inputs(self):
        hx, hy = self.body[0]
        dx, dy = DIRS[self.dir_idx]

        # Relative left and right directions
        left_idx  = (self.dir_idx + 3) % 4
        right_idx = (self.dir_idx + 1) % 4
        lx, ly    = DIRS[left_idx]
        rx, ry    = DIRS[right_idx]

        body_set = set(self.body)

        def danger(x, y):
            # Is this cell a wall or body segment?
            return 1.0 if (x < 0 or y < 0 or x >= GRID or y >= GRID
                           or (x, y) in body_set) else 0.0

        fx, fy = self.food

        return [
            # DANGER (3 signals: straight, left, right)
            danger(hx + dx,  hy + dy),   # straight ahead
            danger(hx + lx,  hy + ly),   # to the left
            danger(hx + rx,  hy + ry),   # to the right

            # CURRENT DIRECTION (one-hot, 4 signals)
            1.0 if self.dir_idx == 0 else 0.0,  # up
            1.0 if self.dir_idx == 1 else 0.0,  # right
            1.0 if self.dir_idx == 2 else 0.0,  # down
            1.0 if self.dir_idx == 3 else 0.0,  # left

            # FOOD LOCATION (4 signals: relative to head)
            1.0 if fy < hy else 0.0,   # food is up
            1.0 if fy > hy else 0.0,   # food is down
            1.0 if fx < hx else 0.0,   # food is left
            1.0 if fx > hx else 0.0,   # food is right
        ]

    def step(self):
        if not self.alive:
            return

        inputs = self.get_inputs()

        # Brain thinks: get activations for visualization + final decision
        self.last_acts = self.brain.forward_verbose(inputs)
        probs          = self.last_acts[-1]

        # Pick the action with highest probability
        action = int(np.argmax(probs))
        # 0 = turn left, 1 = go straight, 2 = turn right

        # Build thoughts for display
        action_names = ['TURN LEFT', 'GO STRAIGHT', 'TURN RIGHT']
        self.thoughts = []
        if inputs[0]: self.thoughts.append(('⚠ DANGER AHEAD',  DANGER_COL))
        if inputs[1]: self.thoughts.append(('⚠ DANGER LEFT',   DANGER_COL))
        if inputs[2]: self.thoughts.append(('⚠ DANGER RIGHT',  DANGER_COL))
        if inputs[7]: self.thoughts.append(('↑ FOOD UP',        FOOD_COL))
        if inputs[8]: self.thoughts.append(('↓ FOOD DOWN',      FOOD_COL))
        if inputs[9]: self.thoughts.append(('← FOOD LEFT',      FOOD_COL))
        if inputs[10]:self.thoughts.append(('→ FOOD RIGHT',     FOOD_COL))
        self.thoughts.append((
            f'→ {action_names[action]} ({probs[action]*100:.0f}%)',
            MYCEL
        ))

        # Apply action: update direction
        if action == 0:
            self.dir_idx = (self.dir_idx + 3) % 4   # turn left
        elif action == 2:
            self.dir_idx = (self.dir_idx + 1) % 4   # turn right
        # action == 1: keep direction

        dx, dy  = DIRS[self.dir_idx]
        hx, hy  = self.body[0]
        nx, ny  = hx + dx, hy + dy

        # Wall collision → die
        if nx < 0 or ny < 0 or nx >= GRID or ny >= GRID:
            self.alive = False
            self._calc_fitness()
            return

        # Self collision → die
        if (nx, ny) in set(self.body):
            self.alive = False
            self._calc_fitness()
            return

        # Move
        self.body.insert(0, (nx, ny))

        # Ate food?
        if (nx, ny) == self.food:
            self.score         += 1
            self.steps_to_food  = MAX_STEPS + self.score * 25
            self.food           = self._spawn_food()
        else:
            self.body.pop()       # remove tail
            self.steps_to_food   -= 1
            if self.steps_to_food <= 0:
                self.alive = False  # starved
                self._calc_fitness()
                return

        self.steps += 1

    def _calc_fitness(self):
        """
        FITNESS FUNCTION — what evolution maximizes.
        Food is rewarded exponentially (eating more is much better).
        Survival gives a small bonus (avoids suicidal snakes).
        The formula shapes the behavior that emerges.
        """
        self.fitness = (self.score ** 2) * 500 + self.steps * 0.5

    def run_to_death(self):
        """Play out a full game silently. Used during fast evolution."""
        while self.alive:
            self.step()


# =============================================================================
# EVOLUTION ENGINE
# =============================================================================

class Evolution:
    """Manages the population across generations."""

    def __init__(self):
        self.generation   = 0
        self.population   = [SnakeAgent() for _ in range(POP_SIZE)]
        self.best_ever    = None
        self.best_fitness = 0.0
        self.history      = []   # list of {gen, best, avg, score}

    def run_generation(self):
        """
        Run one full generation:
        1. All snakes play to completion
        2. Sort by fitness
        3. Breed next generation
        Returns stats for logging.
        """
        # Play all snakes to death
        for snake in self.population:
            snake.reset()
            snake.run_to_death()

        # Sort: best first
        self.population.sort(key=lambda s: s.fitness, reverse=True)

        best   = self.population[0]
        avg_f  = np.mean([s.fitness for s in self.population])

        # Track best ever
        if best.fitness > self.best_fitness:
            self.best_fitness = best.fitness
            self.best_ever    = SnakeAgent(best.brain.copy())

        self.history.append({
            'gen':   self.generation,
            'best':  best.fitness,
            'avg':   avg_f,
            'score': best.score
        })

        # ── BREED NEXT GENERATION ────────────────────────────────────────
        next_pop = []

        # Elitism: keep top ELITES unchanged
        for i in range(ELITES):
            next_pop.append(SnakeAgent(self.population[i].brain.copy()))

        # Fill rest: tournament selection + crossover + mutation
        while len(next_pop) < POP_SIZE:
            pa = tournament_select(self.population)
            pb = tournament_select(self.population)
            child_weights = mutate(crossover(
                pa.brain.to_flat(),
                pb.brain.to_flat()
            ))
            next_pop.append(SnakeAgent(MycelialBrain(child_weights)))

        self.population = next_pop
        self.generation += 1

        return best.fitness, avg_f, best.score


# =============================================================================
# RENDERER
# All drawing logic. Separated cleanly from game logic.
# =============================================================================

class Renderer:
    def __init__(self, screen, font_sm, font_md, font_lg):
        self.screen  = screen
        self.font_sm = font_sm
        self.font_md = font_md
        self.font_lg = font_lg

    # ── Game grid ──────────────────────────────────────────────────────────
    def draw_game(self, snake, generation, best_score):
        # Background
        pygame.draw.rect(self.screen, BG, (0, 0, GAME_W, GAME_H))

        # Grid lines
        for i in range(GRID + 1):
            c = (15, 25, 15)
            pygame.draw.line(self.screen, c, (i * CELL, 0), (i * CELL, GAME_H))
            pygame.draw.line(self.screen, c, (0, i * CELL), (GAME_W, i * CELL))

        if not snake:
            return

        # Food — glowing orange circle
        fx, fy = snake.food
        cx = fx * CELL + CELL // 2
        cy = fy * CELL + CELL // 2
        # Glow rings
        for r, alpha in [(CELL // 2, 30), (CELL // 3, 60)]:
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*FOOD_COL, alpha), (r, r), r)
            self.screen.blit(surf, (cx - r, cy - r))
        pygame.draw.circle(self.screen, FOOD_COL, (cx, cy), CELL // 4)

        # Snake body
        for i, (bx, by) in enumerate(snake.body):
            t       = 1.0 - i / max(len(snake.body), 1)
            rect    = pygame.Rect(bx * CELL + 1, by * CELL + 1, CELL - 2, CELL - 2)
            if i == 0:
                # Head: bright green with glow
                pygame.draw.rect(self.screen, MYCEL, rect, border_radius=3)
                # Eye dots
                ex = bx * CELL + CELL // 2
                ey = by * CELL + CELL // 2
                pygame.draw.circle(self.screen, BG, (ex - 3, ey - 2), 2)
                pygame.draw.circle(self.screen, BG, (ex + 3, ey - 2), 2)
            else:
                g = int(t * 160 + 30)
                pygame.draw.rect(self.screen, (0, g, 40), rect, border_radius=2)

        # HUD text
        gen_surf = self.font_sm.render(
            f'GEN {generation}   SCORE {snake.score}   BEST {best_score}',
            True, MYCEL_DIM)
        self.screen.blit(gen_surf, (6, 4))

    # ── Right panel ────────────────────────────────────────────────────────
    def draw_panel(self, snake, evo, thoughts):
        x0 = GAME_W
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, SIDE_W, WIN_H))
        pygame.draw.line(self.screen, BORDER, (x0, 0), (x0, WIN_H), 1)

        # Title
        t = self.font_md.render('MYCELIAL BRAIN', True, MYCEL)
        self.screen.blit(t, (x0 + SIDE_W // 2 - t.get_width() // 2, 8))

        # ── Network visualization ──────────────────────────────────────────
        self._draw_network(snake, x0, 36, SIDE_W, 240)

        # ── Weight heatmap ─────────────────────────────────────────────────
        self._draw_heatmap(evo, x0, 285, SIDE_W, 30)

        # ── Stats ──────────────────────────────────────────────────────────
        y = 325
        stats = [
            ('GENERATION',  str(evo.generation)),
            ('POPULATION',  str(POP_SIZE)),
            ('BEST FITNESS',f'{evo.best_fitness:.0f}'),
            ('WEIGHTS',     str(MycelialBrain().total_weights)),
        ]
        for label, val in stats:
            ls = self.font_sm.render(label, True, DIM_COL)
            vs = self.font_sm.render(val,   True, MYCEL)
            self.screen.blit(ls, (x0 + 10, y))
            self.screen.blit(vs, (x0 + SIDE_W - vs.get_width() - 10, y))
            y += 18

        # ── Thoughts ───────────────────────────────────────────────────────
        y += 8
        t = self.font_sm.render('SNAKE THOUGHTS', True, MYCEL_DIM)
        self.screen.blit(t, (x0 + 10, y))
        y += 16
        for text, color in (thoughts or []):
            ts = self.font_sm.render(text, True, color)
            self.screen.blit(ts, (x0 + 10, y))
            y += 15
            if y > WIN_H - BOTTOM_H - 20:
                break

        # ── Controls reminder ──────────────────────────────────────────────
        controls = ['SPACE=pause  F=fast  R=reset  +/-=speed  Q=quit']
        for i, c in enumerate(controls):
            cs = self.font_sm.render(c, True, (40, 60, 40))
            self.screen.blit(cs, (x0 + 5, WIN_H - BOTTOM_H - 16))

    def _draw_network(self, snake, x0, y0, W, H):
        """
        Draw the live neural network with glowing activations.
        Each layer is a column of circles. Edges show weights.
        Brightness = activation strength.
        """
        if not snake or not snake.last_acts:
            return

        acts        = snake.last_acts
        layer_count = len(LAYER_SIZES)
        margin      = 14

        # Compute node positions
        positions = []
        for l, n in enumerate(LAYER_SIZES):
            lx = x0 + margin + (l / (layer_count - 1)) * (W - 2 * margin)
            col = []
            for i in range(n):
                ly = y0 + margin + ((i + 1) / (n + 1)) * (H - 2 * margin)
                col.append((int(lx), int(ly)))
            positions.append(col)

        # Draw edges (only significant weights to avoid clutter)
        for l in range(layer_count - 1):
            W_mat = snake.brain.W[l]
            for j in range(LAYER_SIZES[l + 1]):
                for i in range(LAYER_SIZES[l]):
                    w = W_mat[j, i]
                    if abs(w) < 0.4:
                        continue
                    strength = min(abs(w) / 2.5, 1.0)
                    alpha    = int(strength * 120)
                    color    = (*MYCEL, alpha) if w > 0 else (*DANGER_COL, alpha)
                    lw       = max(1, int(strength * 2))
                    # Use aaline for smooth edges
                    c_solid = MYCEL if w > 0 else DANGER_COL
                    c_dim   = tuple(int(x * strength * 0.5) for x in c_solid)
                    pygame.draw.line(
                        self.screen, c_dim,
                        positions[l][i], positions[l + 1][j], lw
                    )

        # Draw nodes
        for l, layer_pos in enumerate(positions):
            layer_acts = acts[l] if l < len(acts) else []
            for i, (nx, ny) in enumerate(layer_pos):
                a     = float(np.clip(layer_acts[i] if i < len(layer_acts) else 0, 0, 1))
                r     = 3 + int(a * 4)
                # Glow effect
                if a > 0.3:
                    for gr, ga in [(r + 4, 30), (r + 2, 60)]:
                        gs = pygame.Surface((gr * 2, gr * 2), pygame.SRCALPHA)
                        pygame.draw.circle(gs, (*MYCEL, ga), (gr, gr), gr)
                        self.screen.blit(gs, (nx - gr, ny - gr))
                g_val = int(a * 200 + 55)
                col   = (0, g_val, int(a * 80 + 20))
                pygame.draw.circle(self.screen, col, (nx, ny), r)

    def _draw_heatmap(self, evo, x0, y0, W, H):
        """
        Visualize all weights of the best brain as a color strip.
        Green = positive weight (excitatory connection).
        Red   = negative weight (inhibitory connection).
        """
        if not evo.best_ever:
            return
        label = self.font_sm.render('WEIGHT MAP (best brain)', True, DIM_COL)
        self.screen.blit(label, (x0 + 10, y0))
        y0 += 14

        flat  = evo.best_ever.brain.to_flat()
        total = len(flat)
        if total == 0:
            return

        bar_w = W - 20
        cell_w = bar_w / total

        for i, w in enumerate(flat):
            norm = math.tanh(w)   # squish to -1..1
            px   = int(x0 + 10 + i * cell_w)
            pw   = max(1, int(cell_w) + 1)
            if norm > 0:
                c = (0, int(norm * 200 + 55), 0)
            else:
                c = (int(-norm * 200 + 55), 0, 0)
            pygame.draw.rect(self.screen, c, (px, y0, pw, H))

        # Legend
        l1 = self.font_sm.render('inhibit', True, DANGER_COL)
        l2 = self.font_sm.render('excite',  True, MYCEL)
        self.screen.blit(l1, (x0 + 10,          y0 + H + 2))
        self.screen.blit(l2, (x0 + W - l2.get_width() - 10, y0 + H + 2))

    # ── Bottom fitness chart ───────────────────────────────────────────────
    def draw_chart(self, evo):
        """
        Live fitness chart. Green = best per generation. Dim = average.
        Watch both lines rise as evolution improves the population.
        """
        y0 = GAME_H
        pygame.draw.rect(self.screen, (8, 12, 8), (0, y0, WIN_W, BOTTOM_H))
        pygame.draw.line(self.screen, BORDER, (0, y0), (WIN_W, y0), 1)

        title = self.font_sm.render(
            f'FITNESS OVER GENERATIONS — BEST: {evo.best_fitness:.0f}',
            True, DIM_COL)
        self.screen.blit(title, (6, y0 + 4))

        h = evo.history
        if len(h) < 2:
            return

        max_f = max(max(e['best'] for e in h), 1)
        chart_h = BOTTOM_H - 24
        chart_w = WIN_W - 12

        def to_screen(idx, val):
            px = int(6 + (idx / (len(h) - 1)) * chart_w)
            py = int(y0 + 18 + chart_h - (val / max_f) * chart_h)
            return (px, py)

        # Average line (dim)
        avg_pts = [to_screen(i, e['avg'])  for i, e in enumerate(h)]
        if len(avg_pts) > 1:
            pygame.draw.lines(self.screen, MYCEL_DIM, False, avg_pts, 1)

        # Best line (bright)
        best_pts = [to_screen(i, e['best']) for i, e in enumerate(h)]
        if len(best_pts) > 1:
            pygame.draw.lines(self.screen, MYCEL, False, best_pts, 2)

        # Gen counter
        gc = self.font_sm.render(f'GEN {evo.generation}', True, MYCEL_DIM)
        self.screen.blit(gc, (WIN_W - gc.get_width() - 8, y0 + 4))


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    pygame.init()
    pygame.display.set_caption('🍄 Mycelial Snake — Neuroevolution')

    screen   = pygame.display.set_mode((WIN_W, WIN_H))
    clock    = pygame.time.Clock()

    # Fonts
    try:
        font_sm = pygame.font.SysFont('couriernew', 11)
        font_md = pygame.font.SysFont('couriernew', 13, bold=True)
        font_lg = pygame.font.SysFont('couriernew', 16, bold=True)
    except:
        font_sm = pygame.font.SysFont('monospace', 11)
        font_md = pygame.font.SysFont('monospace', 13)
        font_lg = pygame.font.SysFont('monospace', 16)

    renderer = Renderer(screen, font_sm, font_md, font_lg)

    # ── State ────────────────────────────────────────────────────────────────
    evo       = Evolution()
    demo      = None       # the snake we're watching play live
    running   = False      # has START been pressed?
    paused    = False
    fast_mode = True       # evolve many gens before showing demo
    demo_fps  = 12         # how fast the demo snake moves
    best_score_seen = 0

    # ── Print startup info ───────────────────────────────────────────────────
    b = MycelialBrain()
    print(f"""
╔══════════════════════════════════════════════════╗
║       🍄  MYCELIAL SNAKE — NEUROEVOLUTION        ║
╠══════════════════════════════════════════════════╣
║  Population:    {POP_SIZE} snakes per generation          ║
║  Architecture:  {' → '.join(str(x) for x in LAYER_SIZES)}                  ║
║  Total weights: {b.total_weights}                             ║
║  Mutation rate: {MUTATION_RATE}                            ║
╠══════════════════════════════════════════════════╣
║  SPACE = start / pause    F = fast mode          ║
║  R     = reset            +/- = demo speed       ║
║  Q/ESC = quit                                    ║
╚══════════════════════════════════════════════════╝
    """)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def do_evolution_step():
        """Run one generation of evolution."""
        best_f, avg_f, score = evo.run_generation()
        nonlocal best_score_seen
        if score > best_score_seen:
            best_score_seen = score
        print(f'  Gen {evo.generation:>4} | '
              f'best fit: {best_f:>8.0f} | '
              f'avg: {avg_f:>7.0f} | '
              f'score: {score}')
        return score

    def refresh_demo():
        """Start watching the best snake play."""
        nonlocal demo
        if evo.best_ever:
            demo = SnakeAgent(evo.best_ever.brain.copy())
            demo.reset()

    # ── Main loop ────────────────────────────────────────────────────────────
    gens_this_session = 0
    last_demo_step    = time.time()
    last_evo_step     = time.time()

    while True:
        dt = clock.tick(60) / 1000.0

        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()

                elif event.key == pygame.K_SPACE:
                    if not running:
                        running = True
                        print('\n🌱 Evolution started...\n')
                    else:
                        paused = not paused
                        print('⏸ Paused' if paused else '▶ Resumed')

                elif event.key == pygame.K_f:
                    fast_mode = not fast_mode
                    print(f'⚡ Fast mode: {"ON" if fast_mode else "OFF"}')

                elif event.key == pygame.K_r:
                    evo              = Evolution()
                    demo             = None
                    running          = False
                    paused           = False
                    gens_this_session = 0
                    best_score_seen  = 0
                    print('↺ Reset.')

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    demo_fps = min(demo_fps + 2, 60)
                    print(f'Speed: {demo_fps} fps')

                elif event.key == pygame.K_MINUS:
                    demo_fps = max(demo_fps - 2, 1)
                    print(f'Speed: {demo_fps} fps')

        # ── Evolution tick ───────────────────────────────────────────────────
        if running and not paused:
            now = time.time()

            if fast_mode:
                # In fast mode: run multiple generations per frame
                for _ in range(3):
                    do_evolution_step()
                    gens_this_session += 1
                    if gens_this_session % 10 == 0:
                        refresh_demo()
            else:
                # Slow mode: one generation per second-ish
                if now - last_evo_step > 0.5:
                    do_evolution_step()
                    gens_this_session += 1
                    refresh_demo()
                    last_evo_step = now

            # Always have a demo snake running
            if demo is None and evo.best_ever:
                refresh_demo()

        # ── Demo snake step ──────────────────────────────────────────────────
        if demo and running and not paused:
            now = time.time()
            if now - last_demo_step >= 1.0 / demo_fps:
                demo.step()
                last_demo_step = now
                if not demo.alive:
                    # Respawn demo with best brain
                    refresh_demo()
                    # Occasionally upgrade demo to latest best
                    if evo.best_ever and random.random() < 0.3:
                        demo = SnakeAgent(evo.best_ever.brain.copy())
                        demo.reset()

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(BG)

        renderer.draw_game(demo, evo.generation, best_score_seen)
        renderer.draw_panel(demo, evo, demo.thoughts if demo else [])
        renderer.draw_chart(evo)

        # Startup message if not running
        if not running:
            lines = [
                '🍄 MYCELIAL SNAKE',
                '',
                'Press SPACE to start evolution',
                '',
                f'Population: {POP_SIZE} snakes',
                f'Network:    {" → ".join(str(x) for x in LAYER_SIZES)}',
                f'Weights:    {b.total_weights}',
                '',
                'F = fast mode (evolve faster)',
                '+/- = demo speed',
            ]
            overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            overlay.fill((6, 9, 16, 200))
            screen.blit(overlay, (0, 0))
            y = GAME_H // 2 - len(lines) * 14
            for line in lines:
                color = MYCEL if line.startswith('🍄') or line.startswith('Press') else DIM_COL
                s = font_md.render(line, True, color)
                screen.blit(s, (GAME_W // 2 - s.get_width() // 2, y))
                y += 22

        # Fast mode indicator
        if fast_mode and running:
            fm = font_sm.render('⚡ FAST EVOLVING...', True, GOLD)
            screen.blit(fm, (6, GAME_H - 20))

        pygame.display.flip()

if __name__ == '__main__':
    main()
