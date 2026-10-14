from collections import deque
from time import sleep
import numpy as np

from .snake import Snake, OPPOSITE
from .fruits import Fruits


DIRS = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}

class Game:
    def __init__(self, model, size: int = 15, log=True, delay=False):
        self.model = model

        self.size = size
        self.grid = np.zeros((size, size))
        self.snake = Snake((size//2, size//2))
        self.fruits = Fruits(size)
        self.fruits.init_game(3, self.snake.positions)

        self.score = 0
        self.stop = False
        self.steps_since_food = 0
        self.max_steps_without_food = size * size

        self.death_log = {"starving": 0, "walls": 0, "self": 0}
        self.last_death = None # details of the last death (see step), None while the game is running

        self.log = log
        self.delay = delay

    def draw(self):
        if not self.log:
            return

        print("\033[H\033[J", end="")
        print(f"SCORE: {self.score}")
        for y,row in enumerate(self.grid):
            for x,col in enumerate(row):
                if (x,y) in self.snake.positions:
                    if (x,y) == self.snake.head.pos():
                        print("O", end="")
                    else:
                        print("X", end="")
                elif (x,y) in self.fruits.positions:
                    print("F", end="")
                else:
                    print(".", end="")
            print("")

        if self.delay:
            sleep(0.05)

    def is_valide_state(self):
        head = self.snake.head
        if head.x < 0 or head.x >= self.size or head.y < 0 or head.y >= self.size:
            self.death_log["walls"] += 1
            return False
        if len(self.snake.positions) != self.snake.size:
            self.death_log["self"] += 1
            return False
        if self.steps_since_food >= self.max_steps_without_food:
            self.death_log["starving"] += 1
            return False
        return True

    def get_state(self):
        head = self.snake.head

        def is_danger(x, y):
            return x < 0 or x >= self.size or y < 0 or y >= self.size or (x, y) in self.snake.positions

        fruit = min(self.fruits.positions, key=lambda p: abs(p[0] - head.x) + abs(p[1] - head.y))

        return np.array([
            is_danger(head.x, head.y - 1),  # danger up
            is_danger(head.x, head.y + 1),  # danger down
            is_danger(head.x - 1, head.y),  # danger left
            is_danger(head.x + 1, head.y),  # danger right
            fruit[1] < head.y,  # food up
            fruit[1] > head.y,  # food down
            fruit[0] < head.x,  # food left
            fruit[0] > head.x,  # food right
        ], dtype=np.float32)

    def action_scalars(self):
        """The 17 engineered scalars shared by get_state_grid and get_state_scalars:
        direction (4), area after each action / size^2 (4), space for the body (4), tail reachable (4),
        hunger = steps since food / max (1), actions in "udlr" order."""
        snake = self.snake
        head = snake.head
        areas, tails = self.action_areas(snake.positions, snake.tail.pos(), head.pos(), snake.direction)
        scalars = np.zeros(17, dtype=np.float32)
        scalars[16] = self.steps_since_food / self.max_steps_without_food
        scalars["udlr".index(snake.direction)] = 1
        for i, d in enumerate("udlr"):
            # AREA IF ACTION
            scalars[4 + i] = areas[d] / (self.size * self.size)
            # SPACE FOR BODY
            scalars[8 + i] = min(areas[d] / snake.size, 2) / 2 # is the area big enough for the body?
            # TAIL DIRECTION
            scalars[12 + i] = tails[d]
        return scalars

    def get_state_grid(self):
        """Flat float32 state: 4 spatial planes (head, body, fruits, walls) of (size+2)^2, then the 17 scalars
        from action_scalars()."""
        n = self.size+2
        planes = np.zeros((4, n, n), dtype=np.float32)

        # WALLS
        planes[3, [0, -1], :] = 1
        planes[3, :, [0, -1]] = 1

        # HEAD
        snake = self.snake
        head = snake.head
        planes[0, head.y+1, head.x+1] = 1

        # BODY
        comp, i = snake.tail, 0
        while comp is not head:
            planes[1, comp.y + 1, comp.x + 1] = (i + 1) / snake.size
            comp, i = comp.next, i + 1

        # FRUITS
        for x,y in self.fruits.positions:
            planes[2, y+1, x+1] = 1

        return np.concatenate([planes.ravel(), self.action_scalars()])

    def get_state_scalars(self):
        """No spatial grid at all: the original 8 features (immediate danger + food direction) plus the
        17 engineered scalars also fed to the CNN (direction, area, space for the body, tail reachable, hunger).
        Used to test how much of the CNN's gain actually comes from the raster map vs. these scalars."""
        return np.concatenate([self.get_state(), self.action_scalars()])

    def _flood(self, start, blocked):
        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRS.values():
                nxt = (x + dx, y + dy)
                if 0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size and nxt not in seen and nxt not in blocked:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def action_areas(self, positions, tail, head, direction):
        """Number of cells reachable after each action (0 if it kills right away).
        The tail is treated as free (it moves away) and a U-turn counts as going straight, like Snake.move.
        Also returns, per action, whether the tail is reachable afterwards (1.0 / 0.0): then the snake always has a way out.
        Neighbors in the same region share a single flood fill."""
        blocked = positions - {tail}
        areas, tails, regions = {}, {}, []
        for d, (dx, dy) in DIRS.items():
            if d == OPPOSITE[direction]:
                continue
            nxt = (head[0] + dx, head[1] + dy)
            if not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size) or nxt in blocked:
                areas[d], tails[d] = 0, 0.0
                continue
            seen = next((r for r in regions if nxt in r), None)
            if seen is None:
                seen = self._flood(nxt, blocked)
                regions.append(seen)
            areas[d], tails[d] = len(seen), float(tail in seen)
        areas[OPPOSITE[direction]] = areas[direction]
        tails[OPPOSITE[direction]] = tails[direction]
        return areas, tails

    def observe(self):
        state_type = getattr(self.model, "state_type", "vector")
        if state_type == "grid":
            return self.get_state_grid()
        if state_type == "scalars":
            return self.get_state_scalars()
        return self.get_state()

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.snake.reset()
        self.fruits.reset()
        self.fruits.init_game(3, self.snake.positions)
        self.score = 0
        self.stop = False
        self.steps_since_food = 0
        self.last_death = None

    def nearest_fruit_dist(self):
        head = self.snake.head
        return min(abs(x - head.x) + abs(y - head.y) for x, y in self.fruits.positions)

    def step(self, action):
        self.draw()

        dist_before = self.nearest_fruit_dist()

        before = (self.snake.positions.copy(), self.snake.tail.pos(), self.snake.head.pos(), self.snake.direction)
        self.snake.move(action)
        if not self.is_valide_state():
            self.snake.alive = False
            self.stop = True
            areas, _ = self.action_areas(*before)
            real = [areas[d] for d in "udlr" if d != OPPOSITE[before[3]]]
            self.last_death = {
                "length": self.snake.size,
                "safe_moves": sum(a > 0 for a in real), # of the 3 possible moves, how many did not kill
                "best_area": max(real),
            }
            return self.observe(), -10, True

        # reward = -0.01 if self.score < 40 else 0.0
        # reward = 0.0
        reward = -0.01
        if self.snake.head.pos() in self.fruits.positions:
            # grow first: the tail cell comes back, so the new fruit must not spawn on it
            self.snake.grow()
            self.fruits.eat(self.snake.head.pos(), self.snake.positions)
            self.score += 1
            self.steps_since_food = 0
            reward += 10
        else:
            self.steps_since_food += 1

        reward += 0.1 * (dist_before - self.nearest_fruit_dist())

        return self.observe(), reward, self.stop

    def play(self):
        state = self.observe()
        while not self.stop:
            action = self.model.action(state)
            state, reward, done = self.step(action)