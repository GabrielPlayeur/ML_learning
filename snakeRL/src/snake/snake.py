from __future__ import annotations

class SnakeComponent:
    def __init__(self, pos: tuple[int, int], prev: SnakeComponent, next: SnakeComponent):
        self.x = pos[0]
        self.y = pos[1]
        self.pos = lambda: (self.x, self.y)
        self.prev = prev
        self.next = next

OPPOSITE = {"u": "d", "d": "u", "l": "r", "r": "l"}

class Snake:
    def __init__(self, pos: tuple[int, int], size: int = 3):
        self._initial_size = size
        self._initial_pos = pos
        self.size = size
        self.positions = set()
        self.alive = True
        self.direction = "r"
        self.tail: SnakeComponent = None

        self.build()
        self._prev_tail_pos = self.tail.pos()

    def build(self):
        self._unlink()
        self.alive = True
        self.direction = "r"
        self.positions.clear()
        tail_pos = (self._initial_pos[0]-2, self._initial_pos[1])
        self.tail = SnakeComponent(tail_pos, None, None)
        self.positions.add(tail_pos)
        prev_comp = self.tail
        for _ in range(self.size-1):
            pos = (prev_comp.x+1, prev_comp.y)
            self.positions.add(pos)
            comp = SnakeComponent(pos, prev_comp, None)
            prev_comp.next = comp
            prev_comp = comp
        self.head = comp

    def _unlink(self):
        comp = self.tail
        while comp is not None:
            nxt = comp.next
            comp.prev = None
            comp.next = None
            comp = nxt

    def reset(self):
        self.size = self._initial_size
        self.build()
        self._prev_tail_pos = self.tail.pos()

    def move(self, dir: str):
        # a U-turn would run the head into the neck: keep going straight instead
        if dir == OPPOSITE[self.direction]:
            dir = self.direction
        self.direction = dir
        # save if the snake eat a fruit
        self._prev_tail_pos = self.tail.pos()
        # make the tail component to be the head to make impression of movement 
        prev_head = self.head
        self.head = self.tail
        self.tail = self.tail.next
        self.tail.prev = None
        self.head.next = None
        self.head.prev = prev_head
        prev_head.next = self.head
        # move the new head
        self.positions.remove(self.head.pos())
        self.head.x, self.head.y = prev_head.x, prev_head.y
        if dir == "u":
            self.head.y -= 1
        elif dir == "d":
            self.head.y += 1
        elif dir == "l":
            self.head.x -= 1
        elif dir == "r":
            self.head.x += 1
        self.positions.add(self.head.pos())

    def grow(self):
        comp = SnakeComponent(self._prev_tail_pos, next=self.tail, prev=None)
        self.tail.prev = comp
        self.tail = comp
        self.positions.add(self._prev_tail_pos)
        self.size += 1