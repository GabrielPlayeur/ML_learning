
from random import randint

class Fruits:
    def __init__(self, size_map: int):
        self.size_map = size_map
        self.positions = set()

    def add(self, snake_positions: set[tuple[int,int]], max_try: int = 100):
        pos = None
        i = 0
        while (not pos or pos in snake_positions or pos in self.positions) and i < max_try:
            i += 1
            pos = (randint(0, self.size_map-1), randint(0, self.size_map-1))
        if pos == None:
            raise ValueError("NEED ADD DETERMINISTE POS FRUTIS")
        self.positions.add(pos)

    def init_game(self, fruits_num: int, snake_positions: set[tuple[int,int]]):
        for _ in range(fruits_num):
            self.add(snake_positions)

    def eat(self, pos: tuple[int, int], snake_positions: set[tuple[int,int]]):
        self.positions.remove(pos)
        self.add(snake_positions)

    def reset(self):
        self.positions.clear()