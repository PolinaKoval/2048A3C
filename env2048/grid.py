import random
import numpy as np

class Grid(object):
    def __init__(self, size):
        self.size = size
        self.cells = [[Tile(x, y, 0) for x in range(size)] for y in range(size)]

    def __getitem__(self, item):
        return self.cells[item]

    def available_cells(self):
        cells = []
        for row in self.cells:
            for tile in row:
                if tile.value == 0:
                    cells.append(tile)
        return cells

    def insert_value_in_random_available_cell(self, value):
        cells = self.available_cells()
        if len(cells) != 0:
            cells[int(random.uniform(0, 1) * len(cells))].value = value

    def get_values(self):
        cells = []
        for row in self.cells:
            for x in row:
                cells.append(x.value)
        return cells

    def set_values(self, cells):
        i = 0
        for row in self.cells:
            for x in row:
                x.value = cells[i]
                i += 1

class Tile(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value
        self.merged = False
