from __init__ import *
from sudoku import SudokuPuzzle

class BacktrackingSudoku(SudokuPuzzle):
	
	def __init__(self, puzzle, size):
		super().__init__(puzzle, size)
		self.numberoftries = 0

	def solve(self):
		empty_cell = self.find_empty_cell()
		if not empty_cell:
			return True

		row, col = empty_cell
		for value in range(1, self.size + 1):
			if self.is_valid(row, col, value):
				self.puzzle[row][col] = value
				self.numberoftries += 1
				if self.solve():
					return self.puzzle
				self.puzzle[row][col] = 0
		return None

	def find_empty_cell(self):
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row][col] == 0:
					return row, col
		return None




class BacktrackingSudokuOptimized(SudokuPuzzle):
	"""  Minimum Remaining Values (MRV), Degree Heuristic, and Forward Checking optimizations """
	def __init__(self, puzzle, size):
		super().__init__(puzzle, size)
		self.numberoftries = 0
		self.remaining_values = self.init_remaining_values()

	def init_remaining_values(self):
		self.remaining_values = [[set(range(1, self.size + 1)) for _ in range(self.size)] for _ in range(self.size)]
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row][col] != 0:
					self.update_remaining_values(row, col, self.puzzle[row][col])
		return self.remaining_values

	def update_remaining_values(self, row, col, value):
		for i in range(self.size):
			if value in self.remaining_values[row][i]:
				self.remaining_values[row][i].remove(value)
			if value in self.remaining_values[i][col]:
				self.remaining_values[i][col].remove(value)

		box_row, box_col = row // self.subgrid_size, col // self.subgrid_size
		for i in range(self.subgrid_size):
			for j in range(self.subgrid_size):
				temp_row = box_row * self.subgrid_size + i
				temp_col = box_col * self.subgrid_size + j
				if value in self.remaining_values[temp_row][temp_col]:
					self.remaining_values[temp_row][temp_col].remove(value)

	def solve(self):
		empty_cell = self.find_empty_cell()
		if not empty_cell:
			return self.puzzle

		row, col = empty_cell
		for value in list(self.remaining_values[row][col]):  # Create a copy of the set using list()
			if self.is_valid(row, col, value):
				self.puzzle[row][col] = value
				self.numberoftries += 1
				self.update_remaining_values(row, col, value)

				if self.solve():
					return self.puzzle

				self.puzzle[row][col] = 0
				self.undo_update_remaining_values(row, col, value)

		return None

	def select_unassigned_cell(self):
		min_remaining_values = self.size + 1
		min_degree = -1
		best_cell = None

		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row][col] == 0:
					remaining_value_count = len(self.remaining_values[row][col])
					degree = sum([1 for i in range(self.size) if self.puzzle[row][i] == 0 or self.puzzle[i][col] == 0])

					if remaining_value_count < min_remaining_values or (
							remaining_value_count == min_remaining_values and degree > min_degree):
						min_remaining_values = remaining_value_count
						min_degree = degree
						best_cell = (row, col)

		return best_cell
	
	def find_empty_cell(self):
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row][col] == 0:
					return row, col
		return None

	def undo_update_remaining_values(self, row, col, value):
		self.puzzle[row][col] = 0
		self.remaining_values[row][col].add(value)

		for i in range(self.size):
			if value not in self.remaining_values[row][i]:
				self.remaining_values[row][i].add(value)
			if value not in self.remaining_values[i][col]:
				self.remaining_values[i][col].add(value)

		row_start, col_start = row - row % self.subgrid_size, col - col % self.subgrid_size

		for i in range(row_start, row_start + self.subgrid_size):
			for j in range(col_start, col_start + self.subgrid_size):
				if value not in self.remaining_values[i][j]:
					self.remaining_values[i][j].add(value)


class MinigridBacktrackingSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size):
		super().__init__(puzzle, size)
		self.numberoftries = 0

	def solve(self):
		empty_cells = self.find_empty_cells()
		if not empty_cells:
			return True

		minigrids = self.get_minigrids()
		minigrid_values = self.get_minigrid_values(minigrids)
		sorted_cells = self.sort_empty_cells(empty_cells, minigrid_values)

		return self.backtrack(sorted_cells, 0)

	def backtrack(self, sorted_cells, index):
		if index == len(sorted_cells):
			return True

		row, col = sorted_cells[index]
		valid_numbers = self.get_valid_numbers(row, col)
		for num in valid_numbers:
			self.puzzle[row][col] = num
			self.numberoftries += 1
			if self.backtrack(sorted_cells, index + 1):
				return self.puzzle
			self.puzzle[row][col] = 0

		return None

	def find_empty_cells(self):
		return [(row, col) for row in range(self.size) for col in range(self.size) if self.puzzle[row][col] == 0]

	def get_minigrids(self):
		return [[self.puzzle[row + i][col + j] for i in range(self.subgrid_size) for j in range(self.subgrid_size)] for row in range(0, self.size, self.subgrid_size) for col in range(0, self.size, self.subgrid_size)]

	def get_minigrid_values(self, minigrids):
		return [[num for num in minigrid if num != 0] for minigrid in minigrids]

	def sort_empty_cells(self, empty_cells, minigrid_values):
		return sorted(empty_cells, key=lambda cell: len(set(range(1, self.size + 1)) - set(self.get_used_numbers(cell[0], cell[1], minigrid_values))))

	def get_valid_numbers(self, row, col):
		return list(set(range(1, self.size + 1)) - set(self.get_used_numbers(row, col)))

	def get_used_numbers(self, row, col, minigrid_values=None):
		used_numbers = set(self.puzzle[row]) | set(self.puzzle[i][col] for i in range(self.size))
		if minigrid_values:
			used_numbers |= set(minigrid_values[self.get_minigrid_index(row, col)])
		return used_numbers

	def get_minigrid_index(self, row, col):
		return (row // self.subgrid_size) * self.subgrid_size + col // self.subgrid_size
