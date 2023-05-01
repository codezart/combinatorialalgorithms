from __init__ import *
from sudoku import SudokuPuzzle

class BatAlgorithmSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size, num_bats=50, generations=1000, alpha=0.9, gamma=0.9):
		super().__init__(puzzle, size)
		self.num_bats = num_bats
		self.generations = generations
		self.alpha = alpha
		self.gamma = gamma

	def solve(self):
		bats = self.initialize_bats()
		best_bat = min(bats, key=self.fitness)

		for _ in range(self.generations):
			for bat in bats:
				new_bat = self.move_bat(bat)
				if self.fitness(new_bat) < self.fitness(bat):
					bat = new_bat

				if random.random() < self.gamma:
					bat = self.local_search(bat, best_bat)

				if self.fitness(bat) < self.fitness(best_bat):
					best_bat = bat

			if self.fitness(best_bat) == 0 and self.check_sudoku():
				return best_bat

		return None

	def initialize_bats(self):
		bats = []
		for _ in range(self.num_bats):
			bat = [row[:] for row in self.puzzle]
			for row in range(self.size):
				for col in range(self.size):
					if bat[row][col] == 0:
						valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, bat)]
						if valid_numbers:
							bat[row][col] = random.choice(valid_numbers)
			bats.append(bat)
		return bats
	
	def move_bat(self, bat):
		new_bat = [row[:] for row in bat]
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row][col] == 0 and random.random() < self.alpha:
					valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, self.puzzle)]
					new_bat[row][col] = random.choice(valid_numbers)
		return new_bat

	def local_search(self, bat, best_bat):
		new_bat = [row[:] for row in bat]
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row][col] == 0:
					valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, self.puzzle)]
					best_number = min(valid_numbers, key=lambda x: abs(x - best_bat[row][col]))
					new_bat[row][col] = best_number
		return new_bat

	def fitness(self, bat):
		total_errors = 0
		for row in range(self.size):
			for col in range(self.size):
				if not self.is_valid(row, col, bat[row][col], bat):
					total_errors += 1
		return total_errors

	def is_valid(self, row, col, value, solution):
		self.numof_is_valid_checks += 1
		if value < 1 or value > self.size:
			return False

		for i in range(self.size):
			if (solution[row][i] == value and i != col) or (solution[i][col] == value and i != row):
				return False

		subgrid_row, subgrid_col = row // self.subgrid_size, col // self.subgrid_size
		for i in range(self.subgrid_size):
			for j in range(self.subgrid_size):
				temp_row = subgrid_row * self.subgrid_size + i
				temp_col = subgrid_col * self.subgrid_size + j
				if solution[temp_row][temp_col] == value and temp_row != row and temp_col != col:
					return False
		return True
