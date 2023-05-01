from __init__ import *
from sudoku import SudokuPuzzle

class SimulatedAnnealingSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size, temperature=100, cooling_rate=0.99, steps=10000):
		super().__init__(puzzle, size)
		self.temperature = temperature
		self.cooling_rate = cooling_rate
		self.steps = steps
		self.numberoftries = 0

	def solve(self):
		current_solution = self.initialize_solution()
		best_solution = current_solution[:]
		best_fitness = self.fitness(best_solution)

		temp = self.temperature

		i = 0
		while best_fitness > 0 and temp > 0:
			if i % 100000 == 0:
				logging.info(f"iteration: {i}")
				logging.info(f"temperature: {temp}")
			candidate_solution = self.random_neighbor(current_solution)
			candidate_fitness = self.fitness(candidate_solution)
			self.numberoftries += 1

			if candidate_fitness < best_fitness:
				best_solution = candidate_solution[:]
				best_fitness = candidate_fitness
				logging.info(f" iteration: {i} Current Best Fitness {best_fitness}")

			if best_fitness == 0:
				logging.info(f"Final: Best Fitness: {best_fitness}; iterations: {i}; temp: {temp}; number of tries: {self.numberoftries}")
				return best_solution

			if self.acceptance_probability(self.fitness(current_solution), candidate_fitness, temp) > random.random():
				current_solution = candidate_solution[:]

			temp *= self.cooling_rate
			i += 1

		logging.info("Could not find the best solved sudoku")
		return None


	def initialize_solution(self):
		solution = [row[:] for row in self.puzzle]
		for row in range(self.size):
			for col in range(self.size):
				if solution[row][col] == 0:
					valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, solution)]
					if valid_numbers:
						solution[row][col] = random.choice(valid_numbers)
		return solution


	def random_neighbor(self, solution):
		new_solution = [row[:] for row in solution]

		# Get a random row and column index
		row, col = random.randint(0, self.size - 1), random.randint(0, self.size - 1)

		# Keep selecting new row and column indices until an empty cell is found
		while self.puzzle[row][col] != 0:
			row, col = random.randint(0, self.size - 1), random.randint(0, self.size - 1)

		# Get a list of valid values that can be assigned to the empty cell
		valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, new_solution)]

		# If there are no valid values, return the original solution
		if not valid_numbers:
			return new_solution

		# Assign a random valid value to the empty cell
		new_solution[row][col] = random.choice(valid_numbers)

		return new_solution


	def fitness(self, individual):
		total_errors = 0
		for row in range(self.size):
			for col in range(self.size):
				if not self.is_valid(row, col, individual[row][col], individual):
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

	def acceptance_probability(self, current_fitness, candidate_fitness, temperature):
		if candidate_fitness < current_fitness:
			return 1.0
		return math.exp((current_fitness - candidate_fitness) / temperature)

