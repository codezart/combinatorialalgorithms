from __init__ import *
from sudoku import SudokuPuzzle

class AntColonyOptimizationSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size, num_ants=100, generations=500, alpha=1, beta=1, evaporation_rate=0.1):
		super().__init__(puzzle, size)
		self.num_ants = num_ants
		self.generations = generations
		self.alpha = alpha
		self.beta = beta
		self.evaporation_rate = evaporation_rate
		self.numberoftries = 0

	def solve(self):
		pheromone_matrix = self.initialize_pheromone_matrix()

		while True:
			ant_solutions = [self.construct_solution(pheromone_matrix) for _ in range(self.num_ants)]
			pheromone_matrix = self.update_pheromone_matrix(pheromone_matrix, ant_solutions)

			best_solution = min(ant_solutions, key=self.fitness)
			if self.fitness(best_solution) == 0:
				return best_solution

		# return None

	def initialize_pheromone_matrix(self):
		return np.ones((self.size, self.size, self.size)) / self.size

	def construct_solution(self, pheromone_matrix):
		solution = [copy.deepcopy(row)for row in self.puzzle]
		for row in range(self.size):
			for col in range(self.size):
				if solution[row][col] == 0:
					self.numberoftries += 1
					valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, solution)]
					probabilities = self.calculate_probabilities(row, col, pheromone_matrix, valid_numbers)
					
					if not probabilities:
						continue

					solution[row][col] = self.pick_number_by_probability(probabilities, valid_numbers)
		return solution

	def calculate_probabilities(self, row, col, pheromone_matrix, valid_numbers):
		if len(valid_numbers) == 0:
			return []

		pheromones = [pheromone_matrix[row, col, num - 1] for num in valid_numbers]
		heuristic_values = [1 / len(valid_numbers)] * len(valid_numbers)
		combined_values = [p ** self.alpha * h ** self.beta for p, h in zip(pheromones, heuristic_values)]
		total = sum(combined_values)
		return [value / total for value in combined_values]

	def pick_number_by_probability(self, probabilities, valid_numbers):
		return random.choices(valid_numbers, weights=probabilities, k=1)[0]

	def update_pheromone_matrix(self, pheromone_matrix, ant_solutions):
		pheromone_matrix = (1 - self.evaporation_rate) * pheromone_matrix
		for solution in ant_solutions:
			fitness_value = self.fitness(solution)
			if fitness_value == 0:
				delta_pheromone = 1
			else:
				delta_pheromone = 1 / fitness_value
			for row in range(self.size):
				for col in range(self.size):
					if self.puzzle[row][col] == 0:
						num = solution[row][col] - 1
						pheromone_matrix[row, col, num] += delta_pheromone
		return pheromone_matrix

	def fitness(self, solution):
		total_errors = 0
		for row in range(self.size):
			for col in range(self.size):
				if not self.is_valid(row, col, solution[row][col], solution):
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


class ACOAlgorithmSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size, num_ants=100, generations=500, alpha=1, beta=1, evaporation_rate=0.1):
		super().__init__(puzzle, size)
		self.num_ants = num_ants
		self.generations = generations
		self.evaporation_rate = evaporation_rate
		self.numberoftries = 0
		self.alpha = alpha
		self.beta = beta

	def solve(self):
		self.initial_propagate_constraints()
		pheromone_matrix = self.initialize_pheromone_matrix()
		best_solution = self.puzzle

		while True:
			ant_solutions = [self.local_solve(pheromone_matrix) for _ in range(self.num_ants)]
			pheromone_matrix = self.global_pheromone_update(pheromone_matrix, ant_solutions)
			
			ant_solutions.append(best_solution)
			best_solution = min(ant_solutions, key=self.fitness)

			if self.fitness(best_solution) == 0:
				return best_solution

	def initial_propagate_constraints(self):
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row][col] != 0:
					self.propagate_constraints(row, col, self.puzzle[row][col], self.puzzle)
		
	def initialize_pheromone_matrix(self):
		return np.ones((self.size, self.size, self.size)) / self.size

	def local_solve(self, pheromone_matrix):
		solution = [copy.deepcopy(row) for row in self.puzzle]
		ant_positions = [(i, j) for i in range(self.size) for j in range(self.size) if solution[i][j] == 0]
		random.shuffle(ant_positions)

		for row, col in ant_positions:
			valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, solution)]

			if not valid_numbers:  # Add this check
				return solution

			probabilities = self.calculate_probabilities(row, col, pheromone_matrix, valid_numbers)
			chosen_number = self.pick_number_by_probability(probabilities, valid_numbers)

			solution[row][col] = chosen_number
			solution = self.propagate_constraints(row, col, chosen_number, solution)  # Update the constraints after assigning a number
			self.local_update_pheromone(pheromone_matrix, row, col, chosen_number)

		return solution
	
	# original one 
	# def calculate_probabilities(self, row, col, pheromone_matrix, valid_numbers):
	# 	pheromones = [pheromone_matrix[row, col, num - 1] for num in valid_numbers]
	# 	total = sum(pheromones)
	# 	return [pheromone / total for pheromone in pheromones]

	def calculate_probabilities(self, row, col, pheromone_matrix, valid_numbers):
		pheromones = [pheromone_matrix[row, col, num - 1] for num in valid_numbers]
		heuristic_values = [1 / len(valid_numbers)] * len(valid_numbers)
		combined_values = [p ** self.alpha * h ** self.beta for p, h in zip(pheromones, heuristic_values)]
		total = sum(combined_values)
		return [value / total for value in combined_values]

	def pick_number_by_probability(self, probabilities, valid_numbers):
		return random.choices(valid_numbers, weights=probabilities, k=1)[0]

	def local_update_pheromone(self, pheromone_matrix, row, col, chosen_number):
		# Implement the local pheromone update rule
		pheromone_matrix[row, col, chosen_number - 1] *= (1 - self.evaporation_rate)
		pheromone_matrix[row, col, chosen_number - 1] += self.evaporation_rate / self.size
	
	def global_pheromone_update(self, pheromone_matrix, ant_solutions):
		best_solution = min(ant_solutions, key=self.fitness)
		best_fitness = self.fitness(best_solution)
		delta_pheromone = 1 / best_fitness if best_fitness != 0 else 1

		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row][col] == 0:
					num = best_solution[row][col] - 1
					pheromone_matrix[row, col, num] += delta_pheromone

		pheromone_matrix *= (1 - self.evaporation_rate)
		return pheromone_matrix

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

	def fitness(self, solution):
		total_errors = 0
		for row in range(self.size):
			for col in range(self.size):
				if not self.is_valid(row, col, solution[row][col], solution):
					total_errors += 1
		return total_errors

	def propagate_constraints(self, row, col, value, solution):
		if isinstance(solution[row][col], set):
			solution[row][col] = {value}
		else:
			solution[row][col] = value

		return solution


