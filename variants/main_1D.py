import random
import numpy as np

class SudokuPuzzle:
	def __init__(self, puzzle, size):
		self.puzzle = puzzle
		self.size = size
		self.subgrid_size = int(size**0.5)

	def is_valid(self, row, col, value):
		for i in range(self.size):
			if self.puzzle[row * self.size + i] == value or self.puzzle[i * self.size + col] == value:
				return False

		subgrid_row, subgrid_col = row // self.subgrid_size, col // self.subgrid_size
		for i in range(self.subgrid_size):
			for j in range(self.subgrid_size):
				if self.puzzle[(subgrid_row * self.subgrid_size + i) * self.size + (subgrid_col * self.subgrid_size + j)] == value:
					return False

		return True

	def solve(self):
		raise NotImplementedError("Subclass must implement the solve() method")

class BacktrackingSudoku(SudokuPuzzle):
	def solve(self):
		empty_cell = self.find_empty_cell()
		if not empty_cell:
			return True

		row, col = empty_cell
		for value in range(1, self.size + 1):
			if self.is_valid(row, col, value):
				self.puzzle[row * self.size + col] = value
				if self.solve():
					return self.puzzle
				self.puzzle[row * self.size + col] = 0
		return None

	def find_empty_cell(self):
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row * self.size + col] == 0:
					return row, col
		return None


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

			if self.fitness(best_bat) == 0:
				return best_bat

		return best_bat

	def initialize_bats(self):
		bats = []
		for _ in range(self.num_bats):
			bat = self.puzzle[:]
			for row in range(self.size):
				for col in range(self.size):
					if bat[row * self.size + col] == 0:
						valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num)]
						bat[row * self.size + col] = random.choice(valid_numbers)
			bats.append(bat)
		return bats

	def move_bat(self, bat):
		new_bat = bat[:]
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row * self.size + col] == 0 and random.random() < self.alpha:
					valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num)]
					new_bat[row * self.size + col] = random.choice(valid_numbers)
		return new_bat

	def local_search(self, bat, best_bat):
		new_bat = bat[:]
		for row in range(self.size):
			for col in range(self.size):
				if self.puzzle[row * self.size + col] == 0:
					valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num)]
					best_number = min(valid_numbers, key=lambda x: abs(x - best_bat[row * self.size + col]))
					new_bat[row * self.size + col] = best_number
		return new_bat

	def fitness(self, bat):
		total_errors = 0
		for row in range(self.size):
			for col in range(self.size):
				if not self.is_valid(row, col, bat[row * self.size + col]):
					total_errors += 1
		return total_errors
		pass

class GeneticAlgorithmSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size, population_size=500, generations=1000, mutation_rate=0.1):
		super().__init__(puzzle, size)
		self.population_size = population_size
		self.generations = generations
		self.mutation_rate = mutation_rate

	def solve(self):
		population = self.initialize_population()

		for _ in range(self.generations):
			population = self.selection(population)
			population = self.crossover(population)
			population = self.mutation(population)

			best_individual = min(population, key=self.fitness)
			if self.fitness(best_individual) == 0:
				return best_individual

		return "No solution Found"

	def initialize_population(self):
		population = []
		for _ in range(self.population_size):
			individual = [row[:] for row in self.puzzle]
			for row in range(self.size):
				for col in range(self.size):
					if individual[row][col] == 0:
						valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num)]
						individual[row][col] = random.choice(valid_numbers)
			population.append(individual)
		return population

	def selection(self, population):
		selected_individuals = []
		for _ in range(self.population_size):
			parents = random.sample(population, 2)
			selected_individuals.append(min(parents, key=self.fitness))
		return selected_individuals

	def crossover(self, population):
		offsprings = []
		for _ in range(self.population_size // 2):
			parents = random.sample(population, 2)
			offspring1 = [[0]*self.size for _ in range(self.size)]
			offspring2 = [[0]*self.size for _ in range(self.size)]
			for row in range(self.size):
				crossover_point = random.randint(0, self.size)
				offspring1[row] = parents[0][row][:crossover_point] + parents[1][row][crossover_point:]
				offspring2[row] = parents[1][row][:crossover_point] + parents[0][row][crossover_point:]
			offsprings.extend([offspring1, offspring2])
		return offsprings

	def mutation(self, population):
		mutated_population = []
		for individual in population:
			mutated_individual = [row[:] for row in individual]
			for row in range(self.size):
				for col in range(self.size):
					if self.puzzle[row][col] == 0 and random.random() < self.mutation_rate:
						valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num)]
						mutated_individual[row][col] = random.choice(valid_numbers)
			mutated_population.append(mutated_individual)
		return mutated_population


	def fitness(self, individual):
		total_errors = 0
		for row in range(self.size):
			for col in range(self.size):
				if not self.is_valid(row, col, individual[row][col]):
					total_errors += 1
		return total_errors

import random
import numpy as np

class AntColonyOptimizationSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size, num_ants=100, generations=500, alpha=1, beta=1, evaporation_rate=0.1):
		super().__init__(puzzle, size)
		self.num_ants = num_ants
		self.generations = generations
		self.alpha = alpha
		self.beta = beta
		self.evaporation_rate = evaporation_rate

	def solve(self):
		pheromone_matrix = self.initialize_pheromone_matrix()

		for _ in range(self.generations):
			ant_solutions = [self.construct_solution(pheromone_matrix) for _ in range(self.num_ants)]
			pheromone_matrix = self.update_pheromone_matrix(pheromone_matrix, ant_solutions)

			best_solution = min(ant_solutions, key=self.fitness)
			if self.fitness(best_solution) == 0:
				return best_solution

		return best_solution

	def initialize_pheromone_matrix(self):
		return np.ones((self.size, self.size, self.size)) / self.size

	def construct_solution(self, pheromone_matrix):
		solution = self.puzzle[:]
		for row in range(self.size):
			for col in range(self.size):
				if solution[row * self.size + col] == 0:
					valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num)]
					probabilities = self.calculate_probabilities(row, col, pheromone_matrix, valid_numbers)
					solution[row * self.size + col] = self.pick_number_by_probability(probabilities, valid_numbers)
		return solution

	def calculate_probabilities(self, row, col, pheromone_matrix, valid_numbers):
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
					if self.puzzle[row * self.size + col] == 0:
						num = solution[row * self.size + col] - 1
						pheromone_matrix[row, col, num] += delta_pheromone
		return pheromone_matrix

	def fitness(self,solution):
		total_errors = 0
		for row in range(self.size):
			for col in range(self.size):
				if not self.is_valid(row, col, solution[row * self.size + col]):
					total_errors += 1
		return total_errors



puzzle = [
	0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 13, 0, 0,
	0, 4, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 6, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0,
	2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0,
	0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0
]
# Choose the desired algorithm by creating an instance of the corresponding class
size = 16  # or 25 for 25x25 puzzles

# BACKTRACKING
solver = BacktrackingSudoku(puzzle, size)  # Pass the size parameter here
# Solve the puzzle using the chosen algorithm
solution = solver.solve()

# ACO
# size = 16  # or 25 for 25x25 puzzles
# num_ants = 100
# generations = 1000
# alpha = 1
# beta = 1
# evaporation_rate = 0.1
# solver = AntColonyOptimizationSudoku(puzzle, size, num_ants, generations, alpha, beta, evaporation_rate)
# solution = solver.solve()

# BAT ALGORITHM
# size = 16  # or 25 for 25x25 puzzles
# num_bats = 100
# generations = 2000
# alpha = 0.9
# gamma = 0.9

# solver = BatAlgorithmSudoku(puzzle, size, num_bats, generations, alpha, gamma)
# solution = solver.solve()

# GENETIC ALGORITHM
# size = 16  # or 25 for 25x25 puzzles
# population_size = 1000
# generations = 5000
# mutation_rate = 0.05

# solver = GeneticAlgorithmSudoku(puzzle, size, population_size, generations, mutation_rate)
# solution = solver.solve()

print(solution)
