from __init__ import *
from sudoku import SudokuPuzzle

class GeneticAlgorithmSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size, population_size=500, generations=1000, mutation_rate=0.1, crossover_rate=0.1):
		super().__init__(puzzle, size)
		self.population_size = population_size
		self.generations = generations
		self.mutation_rate = mutation_rate
		self.crossover_rate = crossover_rate
		self.numberoftries = 0
		self.solutions_generation = 0
	def solve(self):
		population = self.initialize_population()

		for i in range(self.generations):
			if i%100==0:
				print(f"iteration {i}")
			self.numberoftries +=1
			population = self.selection(population)
			population = self.crossover(population)
			population = self.mutation(population)

			best_individual = min(population, key=self.fitness)
			if self.fitness(best_individual) == 0:
				self.solutions_generation = i
				return best_individual

		# return best_individual
	
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

	def initialize_population(self):
		population = []
		for _ in range(self.population_size):
			individual = [row[:] for row in self.puzzle]
			for row in range(self.size):
				for col in range(self.size):
					if individual[row][col] == 0 and random.random() < self.crossover_rate:
						valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num,individual)]
						if valid_numbers:
							individual[row][col] = random.choice(valid_numbers)
			population.append(individual)
		return population

	def mutation(self, population):
		mutated_population = []
		for individual in population:
			mutated_individual = [row[:] for row in individual]
			for row in range(self.size):
				for col in range(self.size):
					if self.puzzle[row][col] == 0 and random.random() < self.mutation_rate:
						valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, mutated_individual)]
						if valid_numbers:
							mutated_individual[row][col] = random.choice(valid_numbers)
			mutated_population.append(mutated_individual)
		return mutated_population

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



class GeneticAlgorithmSudokuElitist(SudokuPuzzle):
	def __init__(self, puzzle, size, population_size=500, generations=1000, mutation_rate=0.1, crossover_rate=0.1):
		super().__init__(puzzle, size)
		self.population_size = population_size
		self.generations = generations
		self.mutation_rate = mutation_rate
		self.crossover_rate = crossover_rate
		self.numberoftries = 0
		self.solutions_generation = 0
	def solve(self):
		population = self.initialize_population()

		for i in range(self.generations):
			if i%100==0:
				print(f"iteration {i}")
			self.numberoftries +=1
			population = self.selection(population)
			population = self.crossover(population)
			population = self.mutation(population)

			best_individual = min(population, key=self.fitness)
			if self.fitness(best_individual) == 0:
				self.solutions_generation = i
				return best_individual

		# return best_individual
	
	def selection(self, population):
		selected_individuals = []

		# Add the best individual to the selected_individuals list
		best_individual = min(population, key=self.fitness)
		selected_individuals.append(best_individual)

		# Now, select the remaining individuals using the original selection process
		for _ in range(self.population_size - 1):
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

	def initialize_population(self):
		population = []
		for _ in range(self.population_size):
			individual = [row[:] for row in self.puzzle]
			for row in range(self.size):
				for col in range(self.size):
					if individual[row][col] == 0 and random.random() < self.crossover_rate:
						valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num,individual)]
						if valid_numbers:
							individual[row][col] = random.choice(valid_numbers)
			population.append(individual)
		return population

	def mutation(self, population):
		mutated_population = []
		for individual in population:
			mutated_individual = [row[:] for row in individual]
			for row in range(self.size):
				for col in range(self.size):
					if self.puzzle[row][col] == 0 and random.random() < self.mutation_rate:
						valid_numbers = [num for num in range(1, self.size + 1) if self.is_valid(row, col, num, mutated_individual)]
						if valid_numbers:
							mutated_individual[row][col] = random.choice(valid_numbers)
			mutated_population.append(mutated_individual)
		return mutated_population

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

