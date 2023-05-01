from __init__ import *

class SudokuPuzzle:
	def __init__(self, puzzle, size):
		self.originalpuzzle = puzzle
		self.puzzle = puzzle
		self.size = size
		self.subgrid_size = int(size**0.5)
		self.numof_is_valid_checks = 0

	def is_valid(self, row, col, value):
		self.numof_is_valid_checks += 1
		if value < 1 or value > self.size:
			return False
		
		for i in range(self.size):
			if (self.puzzle[row][i] == value and i != col) or (self.puzzle[i][col] == value and i != row):
				return False

		subgrid_row, subgrid_col = row // self.subgrid_size, col // self.subgrid_size
		for i in range(self.subgrid_size):
			for j in range(self.subgrid_size):
				temp_row = subgrid_row * self.subgrid_size + i
				temp_col = subgrid_col * self.subgrid_size + j
				if self.puzzle[temp_row][temp_col] == value and temp_row != row and temp_col != col:
					return False
		return True

	def check_sudoku(self):
		for i in range(0,self.size):
			for j in range(0,self.size):
				if not self.is_valid(i,j,self.puzzle[i][j]):
					logging.info("Invalid solution")
					return False
		logging.info("Valid solution")
		return True
	
	def check_sudoku_solution(self,solution):
		for i in range(0,self.size):
			for j in range(0,self.size):
				if not self.is_valid(i,j,solution[i][j]):
					logging.info("Invalid solution")
					return False
		logging.info("Valid solution")
		return True
	
	def is_valid_current_solution(self, solution, row, col, value):
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

	def solve(self):
		raise NotImplementedError("Subclass must implement the solve() method")

