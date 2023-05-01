from __init__ import *
from sudoku import SudokuPuzzle

class DancingLinksSudoku(SudokuPuzzle):
	def __init__(self, puzzle, size):
		super().__init__(puzzle, size)

	def solve(self):
		cover_matrix = self.to_exact_cover()
		columns = [(i, len(cover_matrix)) for i in range(4 * self.size * self.size)]
		dlx_solver = DLX(columns, cover_matrix)
		solution = dlx_solver.solve()
		if solution:
			solved_puzzle = [row[:] for row in self.puzzle]
			for row, col, value in solution:
				solved_puzzle[row // self.size][col % self.size] = value % self.size + 1
			return solved_puzzle
		else:
			return None
	
	def to_exact_cover(self):
		cover_matrix = []
		for row, col, value in itertools.product(range(self.size), range(self.size), range(1, self.size + 1)):
			if self.puzzle[row][col] == 0 or self.puzzle[row][col] == value:
				exact_cover_row = [0] * 4 * self.size * self.size
				exact_cover_row[row * self.size + col] = 1
				exact_cover_row[self.size * self.size + row * self.size + value - 1] = 1
				exact_cover_row[2 * self.size * self.size + col * self.size + value - 1] = 1
				box_idx = (row // self.subgrid_size) * self.subgrid_size + col // self.subgrid_size
				exact_cover_row[3 * self.size * self.size + box_idx * self.size + value - 1] = 1
				cover_matrix.append(tuple(exact_cover_row))
		return cover_matrix

	def create_cover_matrix(self):
		n = self.size
		n_sq = n * n
		n_four = n_sq * n_sq

		matrix = np.zeros((n_four, n_four * 4), dtype=int)

		row_constraint = np.eye(n_four, dtype=int)
		col_constraint = np.eye(n_four, dtype=int)
		box_constraint = np.eye(n_four, dtype=int)
		cell_constraint = np.eye(n_four, dtype=int)

		for r in range(n):
			for c in range(n):
				for num in range(n):
					idx = r * n_sq + c * n + num
					row_constraint[idx, r * n + num] = 1
					col_constraint[idx, n_sq + c * n + num] = 1
					box_constraint[idx, 2 * n_sq + (r // self.subgrid_size * self.subgrid_size + c // self.subgrid_size) * n + num] = 1
					cell_constraint[idx, 3 * n_sq + r * n_sq + c] = 1

		matrix[:, :n_four] = row_constraint
		matrix[:, n_four:2 * n_four] = col_constraint
		matrix[:, 2 * n_four:3 * n_four] = box_constraint
		matrix[:, 3 * n_four:] = cell_constraint

		# Remove rows from the cover matrix that correspond to filled cells in the puzzle
		for r in range(n):
			for c in range(n):
				num = self.puzzle[r][c]
				if num:
					idx = r * n_sq + c * n + (num - 1)
					matrix = np.delete(matrix, idx, axis=0)

		return matrix

	def convert_solution(self, solution):
		n = self.size
		n_sq = n * n

		solution_matrix = [row[:] for row in self.puzzle]

		for row in solution:
			r, c, num = row // n_sq, (row % n_sq) // n, (row % n_sq) % n
			solution_matrix[r][c] = num + 1

		return solution_matrix
