from src.main_2D import SudokuPuzzle

class BacktrackingSudoku(SudokuPuzzle):
    def __init__(self, puzzle, size):
        super().__init__(puzzle, size)
        self.numberoftries = 0

    def solve(self, puzzle):
        self.numberoftries += 1

        if self.is_solved(puzzle):
            return puzzle

        row, col = self.select_unassigned_variable(puzzle)
        for value in range(1, self.size + 1):
            if self.is_valid(row, col, value):
                puzzle[row][col] = value
                next_puzzle = self.propagate_constraints(puzzle, row, col, value)
                if next_puzzle is not None:
                    result = self.search(next_puzzle)
                    if result:
                        return result
                puzzle[row][col] = 0
        return None

    def is_solved(self, puzzle):
        for row in range(self.size):
            for col in range(self.size):
                if puzzle[row][col] == 0:
                    return False
        return True

    def select_unassigned_variable(self, puzzle):
        for row in range(self.size):
            for col in range(self.size):
                if puzzle[row][col] == 0:
                    return row, col
        return None

    def propagate_constraints(self, puzzle, row, col, value):
        new_puzzle = [r[:] for r in puzzle]
        new_puzzle[row][col] = value

        for r in range(self.size):
            for c in range(self.size):
                if new_puzzle[r][c] == 0 and not self.is_valid(r, c, value):
                    new_puzzle[r][c] = -1

        return new_puzzle if all(cell != -1 for row in new_puzzle for cell in row) else None


size =16

puzzle = [
	[9,0,3,0,   0,2,0,8,    4,0,14,0,    0,1,0,7],
	[0,0,11,0,  0,0,9,4,    12,15,0,0,   0,5,0,0],
	[13,10,0,6, 0,0,16,0,   0,11,0,0,    8,0,12,3],
	[0,0,16,8,  11,14,12,0, 0,1,10,6,    2,15,0,0],
	
	[0,0,0,11,   7,0,0,0,    0,0,0,15,   10,0,0,0],
	[16,0,0,7,   0,9,3,0,    0,6,12,0,   15,0,0,13],
	[0,2,15,9,   0,8,0,0,    0,0,11,0,   6,7,1,0],
	[10,3,0,0,   0,0,0,11,   7,0,0,0,    0,0,5,16],
	
	[15,11,0,0,0,0,0,9,6,0,0,0,0,0,8,12],
	[0,14,12,13,0,1,0,0,0,0,5,0,11,3,15,0],
	[2,0,0,16,0,4,7,0,0,8,13,0,14,0,0,1],
	[0,0,0,1,3,0,0,0,0,0,0,12,4,0,0,0],
	
	[0,0,9,15,12,10,1,0,0,13,7,5,3,2,0,0],
	[3,6,0,5,0,0,2,0,0,12,0,0,1,0,9,14],
	[0,0,2,0,0,0,14,13,3,9,0,0,0,4,0,0],
	[6,0,8,0,0,3,0,5,1,0,4,0,0,10,0,11]
]

def backtracking(puzzle,size):
	solver = BacktrackingSudoku(puzzle, size)  # Pass the size parameter here
	solution = solver.solve(puzzle)
	print(solution, solver.numberoftries)

backtracking(puzzle,size)