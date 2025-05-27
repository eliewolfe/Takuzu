import numpy as np
# from itertools import combinations # Moved to generate_full_board.py
from typing import List # generate_valid_rows still needs this
# from functools import cache # Moved to generate_full_board.py
from generate_full_board import has_three_in_row, solve, generate_valid_rows # generate_valid_rows was also moved as it's a dependency of solve


def generate_completed_board(n: int) -> np.ndarray:
    valid_rows = list(np.random.permutation(generate_valid_rows(n)).astype(int)+1) # generate_valid_rows is now imported
    initial_grid = np.zeros((n,n), dtype=int)
    initial_grid[0] = valid_rows.pop()
    solution = solve(grid=initial_grid,
                      valid_rows=valid_rows,
                      n_accomplished=1,
                      n_goal=n)
    if solution.shape == (n,n):
        return solution
    else:
        return generate_completed_board(n)


if __name__ == "__main__":
    # print(has_three_in_row([0,1,1,0,1,0,1,0,0,0]))
    # print("Valid rows of length 6")
    # for row in generate_valid_rows(6):
    #     print(row.astype(int))
    print(generate_completed_board(n=4))