import numpy as np
from itertools import combinations
from typing import List
from functools import cache

def has_three_in_row(arr: np.ndarray) -> bool:
    """Check if array contains three consecutive 1s."""
    for i in range(len(arr) - 2):
        if np.array_equiv(arr[i], arr[i+1:i+3]):
            return True
    return False

@cache
def generate_valid_rows(n: int) -> List[np.ndarray]:
    """Generate all 10-bit vectors with exactly five 1s and no three consecutive 1s."""
    valid_rows = []
    assert not n % 2, "n must be a multiple of 2"
    half_n = int(n // 2)
    blank_row = np.zeros(n, dtype=bool)
    for ones_pos in combinations(range(n), half_n):
        row = blank_row.copy()
        row[list(ones_pos)] = True
        if not has_three_in_row(row):
            valid_rows.append(row)
    return valid_rows

def solve(grid: np.ndarray,
          valid_rows: List[np.ndarray],
          n_accomplished: int,
          n_goal) -> np.ndarray:
    """Backtracking generator for valid nxn matrices."""

    if n_accomplished == n_goal:
        return grid

    grid_as_set_of_tuples = set(map(tuple,grid.tolist()))
    for i, row in enumerate(valid_rows):
        if tuple(row) in grid_as_set_of_tuples:
            continue
        grid_guess = grid.copy()
        grid_guess[n_accomplished] = row
        remaining_rows = valid_rows[:i] + valid_rows[(i + 1):]
        further_testing_required = (n_accomplished > 1)
        if further_testing_required:
            # such that we are attempting a third row or higher
            to_check = grid_guess[n_accomplished-2:n_accomplished+1].T
            if has_three_in_row(to_check):
                continue
        further_testing_required = 2*(n_accomplished+2)//3 >= n_goal/2
        if further_testing_required:
            excess_found = False
            for y in range(n_goal):
                if np.sum(grid_guess[:,y] == 1) > n_goal/2:
                    excess_found = True
                    break
            if excess_found:
                continue
        further_testing_required = n_accomplished == n_goal - 2
        if further_testing_required:
            if len(set(map(tuple,grid_guess.T.astype(int)))) < n_goal:
                # duplicate column detected
                continue
        return solve(grid_guess,
                     remaining_rows,
                     n_accomplished + 1,
                     n_goal)
        print("ERROR -- this code should not be reachable")
        break
    return np.empty((0,0), dtype=int)


def generate_completed_board(n: int) -> np.ndarray:
    valid_rows = list(np.random.permutation(generate_valid_rows(n)).astype(int)+1)
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