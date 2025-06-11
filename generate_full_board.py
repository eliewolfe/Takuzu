import numpy as np
from typing import List # For type hints
import functools # For @functools.cache
import itertools # For itertools.combinations
from numba import njit, uint8, bool_, int_


@njit(bool_(uint8[:]), cache=True)
def vec_has_three_in_row(arr_input: np.ndarray) -> bool:
    """
    Check if array contains three consecutive identical non-zero values.
    Handles 1D array (checks along its length)
    """
    if len(arr_input) < 3:
        return False
    counter = 0
    last_seen_value = 0
    for val in arr_input.flat:
        if val == 0: # blank tile, reset counter
            counter = 0
            continue
        elif val == last_seen_value:
            counter += 1
            if counter == 3:
                return True
        else:
            last_seen_value = val
            counter = 1
    return False


@functools.cache  # caching better than njitting.
def generate_valid_rows(n: int) -> List[np.ndarray]:
    """
    Generates all valid rows for a square binary puzzle board of size 'n'.

    A valid row must satisfy two conditions:
    1. It must contain an equal number of 1s and 2s (or True/False, then converted to 1/2).
       Specifically, it must have n/2 of one color (e.g., 1s or True).
    2. It must not contain three consecutive identical values (e.g., not 1,1,1 or 2,2,2).
       (The `has_three_in_row` check is for non-zero values, so 0,0,0 is implicitly allowed by that helper).
       The rows generated here are boolean, then converted to int64 for `has_three_in_row`.

    This function is cached using @functools.cache to memoize results for a given 'n'.

    Args:
        n (int): The dimension of the board (number of cells in a row).
                 It is expected to be an even number.

    Returns:
        List[np.ndarray]: A list of valid rows. Each row is a 1D NumPy array
                          of int64 type, representing the binary values (e.g., 0s and 1s,
                          or potentially 1s and 2s if adapted later, though boolean source
                          implies 0s and 1s).
    """
    valid_rows: List[np.ndarray] = []

    if n % 2 != 0:  # Original assert implies n must be even
        raise ValueError("n must be a multiple of 2")
        # return valid_rows # Return empty list for consistency if n is odd

    half_n = n // 2
    blank_row_template = np.ones(n, dtype=np.uint8)

    # Use itertools.combinations as in the original pure Python version
    for ones_pos in itertools.combinations(range(n), half_n):
        candidate_row = blank_row_template.copy()
        # ones_pos is a tuple of indices, convert to list for assignment
        candidate_row[list(ones_pos)] = 2

        # has_three_in_row expects an int array
        if not vec_has_three_in_row(candidate_row):
            valid_rows.append(candidate_row)  # Store as uint8 arrays

    return valid_rows

@njit(bool_(uint8[:,:]), cache=True) # has_three_in_row remains JITted
def any_vec_has_last_three_in_row(arr_input: np.ndarray) -> bool:
    """
    2D array (checks each row, assuming rows are the segments to check, e.g., column segments of length 3).
    arr_input is expected to be np.int64
    """
    for (x,y,z) in arr_input[:,-3:]:
        if x != 0 and x == y == z:
            return True
    return False

@njit(bool_(uint8[:], int_), cache=True)
def vec_content_exceeds_limit(arr_input: np.ndarray, limit: int) -> bool:
    """
    Check if array contains more 1s or more 2s than the limit
    """
    if len(arr_input) < limit:
        return False
    ones_counter = 0
    twos_counter = 0
    for val in arr_input.flat:
        if val == 0:
            continue
        ones_counter += 2 - val  # increments by one only if val==1
        twos_counter += val - 1 # increments by one only if val==2
        if (ones_counter > limit) or (twos_counter > limit):
            return True
    return False

@njit(bool_(uint8[:,:], int_), cache=True)
def any_vec_content_exceeds_limit(arr_input: np.ndarray, limit: int) -> bool:
    for vec in arr_input:
        if vec_content_exceeds_limit(vec, limit):
            return True
    return False


# --- Reverted solve to Pure Python ---
def solve(grid: np.ndarray, valid_rows: List[np.ndarray], 
          n_accomplished: int, n_goal: int) -> np.ndarray:
    """
    Recursively attempts to solve or complete a binary puzzle board using backtracking.

    It tries to place valid rows one by one onto the `grid`. For each placement,
    it checks against board rules (no identical rows already placed, no three
    consecutive identical numbers in columns, balanced color counts in columns,
    and no identical columns upon completion).

    Args:
        grid (np.ndarray): The current state of the board (n_goal x n_goal).
                           Rows from 0 to n_accomplished-1 are considered fixed.
        valid_rows (List[np.ndarray]): A list of pre-generated valid rows
                                       (1D NumPy arrays) that can be used.
        n_accomplished (int): The number of rows already successfully placed
                              onto the grid. The function attempts to place a
                              row at index `n_accomplished`.
        n_goal (int): The target dimension of the board (e.g., if n_goal=6,
                      the board is 6x6).

    Returns:
        np.ndarray: If a solution is found, it returns the completed n_goal x n_goal
                    NumPy array. If no solution is found from the current path,
                    it returns an empty 2D NumPy array (shape (0,0)).
    """
    if n_accomplished == n_goal:
        return grid # Successfully filled the grid
    
    # No need to ever check for duplicate rows, impossible by construction.
    max_colors_val = n_goal // 2
    for i in range(len(valid_rows)): 
        current_row_to_try = valid_rows[i] # This is a 1D NumPy array
        remaining_rows = valid_rows[:i] + valid_rows[i+1:]
        grid[n_accomplished] = current_row_to_try
        subgrid_for_checking = grid[: n_accomplished + 1].T

        # Check 3-in-a-row for columns if enough rows are placed
        if n_accomplished < 2:
            pass
        elif any_vec_has_last_three_in_row(subgrid_for_checking):
            continue
        
        # Check for too many of one color in any column
        if 2*(n_accomplished+2)//3 < n_goal/2:
            pass
        elif any_vec_content_exceeds_limit(subgrid_for_checking, max_colors_val):
            continue
            
        # Duplicate column check (only if grid is almost full)
        if n_accomplished == n_goal - 1: 
            # Convert columns to tuples and put in a set to count unique columns
            temp_cols_as_tuples = set(map(tuple, subgrid_for_checking.tolist()))
            if len(temp_cols_as_tuples) < n_goal:
                continue # Duplicate column detected

        # If we get to this point in the code, row addition has been successful.
        # Recursive call
        return solve(grid, remaining_rows, n_accomplished + 1, n_goal)
    # Exiting the loop with n_accomplished not equalling n_goal means all valid rows have been explored and eliminated
    # and thus the construction failed. Returning an empty grid to signify restart from scratch required.
    return np.empty((0,0), dtype=np.uint8) # Return empty if no solution from this path


def generate_completed_board(n: int) -> np.ndarray:
    valid_rows = list(np.random.default_rng().permutation(generate_valid_rows(n))) # generate_valid_rows is now imported
    initial_grid = np.zeros((n,n), dtype=np.uint8)
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
    for n in range(2,7):
        print(f"Number of valid rows of length {2*n}: {len(generate_valid_rows(2*n))}")
    # for row in generate_valid_rows(6):
    #     print(row-1)
    for i in range(3):
        print(generate_completed_board(n=10)-1)