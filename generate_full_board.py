import numpy as np
from typing import List # For type hints
import functools # For @functools.cache
import itertools # For itertools.combinations
from numba import njit
# Removed: import numba (for numba.objmode)

@njit # has_three_in_row remains JITted
def has_three_in_row(arr_input: np.ndarray) -> bool:
    """
    Check if array contains three consecutive identical non-zero values.
    Handles 1D array (checks along its length) or 
    2D array (checks each row, assuming rows are the segments to check, e.g., column segments of length 3).
    arr_input is expected to be np.int64
    """
    if arr_input.ndim == 1:
        if len(arr_input) < 3:
            return False
        for i in range(len(arr_input) - 2):
            val = arr_input[i]
            if val != 0 and val == arr_input[i+1] and val == arr_input[i+2]:
                return True
    elif arr_input.ndim == 2: 
        for i in range(arr_input.shape[0]): 
            if arr_input.shape[1] != 3: 
                continue 
            val = arr_input[i, 0]
            if val != 0 and val == arr_input[i, 1] and val == arr_input[i, 2]:
                return True
    return False

# --- Reverted Combination Generation & valid_rows to Pure Python ---
@functools.cache # Reinstate functools.cache
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
    
    if n % 2 != 0: # Original assert implies n must be even
        # Or raise ValueError("n must be a multiple of 2")
        return valid_rows # Return empty list for consistency if n is odd

    half_n = n // 2
    blank_row_bool = np.zeros(n, dtype=bool)

    # Use itertools.combinations as in the original pure Python version
    for ones_pos in itertools.combinations(range(n), half_n):
        row_bool = blank_row_bool.copy()
        # ones_pos is a tuple of indices, convert to list for assignment if necessary,
        # though direct tuple indexing might also work for advanced indexing.
        # Using list() for safety as per original.
        row_bool[list(ones_pos)] = True
        
        # has_three_in_row expects an int array
        if not has_three_in_row(row_bool.astype(np.int64)): 
            valid_rows.append(row_bool.astype(np.int64)) # Store as int64 arrays
            
    return valid_rows

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
    
    # Original logic for checking existing rows in the grid
    grid_as_set_of_tuples = set(map(tuple, grid[:n_accomplished].tolist()))

    for i in range(len(valid_rows)): 
        current_row_to_try = valid_rows[i] # This is a 1D NumPy array
        
        # Check if current_row_to_try (as a tuple) is already in the set of existing rows
        if tuple(current_row_to_try) in grid_as_set_of_tuples:
            continue

        grid_guess = grid.copy()
        grid_guess[n_accomplished, :] = current_row_to_try
        
        # Original logic for remaining_rows (Python list slicing and concatenation)
        remaining_rows = valid_rows[:i] + valid_rows[i+1:]
            
        # Check 3-in-a-row for columns if enough rows are placed
        if n_accomplished >= 2: 
            sub_grid_for_col_check = grid_guess[n_accomplished-2 : n_accomplished+1, :]
            if has_three_in_row(sub_grid_for_col_check.T):
                continue
        
        # Check for too many of one color in any column
        # Original condition: further_testing_required = 2*(n_accomplished+2)//3 >= n_goal/2
        # This was a heuristic. A direct check is more robust.
        # Let's use the more direct check developed during Numba refactoring.
        max_colors_val = n_goal // 2
        excess_color_in_col = False
        for col_idx in range(n_goal):
            col_sum_color1 = 0
            col_sum_color2 = 0
            for row_idx in range(n_accomplished + 1): # Include the newly placed row
                val = grid_guess[row_idx, col_idx]
                if val == 1:
                    col_sum_color1 += 1
                elif val == 2: 
                    col_sum_color2 += 1
            if col_sum_color1 > max_colors_val or col_sum_color2 > max_colors_val:
                excess_color_in_col = True
                break
        if excess_color_in_col:
            continue
            
        # Duplicate column check (only if grid is full)
        # Original condition: further_testing_required = n_accomplished == n_goal - 2
        # And then: if len(set(map(tuple,grid_guess.T.astype(int)))) < n_goal:
        if n_accomplished == n_goal - 1: 
            # Convert columns to tuples and put in a set to count unique columns
            temp_cols_as_tuples = set()
            for col_idx in range(n_goal):
                temp_cols_as_tuples.add(tuple(grid_guess[:, col_idx]))
            
            if len(temp_cols_as_tuples) < n_goal:
                continue # Duplicate column detected

        # Recursive call
        solution = solve(grid_guess, remaining_rows, n_accomplished + 1, n_goal)
        if solution.shape[0] == n_goal and solution.shape[1] == n_goal: 
             return solution

    return np.empty((0,0), dtype=np.int64) # Return empty if no solution from this path
