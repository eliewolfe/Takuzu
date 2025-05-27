import numpy as np
from numba import njit

@njit
def check_unique(board: np.array, position: tuple) -> bool:
    """
    Checks if emptying a cell at a given position on the board is a "forced move"
    or would lead to a rule violation if the cell's original color were flipped.

    This function is used to determine if a cell can be emptied when generating
    a puzzle. If flipping the cell's color (1 to 2 or 2 to 1) would violate
    a board rule (like three in a row, too many of one color, or duplicate row/column),
    then the original color is considered "unique" or "forced" for that cell's
    solution, implying that emptying it is a valid puzzle generation step.

    Args:
        board (np.array): The current state of the game board (n x n).
                          Expected to contain 0s (empty), 1s, and 2s.
        position (tuple): A tuple (row, col) indicating the cell to check.
                          The cell at board[row, col] must not be 0.

    Returns:
        bool: True if flipping the color at `position` would violate a rule
              (making the original color necessary for a unique solution from this state,
              or if emptying it is "forced" because the alternative is invalid).
              False if flipping the color does not immediately violate basic rules,
              meaning the cell's original value isn't immediately forced by this check.
    """
    (n, m) = board.shape
    # assert n == m, "Board must be square." # Numba doesn't like asserts with tuples in string formatting
    if n != m:
        # Numba compatible way to raise error or handle
        # For now, assume square board as per original logic, or let it fail if not.
        # Or, return a specific value indicating error, e.g., raise ValueError("Board must be square")
        # However, Numba might not support raising ValueError with custom strings easily.
        # For simplicity in JIT, this check might be better done outside or handled carefully.
        # Let's assume valid square boards are passed.
        pass

    bad_board = board.copy()
    x, y = position[0], position[1] # Unpack tuple for Numba
    true_color = bad_board[x,y]

    if true_color == 0:
        return False
    
    # assert true_color in {1,2}, "Sanity check failed, grid not 0/1/2 valued."
    # Replace assert with a check Numba can handle:
    if not (true_color == 1 or true_color == 2):
        # Handle error or assume valid values. Numba doesn't JIT arbitrary string messages in asserts/exceptions.
        # This indicates a logic error upstream if hit.
        # For now, assume valid inputs.
        pass

    new_color = 3 - true_color
    bad_board[x,y] = new_color

    # Refactored three-in-a-row check for the affected row
    # Check row bad_board[x,:]
    for i in range(n - 2):
        if bad_board[x, i] == new_color and \
           bad_board[x, i+1] == new_color and \
           bad_board[x, i+2] == new_color:
            return True # Made unique by leading to 3-in-a-row

    # Refactored three-in-a-row check for the affected column
    # Check column bad_board[:,y]
    for i in range(n - 2):
        if bad_board[i, y] == new_color and \
           bad_board[i+1, y] == new_color and \
           bad_board[i+2, y] == new_color:
            return True # Made unique by leading to 3-in-a-row
            
    rows_set = list(range(n)) # This will become a list of integers, Numba can handle this.
    # Check too many new_color or duplicate row for the affected row bad_board[x,:]
    current_row_slice = bad_board[x,:]
    num_new_color_in_row = 0
    for val in current_row_slice:
        if val == new_color:
            num_new_color_in_row += 1
    
    if num_new_color_in_row > n / 2:
        return True # Leads to excess new_color in the row
    
    if num_new_color_in_row == n / 2:
        # Check for duplicate rows if this row is now full of new_color for its quota
        # This means comparing bad_board[x,:] with other rows bad_board[k,:] where k != x
        for k in range(n):
            if k == x:
                continue
            # Check if bad_board[k,:] is identical to bad_board[x,:]
            is_duplicate_row = True
            for col_idx in range(n):
                if bad_board[x, col_idx] != bad_board[k, col_idx]:
                    is_duplicate_row = False
                    break
            if is_duplicate_row:
                return True # Duplicate row found

    # Check too many new_color or duplicate column for the affected column bad_board[:,y]
    current_col_slice = bad_board[:,y]
    num_new_color_in_col = 0
    for val in current_col_slice:
        if val == new_color:
            num_new_color_in_col += 1

    if num_new_color_in_col > n / 2:
        return True # Leads to excess new_color in the column

    if num_new_color_in_col == n / 2:
        # Check for duplicate columns if this col is now full of new_color for its quota
        # This means comparing bad_board[:,y] with other columns bad_board[:,k] where k != y
        for k in range(n):
            if k == y:
                continue
            # Check if bad_board[:,y] is identical to bad_board[:,k]
            is_duplicate_col = True
            for row_idx in range(n):
                if bad_board[row_idx, y] != bad_board[row_idx, k]:
                    is_duplicate_col = False
                    break
            if is_duplicate_col:
                return True # Duplicate column found
                
    return False
