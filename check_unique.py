import numpy as np
from numba import njit, bool_, uint8, int_
from generate_full_board import vec_has_three_in_row

@njit(bool_(uint8[:,:], int_, int_), cache=True)
def rules_2_and_3_check_on_row_for_specific_color(board: np.ndarray, x: int, color: int) -> bool:
    n = board.shape[1]
    max_colors_val = n // 2
    where_color = (board[x] == color)
    color_count = np.count_nonzero(where_color)
    if color_count > max_colors_val:
        return False
    elif color_count < max_colors_val:
        return True
    for k in range(n):
        if k == x:
            continue
        if np.all(board[k,where_color] == color):
            return False
    return True

@njit(bool_(uint8[:,:], int_), cache=True)
def rules_2_and_3_check_on_row_for_both_colors(board: np.ndarray, x: int) -> bool:
    return (rules_2_and_3_check_on_row_for_specific_color(board, x, 1) and
            rules_2_and_3_check_on_row_for_specific_color(board, x, 2))



@njit(bool_(uint8[:,:], int_, int_), cache=True)
def check_unique(board: np.array, x: int, y: int) -> bool:
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
        x:                Row index of cell to check.
        y:                Column index of cell to check.

    Returns:
        bool: True if flipping the color at `position` would violate a rule
              (making the original color necessary for a unique solution from this state,
              or if emptying it is "forced" because the alternative is invalid).
              False if flipping the color does not immediately violate basic rules,
              meaning the cell's original value isn't immediately forced by this check.
    """

    bad_board = board.copy()
    true_color = bad_board[x,y]

    # # true color is never 0
    # if true_color == 0:
    #     return False

    # Numba does not like assertions
    # assert true_color in {1,2}, "Sanity check failed, grid not 0/1/2 valued."

    new_color = 3 - true_color
    bad_board[x,y] = new_color

    # Check for 3 in a row or 3 in a column
    if vec_has_three_in_row(bad_board[x-2:x+2, y]):
        return True # Made unique by leading to 3-in-a-row
    if vec_has_three_in_row(bad_board[x, y-2:y+2]):
        return True # Made unique by leading to 3-in-a-column

    if not rules_2_and_3_check_on_row_for_both_colors(bad_board, x):
        return True
    if not rules_2_and_3_check_on_row_for_both_colors(bad_board.T, y):
        return True
                
    return False
