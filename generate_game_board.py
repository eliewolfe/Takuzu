import numpy as np
from _generate_completed_board import generate_completed_board
from check_unique import check_unique
from numba import njit

@njit
def create_emptier_board(partial_board: np.array) -> np.array:
    """
    Recursively removes cells from a completed or partially filled game board
    to create a puzzle, ensuring that each removal step maintains a state
    where the removed cell's original value was "forced" (i.e., flipping it
    would violate game rules, checked by `check_unique`).

    The function tries to empty cells one by one from a shuffled list of
    candidates. If emptying a cell (via `check_unique` returning True for its
    position) is valid, it recursively calls itself with the emptier board.
    If no more cells can be emptied according to this rule, the board state
    is returned.

    Args:
        partial_board (np.array): A NumPy array representing the current
                                  state of the game board. This could be a
                                  fully solved board or one from which some
                                  cells have already been removed.

    Returns:
        np.array: A NumPy array representing the board with some cells emptied,
                  forming a puzzle that should (ideally) have a unique solution
                  based on the "forced move" heuristic.
    """
    # Refactored candidate positions generation for Numba compatibility
    # np.argwhere returns a 2D array (N, 2) where N is number of non-zero elements
    coords = np.argwhere(partial_board != 0)
    
    # np.random.shuffle shuffles the array in-place along the first axis.
    # This is Numba-compatible.
    if coords.shape[0] == 0: # No non-zero elements to remove
        return partial_board
    np.random.shuffle(coords)

    for i in range(coords.shape[0]):
        x, y = coords[i, 0], coords[i, 1]
        
        # Pass coordinates as a tuple to check_unique, as it expects.
        # Numba can handle creating small tuples.
        position_tuple = (x, y)
        
        if check_unique(partial_board, position_tuple): # check_unique is already JITted
            new_board = partial_board.copy()
            new_board[x,y] = 0
            # Recursive call to a JITted function is fine.
            return create_emptier_board(new_board)
            
    return partial_board

def generate_game_board(n: int) -> np.array:
    return create_emptier_board(generate_completed_board(n))

if __name__ == "__main__":
    print(generate_game_board(4))
    print(generate_game_board(6))


