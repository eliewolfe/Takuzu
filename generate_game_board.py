import numpy as np
from _generate_completed_board import generate_completed_board

def check_unique(board: np.array, position: tuple) -> bool:
    (n, m) = board.shape
    assert n == m, "Board must be square."
    bad_board = board.copy()
    (x,y) = position
    true_color = bad_board[x,y]
    if true_color == 0:
        # do not bother trying to drop an already-empty position
        return False
    assert true_color in {1,2}, "Sanity check failed, grid not 0/1/2 valued."
    new_color = 3-true_color
    bad_board[x,y] = new_color
    # test for two new color instances in a row
    three_in_a_row_pattern = str(new_color)*3
    if ''.join(map(str,bad_board[x,:])).count(three_in_a_row_pattern):
        # Made unique by virtue of leading to 3-in-a-row otherwise
        return True
    if ''.join(map(str,bad_board[:,y])).count(three_in_a_row_pattern):
        return True
    rows_set = list(range(n))
    # Check too many new_color or duplicate row
    row_to_check = bad_board[x,:]
    bit_mask = (row_to_check == new_color)
    number_of_new_color = int(np.sum(bit_mask))
    if number_of_new_color > n/2:
        # leads to excess new color
        return True
    elif number_of_new_color == n/2:
        other_rows = rows_set[:x] + rows_set[(x + 1):]
        for i in other_rows:
            spots_to_check = bad_board[i,bit_mask]
            if np.array_equiv(spots_to_check, new_color):
                # Another row also has full spots of the new color
                return True
    # Check duplicate columns
    row_to_check = bad_board[:,y]
    bit_mask = (row_to_check == new_color)
    number_of_new_color = int(np.sum(bit_mask))
    if number_of_new_color > n/2:
        # leads to excess new color
        return True
    elif number_of_new_color == n/2:
        other_rows = rows_set[:y] + rows_set[(y + 1):]
        for i in other_rows:
            spots_to_check = bad_board[bit_mask, i]
            if np.array_equiv(spots_to_check, new_color):
                # Another row also has full spots of the new color
                return True
    return False

def create_emptier_board(partial_board: np.array) -> np.array:
    candidate_positions = np.random.permutation(
        np.vstack(np.nonzero(partial_board)).T)
    candidate_positions = candidate_positions.astype(int).tolist()
    candidate_positions = list(map(tuple, candidate_positions))
    while len(candidate_positions) > 0:
        (x,y)  = candidate_positions.pop()
        if check_unique(partial_board, (x,y)):
            new_board = partial_board.copy()
            new_board[x,y] = 0
            return create_emptier_board(new_board)
    return partial_board

def generate_game_board(n: int) -> np.array:
    return create_emptier_board(generate_completed_board(n))

if __name__ == "__main__":
    print(generate_game_board(4))
    print(generate_game_board(6))


