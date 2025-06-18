import numpy as np
from numba import njit, uint8, int_, float64
from generate_full_board import generate_completed_board


@njit(uint8(uint8[:, :], int_, int_), cache=True)
def rules_count(board: np.ndarray, x: int, y: int) -> uint8:

    n_rows, n_columns = board.shape
    color = board[x, y]
    max_colors_val_for_row = n_columns // 2
    where_color_in_row = np.asarray(board[x] == color, dtype=np.bool_)
    color_count_in_row = np.count_nonzero(where_color_in_row)

    boardT = board.T
    max_colors_val_for_column = n_rows // 2
    where_color_in_column = np.asarray(boardT[y] == color, dtype=np.bool_)
    color_count_in_column = np.count_nonzero(where_color_in_column)

    rules_violated = 0
    if color_count_in_row > max_colors_val_for_row:
        rules_violated += 1
    if color_count_in_column > max_colors_val_for_column:
        rules_violated += 1


    if y>1 and where_color_in_row[y-2] and where_color_in_row[y-1]:
        rules_violated += 1
    if min(y,n_columns-y-1)>0 and where_color_in_row[y-1] and where_color_in_row[y+1]:
        rules_violated += 1
    if n_columns>y+2 and where_color_in_row[y+1] and where_color_in_row[y+2]:
        rules_violated += 1


    if x>1 and where_color_in_column[x-2] and where_color_in_column[x-1]:
        rules_violated += 1
    if min(x,n_rows-x-1)>0 and where_color_in_column[x-1] and where_color_in_column[x+1]:
        rules_violated += 1
    if n_rows>x+2 and where_color_in_column[x+1] and where_color_in_column[x+2]:
        rules_violated += 1

    if ((color_count_in_row < max_colors_val_for_row) and
        (color_count_in_column < max_colors_val_for_column)):
        return rules_violated
    if color_count_in_row == max_colors_val_for_row:
        to_check = board.T[where_color_in_row].T
        for k in range(n_rows):
            if k == x:
                continue
            if np.all(to_check[k] == color):
                rules_violated += 1
    if color_count_in_column == max_colors_val_for_column:
        to_check = board[where_color_in_column].T
        for k in range(n_columns):
            if k == y:
                continue
            if np.all(to_check[k] == color):
                rules_violated += 1
    return rules_violated


@njit(uint8[:, :](uint8[:, :]), cache=True)
def reliance_scores(board: np.ndarray) -> np.ndarray:
    violations_count = board.copy()
    for (x, y), true_color in np.ndenumerate(board):
        if not true_color:
            continue
        new_color = 3 - int(true_color)
        board[x, y] = new_color
        violations_count[x, y] = rules_count(board, x, y)
        board[x, y] = true_color
    return violations_count


@njit(uint8[:, :](uint8[:, :]), cache=True)
def _generate_game_board(board: np.ndarray) -> np.ndarray:
    d = board.shape[1]
    violations_count = reliance_scores(board)
    actual_violations = violations_count.ravel()[np.flatnonzero(violations_count)]
    last_removed_pair = np.asarray([d // 2, d // 2])
    while actual_violations.shape[0]:
        min_violations = actual_violations.min()
        where_min_violations = np.flatnonzero(violations_count == min_violations)
        coord_pairs = np.vstack(np.divmod(where_min_violations, d)).T
        distances = [np.sum(np.square(np.subtract(
            last_removed_pair,
            coord_pair))) for coord_pair in coord_pairs]
        max_distance = max(distances)
        remote_min_violations = where_min_violations[np.asarray(distances) == max_distance]
        chosen_cell = np.random.choice(remote_min_violations)
        last_removed_pair = np.divmod(chosen_cell, d)
        board[last_removed_pair] = 0
        last_removed_pair = np.asarray(last_removed_pair)
        violations_count = reliance_scores(board)
        actual_violations = violations_count.ravel()[np.flatnonzero(violations_count)]
    return board


@njit(float64(uint8[:, :]), cache=True)
def filled_fraction(partial_board: np.array) -> float:
    return np.divide(np.count_nonzero(partial_board), partial_board.size)


def generate_game_board(n: int) -> np.array:
    completed_board = generate_completed_board(n)
    return _generate_game_board(completed_board)
    # candidate_board = np.ones((1, 1), dtype=np.uint8)
    # i = 0
    # while True:
    #     if filled_fraction(candidate_board) <= 0.2:
    #         print(f"Phew, that took {i} tries!")
    #         return candidate_board
    #     completed_board = generate_completed_board(n)  # Entirely new filled solution, as this makes a difference!
    #     candidate_board = _generate_game_board(completed_board)
    #     i += 1


if __name__ == "__main__":
    print(generate_game_board(4))
    print(generate_game_board(6))
    print(generate_game_board(8))
    print(generate_game_board(10))
    print(generate_game_board(12))
    average_filled_fraction = sum(filled_fraction(generate_game_board(10)) for _ in range(100))
    print("Full Board = Numpy, Partial Board = Numpy: ", average_filled_fraction)

