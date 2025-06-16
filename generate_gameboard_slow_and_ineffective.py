import numpy as np
from numba import njit, uint8, int_, float64, bool_
from generate_full_board import vec_has_three_in_row, generate_completed_board

@njit(uint8(uint8[:, :], int_, int_), cache=True)
def violation_detected(board: np.ndarray, x: int, y: int) -> uint8:

    n_rows, n_columns = board.shape


    color = board[x, y]

    max_colors_val_for_row = n_columns // 2
    where_color_in_row = np.asarray(board[x] == color)
    color_count_in_row = np.count_nonzero(where_color_in_row)
    if color_count_in_row > max_colors_val_for_row:
        return True

    boardT = board.T
    max_colors_val_for_column = n_rows // 2
    where_color_in_column = np.asarray(boardT[y] == color)
    color_count_in_column = np.count_nonzero(where_color_in_column)
    if color_count_in_column > max_colors_val_for_column:
        return True

    if color_count_in_row <= max_colors_val_for_row:
        if where_color_in_row[x-2] and where_color_in_row[x-1]:
            return True
        elif where_color_in_row[x-1] and where_color_in_row[x+1]:
            return True
        elif where_color_in_row[x+1] and where_color_in_row[x+2]:
            return True


    if color_count_in_column <= max_colors_val_for_column:
        if where_color_in_column[y-2] and where_color_in_column[y-1]:
            return True
        elif where_color_in_column[y-1] and where_color_in_column[y+1]:
            return True
        elif where_color_in_column[y+1] and where_color_in_column[y+2]:
            return True

    if ((color_count_in_row < max_colors_val_for_row) and
        (color_count_in_column < max_colors_val_for_column)):
        return False
    if color_count_in_row == max_colors_val_for_row:
        to_check = board.T[where_color_in_row]
        for k in range(n_rows):
            if k == x:
                continue
            if np.all(to_check[k] == color):
                return True
    if color_count_in_column == max_colors_val_for_column:
        to_check = board[where_color_in_column]
        for k in range(n_columns):
            if k == y:
                continue
            if np.all(to_check[k] == color):
                return True
    return False

@njit(bool_[:, :](uint8[:, :]), cache=True)
def violation_locations(board: np.ndarray) -> np.ndarray:
    violations_places = np.asarray(board > 1)
    for (x, y), true_color in np.ndenumerate(board):
        if not true_color:
            continue
        new_color = 3 - int(true_color)
        board[x, y] = new_color
        violations_places[x, y] = violation_detected(board, x, y)
        board[x, y] = true_color
    return violations_places

@njit(int_[:, :](uint8[:, :]), cache=True)
def subsequent_violation_counts(board: np.ndarray) -> np.ndarray:
    subsequent_violation_counts = np.zeros(board.shape, dtype=np.int_)
    for (x, y), true_color in np.ndenumerate(board):
        if not true_color:
            continue
        board[x, y] = 0
        subsequent_violation_counts[x, y] = np.count_nonzero(violation_locations(board))
        board[x, y] = true_color
    return subsequent_violation_counts


@njit(uint8[:, :](uint8[:, :]), cache=True)
def _generate_game_board(board: np.ndarray) -> np.ndarray:
    d = board.shape[1]
    violations_places = violation_locations(board)
    flat_violation_locations = np.flatnonzero(violations_places)
    while flat_violation_locations.shape[0]:
        next_violations = subsequent_violation_counts(board).ravel()[flat_violation_locations]
        min_next_violation = next_violations.min()
        where_min_next_violation = flat_violation_locations[next_violations == min_next_violation]
        chosen_cell = np.random.choice(where_min_next_violation)
        last_removed_pair = np.divmod(chosen_cell, d)
        board[last_removed_pair] = 0
        violations_places = violation_locations(board)
        flat_violation_locations = np.flatnonzero(violations_places)
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
    #     if filled_fraction(candidate_board) <= 1/3:
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
    average_filled_fraction = list(filled_fraction(generate_game_board(10)) for _ in range(100))
    print("Average filled fraction: {}".format(average_filled_fraction))