import numpy as np
from numba import njit, uint8, int_, float64
from generate_full_board import vec_has_three_in_row, generate_completed_board


@njit(uint8(uint8[:, :], int_, int_), cache=True)
def rules_count_on_rows(board: np.ndarray, x: int, y: int) -> uint8:
    rules_violated = 0
    rules_violated += vec_has_three_in_row(board[x - 2:x + 2, y])
    n_rows, n_columns = board.shape
    max_colors_val = n_columns // 2
    color = board[x, y]
    where_color = (board[x] == color)
    color_count = np.count_nonzero(where_color)
    if color_count > max_colors_val:
        rules_violated += 1
        return rules_violated
    elif (color_count < max_colors_val) or rules_violated:
        return rules_violated
    for k in range(n_rows):
        if k == x:
            continue
        if np.all(board[k, where_color] == color):
            return 1
    return 0


@njit(uint8(uint8[:, :], int_, int_), cache=True)
def rules_count(board: np.ndarray, x: int, y: int) -> uint8:
    return np.add(rules_count_on_rows(board, x, y),
                  rules_count_on_rows(board.T, y, x))


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
    average_filled_fraction = sum(filled_fraction(generate_game_board(8)) for _ in range(100))
    print("Average filled fraction: {}".format(average_filled_fraction))
