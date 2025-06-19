import numpy as np
from numba import njit, uint8, int_, float64, bool_
from generate_full_board import generate_completed_board
from generate_sparse_gameboard import reliance_scores


@njit(bool_(uint8[:, :], int_, int_), cache=True)
def violation_detected(board: np.ndarray, x: int, y: int) -> bool:

    n_rows, n_columns = board.shape
    color = board[x, y]
    max_colors_val_for_row = n_columns // 2
    where_color_in_row = np.asarray(board[x] == color, dtype=np.bool_)
    color_count_in_row = np.count_nonzero(where_color_in_row)

    boardT = board.T
    max_colors_val_for_column = n_rows // 2
    where_color_in_column = np.asarray(boardT[y] == color, dtype=np.bool_)
    color_count_in_column = np.count_nonzero(where_color_in_column)

    if color_count_in_row > max_colors_val_for_row:
        return True
    if color_count_in_column > max_colors_val_for_column:
        return True


    if y>1 and where_color_in_row[y-2] and where_color_in_row[y-1]:
        return True
    if min(y,n_columns-y-1)>0 and where_color_in_row[y-1] and where_color_in_row[y+1]:
        return True
    if n_columns>y+2 and where_color_in_row[y+1] and where_color_in_row[y+2]:
        return True


    if x>1 and where_color_in_column[x-2] and where_color_in_column[x-1]:
        return True
    if min(x,n_rows-x-1)>0 and where_color_in_column[x-1] and where_color_in_column[x+1]:
        return True
    if n_rows>x+2 and where_color_in_column[x+1] and where_color_in_column[x+2]:
        return True

    if ((color_count_in_row < max_colors_val_for_row) and
        (color_count_in_column < max_colors_val_for_column)):
        return False
    if color_count_in_row == max_colors_val_for_row:
        to_check = board.T[where_color_in_row].T
        for k in range(n_rows):
            if k == x:
                continue
            if np.all(to_check[k] == color):
                return True
    if color_count_in_column == max_colors_val_for_column:
        to_check = board[where_color_in_column].T
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
        # subsequent_violation_counts[x, y] = reliance_scores(board).sum()
        subsequent_violation_counts[x, y] = violation_locations(board).sum()
        board[x, y] = true_color
    return subsequent_violation_counts

@njit(int_(uint8[:, :], uint8[:, :]), cache=True)
def ambiguity_count(full_board: np.ndarray, partial_board: np.ndarray) -> int:
    ambiguities_encountered = 0
    for (x, y), current_color in np.ndenumerate(partial_board):
        if current_color:
            continue
        true_color = full_board[x, y]
        false_color = 3 - int(true_color)
        partial_board[x, y] = false_color
        if not violation_detected(partial_board, x, y):
            ambiguities_encountered += 1
        partial_board[x, y] = 0
    return ambiguities_encountered

@njit(int_[:, :](uint8[:, :], uint8[:, :]), cache=True)
def subsequent_ambiguity_counts(full_board: np.ndarray, partial_board: np.ndarray) -> np.ndarray:
    ambiguity_count_if_blanked = np.zeros(partial_board.shape, dtype=np.int_)
    for (x, y), true_color in np.ndenumerate(partial_board):
        if not true_color:
            continue
        new_color = 3 - int(true_color)
        partial_board[x, y] = new_color
        if violation_detected(partial_board, x, y):
            partial_board[x, y] = 0
            ambiguity_count_if_blanked[x,y] = 1 + ambiguity_count(full_board, partial_board)
        partial_board[x, y] = true_color
    return ambiguity_count_if_blanked

# @njit(uint8[:, :](uint8[:, :]), cache=True)
# def _generate_game_board(board: np.ndarray) -> np.ndarray:
#     d = board.shape[1]
#     violations_places = violation_locations(board)
#     flat_violation_locations = np.flatnonzero(violations_places)
#     while flat_violation_locations.shape[0]:
#         next_violations = subsequent_violation_counts(board).ravel()[flat_violation_locations]
#         min_next_violation = next_violations.min()
#         where_min_next_violation = flat_violation_locations[next_violations == min_next_violation]
#         chosen_cell = np.random.choice(where_min_next_violation)
#         last_removed_pair = np.divmod(chosen_cell, d)
#         board[last_removed_pair] = 0
#         last_color = np.random.randint(1,3)
#         violations_places = violation_locations(board)
#         violations_places = np.logical_and(violations_places, board == last_color)
#         flat_violation_locations = np.flatnonzero(violations_places)
#     return board

# @njit(uint8[:, :](uint8[:, :]), cache=True)
# def _generate_game_board(full_board: np.ndarray) -> np.ndarray:
#     partial_board = full_board.copy()
#     how_ambiguous_each = subsequent_ambiguity_counts(full_board, partial_board)
#     max_ambiguity = how_ambiguous_each.max()
#     while max_ambiguity > 0:
#         # sorted_ambiguities = np.unique(how_ambiguous_each)
#         # if sorted_ambiguities[0]:
#         #     min_ambiguity = sorted_ambiguities[0]
#         # else:
#         #     min_ambiguity = sorted_ambiguities[1]
#         where_max_ambiguity = np.flatnonzero(how_ambiguous_each == max_ambiguity)
#         chosen_cell = np.random.choice(where_max_ambiguity)
#         partial_board.flat[chosen_cell] = 0
#         how_ambiguous_each = subsequent_ambiguity_counts(full_board, partial_board)
#         max_ambiguity = how_ambiguous_each.max()
#     return partial_board

# @njit(uint8[:, :](uint8[:, :]), cache=True)
# def _generate_game_board(full_board: np.ndarray) -> np.ndarray:
#     """We select cells to drop based on least number of rule violations, but then further sort by maximum induced ambiguity."""
#     partial_board = full_board.copy()
#     violations_count = reliance_scores(partial_board)
#     actual_violations = violations_count.ravel()[np.flatnonzero(violations_count)]
#     while actual_violations.shape[0]:
#         min_violations = actual_violations.min()
#         where_min_violations = np.flatnonzero(violations_count == min_violations)
#         for_ambiguity_count_sorting = np.zeros(full_board.shape, dtype=np.int_).ravel()
#         for flat_index in where_min_violations:
#             true_color = partial_board.flat[flat_index]
#             partial_board.flat[flat_index] = 0
#             for_ambiguity_count_sorting[flat_index] = 1 + ambiguity_count(full_board, partial_board)
#             partial_board.flat[flat_index] = true_color
#         max_ambiguity = for_ambiguity_count_sorting.max()
#         # sorted_ambiguities = np.unique(for_ambiguity_count_sorting)
#         # if sorted_ambiguities[0]:
#         #     min_ambiguity = sorted_ambiguities[0]
#         # else:
#         #     min_ambiguity = sorted_ambiguities[1]
#         where_max_ambiguous = np.flatnonzero(for_ambiguity_count_sorting == max_ambiguity)
#         chosen_cell = np.random.choice(where_max_ambiguous)
#         partial_board.flat[chosen_cell] = 0
#         violations_count = reliance_scores(partial_board)
#         actual_violations = violations_count.ravel()[np.flatnonzero(violations_count)]
#     return partial_board

@njit(uint8[:, :](uint8[:, :]), cache=True)
def _generate_game_board(full_board: np.ndarray) -> np.ndarray:
    """We select cells to drop based on least number of rule violations, but then further sort by maximum number of cells that can still be blanked"""
    partial_board = full_board.copy()
    violations_count = reliance_scores(partial_board)
    actual_violations = violations_count.ravel()[np.flatnonzero(violations_count)]
    while actual_violations.shape[0]:
        min_violations = actual_violations.min()
        where_min_violations = np.flatnonzero(violations_count == min_violations)
        for_ambiguity_count_sorting = np.zeros(full_board.shape, dtype=np.int_).ravel()
        for flat_index in where_min_violations:
            true_color = partial_board.flat[flat_index]
            partial_board.flat[flat_index] = 0
            for_ambiguity_count_sorting[flat_index] = 1 + violation_locations(partial_board).sum()
            partial_board.flat[flat_index] = true_color
        max_ambiguity = for_ambiguity_count_sorting.max()
        # sorted_ambiguities = np.unique(for_ambiguity_count_sorting)
        # if sorted_ambiguities[0]:
        #     min_ambiguity = sorted_ambiguities[0]
        # else:
        #     min_ambiguity = sorted_ambiguities[1]
        where_max_ambiguous = np.flatnonzero(for_ambiguity_count_sorting == max_ambiguity)
        chosen_cell = np.random.choice(where_max_ambiguous)
        partial_board.flat[chosen_cell] = 0
        violations_count = reliance_scores(partial_board)
        actual_violations = violations_count.ravel()[np.flatnonzero(violations_count)]
    return partial_board

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
    average_filled_fraction = sum(filled_fraction(generate_game_board(10)) for _ in range(100))
    print("Average filled fraction: {}".format(average_filled_fraction))