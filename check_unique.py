import numpy as np
from numba import njit, bool_, uint8, int_
from generate_full_board import vec_has_three_in_row, generate_completed_board
from collections import defaultdict

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


@njit(uint8[:,:](uint8[:,:]), cache=False)
def _generate_game_board(full_board: np.ndarray) -> np.ndarray:
    d = full_board.shape[1]
    n_elements = full_board.size
    flat_coords = np.arange(n_elements)
    np.random.shuffle(flat_coords)
    xs, ys = np.divmod(flat_coords, d)
    partial_board = full_board.copy()
    i=0

    while i < n_elements:
        x = xs[i]
        y = ys[i]
        i += 1
        true_color = partial_board[x, y]
        if not true_color:
            continue
        new_color = 3 - true_color
        partial_board[x, y] = new_color

        if (not rules_2_and_3_check_on_row_for_both_colors(partial_board, x)
            or not rules_2_and_3_check_on_row_for_both_colors(partial_board.T, y)
            or vec_has_three_in_row(partial_board[x - 2:x + 2, y])
            or vec_has_three_in_row(partial_board[x, y - 2:y + 2])):
            partial_board[x, y] = 0
            i = 0
            continue
        else:
            partial_board[x, y] = true_color

    return partial_board

def _generate_game_board(full_board: np.ndarray) -> np.ndarray:
    d = full_board.shape[1]
    n_elements = full_board.size
    flat_coords = np.arange(n_elements)
    xs, ys = np.divmod(flat_coords, d)
    init_coord = np.random.choice(flat_coords)
    init_x = xs[init_coord]
    init_y = ys[init_coord]
    last_removed_pair = (init_x, init_y)

    partial_board = full_board.copy()
    partial_board[last_removed_pair] = 0 # Kickoff the deletion process
    coord_remaining = set(zip(xs, ys))
    coord_remaining.remove(last_removed_pair)
    coords_currently_remaining = coord_remaining.copy()

    while len(coords_currently_remaining) > 0:
        coord_dict = defaultdict(list)
        for pair in coords_currently_remaining:
            distance = np.sum(np.square(np.subtract(last_removed_pair,
                                                    pair)))
            coord_dict[distance].append(pair)
        distances = sorted(coord_dict.keys())
        while len(distances) > 0:
            distance = distances.pop()
            nearby_coords = coord_dict[distance]
            np.random.shuffle(nearby_coords)
            for (x,y) in nearby_coords:
                true_color = partial_board[x, y]
                new_color = 3 - true_color
                partial_board[x, y] = new_color
                if (not rules_2_and_3_check_on_row_for_both_colors(
                        partial_board, x)
                        or not rules_2_and_3_check_on_row_for_both_colors(
                            partial_board.T, y)
                        or vec_has_three_in_row(partial_board[x - 2:x + 2, y])
                        or vec_has_three_in_row(
                            partial_board[x, y - 2:y + 2])):
                    partial_board[x, y] = 0
                    last_removed_pair = (x, y)
                    coord_remaining.remove(last_removed_pair)
                    coords_currently_remaining = coord_remaining.copy()
                    distances = []
                    break
                else:
                    partial_board[x, y] = true_color
                    coords_currently_remaining.remove((x,y))
    return partial_board



def filled_fraction(partial_board: np.array) -> float:
    return np.divide(np.count_nonzero(partial_board), partial_board.size)

def generate_game_board(n: int) -> np.array:
    return _generate_game_board(generate_completed_board(n))
    # completed_board = generate_completed_board(n)
    # candidate_board = np.ones((1,1), dtype=np.uint8)
    # i = 0
    # while True:
    #     if filled_fraction(candidate_board) <= 0.45:
    #         print(f"Phew, that took {i} tries!")
    #         return candidate_board
    #     candidate_board = _generate_game_board(completed_board)
    #     i += 1

if __name__ == "__main__":
    print(generate_game_board(4))
    print(generate_game_board(6))
    print(generate_game_board(8))
    # print(generate_game_board(12))
    average_filled_fraction = sum(filled_fraction(generate_game_board(10)) for _ in range(100))
    print("Average filled fraction: {}".format(average_filled_fraction))