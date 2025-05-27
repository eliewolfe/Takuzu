import unittest
import numpy as np
from check_unique import check_unique # Assuming check_unique.py is in the same directory or accessible

class TestCheckUnique(unittest.TestCase):

    def test_forced_move_three_in_a_row(self):
        """Test that flipping creates three-in-a-row (forced move)."""
        # Board:
        # 1 1 X 0
        # 0 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        # If X is 2, flipping it to 1 would make 1 1 1 0
        board = np.array([
            [1, 1, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.int64)
        position = (0, 2) # Cell with '2'
        self.assertTrue(check_unique(board, position), "Flipping to 1 should create three 1s in a row")

        # Test for column
        board_col = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0], # If this 2 is flipped to 1 -> three 1s in col 0
            [0, 0, 0, 0]
        ], dtype=np.int64)
        position_col = (2, 0)
        self.assertTrue(check_unique(board_col, position_col), "Flipping to 1 should create three 1s in a col")

    def test_forced_move_imbalance(self):
        """Test that flipping creates color imbalance (forced move)."""
        # Board 4x4, max 2 of each color.
        # 1 1 0 0
        # 2 X 0 0  <- If X is 1, flipping it to 2 would make row [2,2,0,0] (ok)
        # 0 0 0 0     but if X is 2, flipping it to 1 would make row [2,1,0,0] (ok)
        # This test needs a board where flipping *does* create imbalance.
        # 1 1 2 X  <- If X is 2, flipping to 1: [1,1,2,1] (ok for row)
        # Let board be:
        # 1 1 0 Y  (Y is the test position)
        # 2 2 1 0
        # 0 0 0 0
        # 0 0 0 0
        # If Y is 1, flipping to 2 makes row [1,1,0,2] (ok)
        # If Y is 2, flipping to 1 makes row [1,1,0,1] (ok)
        # The logic of check_unique: true_color = board[x,y], new_color = 3-true_color
        # bad_board[x,y] = new_color. Then check rules for bad_board.
        # If Y is 1 (true_color=1), new_color=2. bad_board[0,3]=2. Row is [1,1,0,2]. This is fine.
        # So check_unique should be False if this state is fine.

        # Example: If flipping to new_color leads to > n/2 of new_color in row
        # 1 2 2 0 -> if we consider flipping the 1 (pos 0,0)
        # true_color = 1, new_color = 2. bad_board[0,0] = 2. Row becomes [2,2,2,0]
        # This is a three-in-a-row, which is already tested.
        
        # Let's make a direct imbalance case:
        # Board: (n=4, n/2=2)
        # 1 X 1 0   X is at (0,1). Current colors: two 1s.
        # 0 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        # If X is 1 (true_color=1), new_color=2. bad_board[0,1]=2. Row is [1,2,1,0]. (1x'2', 2x'1'). OK.
        # check_unique should return False here.

        # If X is 2 (true_color=2), new_color=1. bad_board[0,1]=1. Row is [1,1,1,0]. (3x'1').
        # This is a three-in-a-row, already covered. check_unique returns True.

        # Test for imbalance:
        # Board:
        # 1 1 2 X (X at 0,3)
        # 0 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        # If X is 1 (true_color=1). new_color=2. bad_board[0,3]=2. Row: [1,1,2,2]. Bal: two 1s, two 2s. OK.
        # Here check_unique should be False.

        # Let's construct a case where flipping *causes* imbalance of the *new_color*
        # n=4, n/2=2
        # 1 0 0 Y  (Y at 0,3)
        # If Y is 1 (true_color=1), new_color=2. bad_board[0,3]=2. Row: [1,0,0,2]. OK. (False from check_unique)
        # If Y is 2 (true_color=2), new_color=1. bad_board[0,3]=1. Row: [1,0,0,1]. OK. (False from check_unique)

        # Consider:
        # 1 X 0 0
        # 1 0 0 0
        # 2 0 0 0
        # 2 0 0 0
        # Position (0,1), value is 1 (true_color). new_color = 2.
        # bad_board[0,1] = 2. Col 1: [2,0,0,0]. OK.
        # So check_unique should be False for (0,1) if board[0,1]=1
        board = np.array([
            [1, 1, 0, 0], # two 1s
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.int64)
        # If we test position (0,0) (value 1). new_color = 2. bad_board[0,0]=2. Row is [2,1,0,0]. OK.
        # This means check_unique for (0,0) should be False.

        # A case that *does* cause imbalance:
        # Board:
        # 1 2 0 0 (pos (0,0) is 1)
        # 2 0 0 0
        # 2 0 0 0
        # 0 0 0 0
        # Test pos (0,0), true_color=1. new_color=2.
        # bad_board[0,0] = 2. Row becomes [2,2,0,0]. (OK for row counts)
        # Column 0 becomes [2,2,2,0]. This is a 3-in-a-row.
        # This example is for 3-in-a-row, not imbalance.

        # Imbalance by flipping:
        # Target: new_color count becomes > n/2
        # n=4, n/2=2
        # Board:
        # 1 2 0 0
        # 1 0 0 0
        # 1 0 0 0 <- If this 1 (pos 2,0) is flipped to 2, col 0 becomes [1,1,2,2]. OK.
        # 0 0 0 0
        board_imb = np.array([
            [2, 0, 0, 0], # Cell (0,0) is 2. new_color=1.
            [1, 0, 0, 0], # If (0,0) becomes 1, col 0: [1,1,1,1] -> too many 1s
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=np.int64)
        position_imb = (0,0) # Value is 2. If flipped to 1, col 0 gets four 1s.
        self.assertTrue(check_unique(board_imb, position_imb), "Flipping 2->1 at (0,0) should make col 0 have four 1s (imbalance)")
        
    def test_forced_move_duplicate_row_or_col(self):
        """Test that flipping creates a duplicate row/column (forced move)."""
        # n=4
        # Row 0: 1 2 1 2
        # Row 1: X Y Z W (test position is one of these, e.g. X)
        # Row 2: 1 2 1 2 (already identical to Row 0)
        # Row 3: 0 0 0 0
        # If Row 1 becomes [1,2,1,2] after flipping X, then it's a forced move.
        # Let Row 1 be [1,1,1,2]. Pos (1,1) is 1. true_color=1, new_color=2.
        # bad_board[1,1]=2. Row 1 becomes [1,2,1,2]. This matches Row 0.
        board = np.array([
            [1, 2, 1, 2],
            [1, 1, 1, 2], # Test pos (1,1), value 1. Flip to 2 -> [1,2,1,2]
            [0, 0, 0, 0], # Placeholder, not part of this specific duplicate check
            [0, 0, 0, 0]
        ], dtype=np.int64)
        position = (1, 1) 
        # After flip: bad_board[1,:] is [1,2,1,2]. This is NOT a duplicate of board[0,:] YET by check_unique's logic.
        # check_unique's duplicate check: "if num_new_color_in_row == n / 2:" THEN compares with other rows.
        # The row [1,2,1,2] has two 1s and two 2s. So num_new_color_in_row (for color 2) is 2 (n/2).
        # It then compares this [1,2,1,2] with board[0,:], which is [1,2,1,2]. They are identical. So True.
        self.assertTrue(check_unique(board, position), "Flipping (1,1) should make row 1 a duplicate of row 0")

    def test_not_forced_move(self):
        """Test a scenario where emptying a cell is not forced."""
        board = np.array([
            [1, 2, 1, 0],
            [2, 1, 2, 0],
            [1, 2, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.int64)
        position = (0, 0) # Value is 1. Flipping to 2 makes row [2,2,1,0]. No immediate violation.
        self.assertFalse(check_unique(board, position))

    def test_empty_cell(self):
        """Test with an already empty cell (should return False)."""
        board = np.array([
            [1, 2, 0, 0], # Position (0,2) is empty
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.int64)
        position = (0, 2)
        self.assertFalse(check_unique(board, position))

    def test_small_board_edge_case(self):
        """Test with a very small board (e.g., 2x2, though rules apply from 4x4 usually).
           check_unique itself doesn't restrict board size beyond what its rules imply (e.g. n/2).
           A 2x2 board with n/2 = 1.
        """
        # Board:
        # 1 X  (X at 0,1)
        # 0 0
        # If X is 1 (true_color=1), new_color=2. bad_board[0,1]=2. Row [1,2]. Count of 2 is 1 (n/2). OK.
        # No three-in-a-row possible.
        board = np.array([
            [1, 1],
            [2, 0]
        ], dtype=np.int64)
        position = (0, 1) # Value is 1. Flip to 2 makes row [1,2]. This is fine.
        self.assertFalse(check_unique(board, position))
        
        # Forced case on 2x2
        # 1 X (X at 0,1)
        # 1 0
        # If X is 2 (true_color=2), new_color=1. bad_board[0,1]=1. Row [1,1].
        # Col 0 is [1,1]. Col 1 is [1,0].
        # Row 0 ([1,1]) is full of 1s (n/2=1).
        # Row 1 ([1,0]) is not identical.
        # Col 0 ([1,1]) is full of 1s.
        # No other col is identical.
        # No 3-in-a-row possible.
        # It seems hard to make a forced move on 2x2 with current rules other than direct rule violation.
        # Consider this:
        # 1 X  (X is 2 at (0,1))
        # 2 Y  (Y is 1 at (1,1))
        # If we test (0,1) (value 2). Flip to 1. Board becomes:
        # 1 1
        # 2 Y
        # Row (0) is [1,1]. This is fine (two 1s, n/2=1).
        # Col (1) is [1,Y].
        # No 3-in-a-row. This flip is ok. So check_unique is False.
        board_2x2_forced = np.array([
            [1,2], # Test (0,1) = 2. Flip to 1. Row becomes [1,1]. Col 1 becomes [1,1].
            [1,1]  # Now Col 1 ([1,1]) is a duplicate of Col 0 ([1,1])
        ], dtype=np.int64)
        # If we test (0,1) which is 2. new_color is 1. bad_board[0,1]=1.
        # Board becomes:
        # 1 1
        # 1 1
        # Rule: num_new_color_in_col (for col 1, color 1) is 2. (n/2 is 1). This is > n/2.
        # So, it should return True.
        self.assertTrue(check_unique(board_2x2_forced, (0,1)))


if __name__ == '__main__':
    unittest.main()
