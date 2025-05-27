import unittest
import numpy as np
from generate_full_board import has_three_in_row, generate_valid_rows, solve # Assuming file is accessible

class TestHasThreeInRow(unittest.TestCase):
    def test_1d_has_three(self):
        self.assertTrue(has_three_in_row(np.array([0, 1, 1, 1, 0], dtype=np.int64)))
        self.assertTrue(has_three_in_row(np.array([2, 2, 2, 0, 1], dtype=np.int64)))
        self.assertTrue(has_three_in_row(np.array([1, 1, 1], dtype=np.int64)))

    def test_1d_no_three(self):
        self.assertFalse(has_three_in_row(np.array([0, 1, 0, 1, 0], dtype=np.int64)))
        self.assertFalse(has_three_in_row(np.array([1, 1, 2, 2, 1], dtype=np.int64)))
        self.assertFalse(has_three_in_row(np.array([1, 2], dtype=np.int64))) # Shorter than 3
        self.assertFalse(has_three_in_row(np.array([], dtype=np.int64))) # Empty

    def test_1d_with_zeros(self):
        # Current has_three_in_row checks for non-zero identicals
        self.assertFalse(has_three_in_row(np.array([1, 0, 0, 0, 1], dtype=np.int64))) 
        self.assertFalse(has_three_in_row(np.array([0, 0, 0, 1, 1], dtype=np.int64)))

    def test_2d_has_three_in_col(self):
        # Test case for when solve calls it with grid_slice.T
        # Each row of this 2D array represents a column segment from the original grid
        # has_three_in_row should check each row of this input for three identicals
        board_slice_T_valid = np.array([
            [1, 0, 2], # col segment 1
            [0, 1, 0], # col segment 2
            [2, 2, 2], # col segment 3 has three 2s
            [0, 0, 1]  # col segment 4
        ], dtype=np.int64)
        self.assertTrue(has_three_in_row(board_slice_T_valid))

        board_slice_T_invalid = np.array([
            [1, 0, 2],
            [0, 1, 0],
            [2, 0, 2], 
            [0, 0, 1]
        ], dtype=np.int64)
        self.assertFalse(has_three_in_row(board_slice_T_invalid))
        
        # Edge case: 2D array but less than 3 columns (segments are too short)
        board_slice_T_short_cols = np.array([
            [1, 1],
            [2, 2]
        ], dtype=np.int64) # has_three_in_row handles this by checking arr_input.shape[1] != 3
        self.assertFalse(has_three_in_row(board_slice_T_short_cols))


class TestGenerateValidRows(unittest.TestCase):
    def test_generate_valid_rows_n4(self):
        n = 4
        valid_rows = generate_valid_rows(n) # Returns List[np.ndarray]
        
        # For n=4, half_n=2. Combinations of 2 from 4 is 6.
        # All 6 combinations are valid as no 3-in-a-row possible with two 1s.
        # e.g., [1,1,0,0], [1,0,1,0], [1,0,0,1], [0,1,1,0], [0,1,0,1], [0,0,1,1]
        self.assertEqual(len(valid_rows), 6)
        
        for row in valid_rows:
            self.assertIsInstance(row, np.ndarray)
            self.assertEqual(row.shape, (n,))
            # Rows are stored as 0s and 1s (np.int64). Summing the row gives count of 1s.
            self.assertEqual(np.sum(row), n / 2, f"Row {row} does not have n/2 ones.")
            # Check for three consecutive (has_three_in_row should have ensured this)
            # The rows are stored as int64 (0s and 1s)
            self.assertFalse(has_three_in_row(row), f"Row {row} has three consecutive identical numbers.")

    def test_generate_valid_rows_n6(self):
        n = 6
        valid_rows = generate_valid_rows(n) # Returns List[np.ndarray]
        
        # For n=6, half_n=3. Combinations of 3 from 6 is (6*5*4)/(3*2*1) = 20.
        # Some are invalid: e.g., [1,1,1,0,0,0]
        # Valid: [1,1,0,1,0,0], [1,1,0,0,1,0], [1,1,0,0,0,1], [1,0,1,1,0,0] etc.
        # Manually calculated known valid count for n=6 (where no three consecutive 1s):
        # Total C(6,3) = 20. Invalid are [1,1,1,0,0,0] and [0,0,0,1,1,1] (if considering 0s too)
        # or [T,T,T,F,F,F] and its permutations that keep 1s together.
        # [1,1,1,0,0,0] - invalid
        # [0,1,1,1,0,0] - invalid
        # [0,0,1,1,1,0] - invalid
        # [0,0,0,1,1,1] - invalid (if we were checking for 0s too, but we check for 1s)
        # The function generates rows with 1s and 0s (from boolean), so [1,1,1,0,0,0] is the pattern.
        # Permutations of indices for 1s:
        # (0,1,2) -> 111000 - no
        # (0,1,3) -> 110100 - yes
        # ...
        # Expected count is 20 - 4 = 16, if has_three_in_row correctly identifies [1,1,1,...] patterns.
        # Actually, it's C(n, n/2) - (n - (3-1)) * C(n-3, n/2 - 3) if simplified... this is complex.
        # Known correct values for Takuzu-like row generation (n/2 ones, no k consecutive):
        # For n=6, k=3 (three consecutive), n/2=3 ones: Result is 14.
        # (Total C(6,3)=20. Invalid: 111000, 011100, 001110, 000111 - wait, these are 4.
        #  No, it's indices (0,1,2), (1,2,3), (2,3,4), (3,4,5) for the start of a run of 3.
        #  (0,1,2) -> 111000
        #  (1,2,3) -> 011100
        #  (2,3,4) -> 001110
        #  (3,4,5) -> 000111
        #  These are the 4 invalid patterns. So 20 - 4 = 16.
        #  Let me recheck has_three_in_row. It checks for non-zero.
        #  So [1,1,1,0,0,0] is True from has_three_in_row.
        #  The rows are generated as boolean (True/False) and then converted to int for has_three_in_row.
        #  So, it checks for three consecutive Trues.
        #  Thus, 20 total combinations of 3 Trues. 4 of them have three Trues in a row.
        #  So, 16 valid rows.
        self.assertEqual(len(valid_rows), 16, "Number of valid rows for n=6 is not 16.")

        for row in valid_rows:
            self.assertIsInstance(row, np.ndarray)
            self.assertEqual(row.shape, (n,))
            self.assertEqual(np.sum(row), n / 2) # Sum of 1s
            self.assertFalse(has_three_in_row(row))

class TestSolve(unittest.TestCase):
    def test_solve_simple_4x4_last_row(self):
        n = 4
        # Almost complete grid, solve for the last row
        grid = np.array([
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 2, 1], # This row has three 2s if we consider 2s, but rules are about balance and 3-in-a-row of *either*
                          # The rules applied by solve are: no identical rows, no 3-in-a-row in cols, balanced cols, no identical cols.
                          # Let's use a valid start to avoid rule breakage before solve starts.
            [0, 0, 0, 0]  # Row to be solved
        ], dtype=np.int64)
        
        # A valid partial grid for testing:
        grid_test = np.array([
            [1,0,1,0], # using 0 and 1 for simplicity, as generate_valid_rows produces these
            [0,1,0,1],
            [1,1,0,0],
            [0,0,0,0] # empty row to fill
        ], dtype=np.int64)
        
        n_accomplished = 3
        
        # Provide only the correct last row as a valid_row option
        # Correct last row must balance cols and not be duplicate of existing rows or cols.
        # Col sums for first 3 rows:
        # C0: 1+0+1 = 2 (needs zero '1's)
        # C1: 0+1+1 = 2 (needs zero '1's)
        # C2: 1+0+0 = 1 (needs one '1')
        # C3: 0+1+0 = 1 (needs one '1')
        # So, last row should be [0,0,1,1] (two 1s, two 0s)
        # Check for 3-in-a-row with this row:
        # Grid becomes:
        # [1,0,1,0]
        # [0,1,0,1]
        # [1,1,0,0]
        # [0,0,1,1]
        # Col 0: [1,0,1,0] - OK
        # Col 1: [0,1,1,0] - OK
        # Col 2: [1,0,0,1] - OK
        # Col 3: [0,1,0,1] - OK
        # No identical rows. No identical columns.
        
        valid_row_option = np.array([0,0,1,1], dtype=np.int64)
        valid_rows_for_solve = [valid_row_option] # Python list of np.ndarray

        solution = solve(grid_test.copy(), valid_rows_for_solve, n_accomplished, n)
        
        self.assertEqual(solution.shape, (n,n), "Solution should be a 4x4 grid.")
        self.assertTrue(np.array_equal(solution[3,:], valid_row_option), "Last row not solved correctly.")

    def test_solve_no_solution_possible(self):
        n = 4
        grid = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [1,1,0,0],
            [0,0,0,0] 
        ], dtype=np.int64)
        n_accomplished = 3
        
        # Provide an invalid row that would violate rules
        invalid_row_option = np.array([1,0,1,0], dtype=np.int64) # Duplicate of first row
        valid_rows_for_solve = [invalid_row_option]

        solution = solve(grid.copy(), valid_rows_for_solve, n_accomplished, n)
        self.assertEqual(solution.shape, (0,0), "Should return empty array if no solution.")

if __name__ == '__main__':
    unittest.main()
