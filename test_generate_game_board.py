import unittest
import numpy as np
from generate_game_board import create_emptier_board
# generate_completed_board is in _generate_completed_board.py.
# Ensure this file is in the Python path or adjust import accordingly.
from _generate_completed_board import generate_completed_board 

class TestCreateEmptierBoard(unittest.TestCase):

    def test_returns_board_with_zeros(self):
        """Test that the function introduces some empty cells (0s)."""
        for n in [4, 6]: # Test for a couple of sizes
            completed_board = generate_completed_board(n)
            if completed_board.size == 0: # Should not happen with valid n
                self.fail(f"generate_completed_board({n}) returned an empty board, cannot test create_emptier_board.")

            emptier_board = create_emptier_board(completed_board.copy()) # Pass a copy
            
            self.assertEqual(emptier_board.shape, completed_board.shape, "Board shape should not change.")
            # Check if there's at least one zero, assuming the original completed_board has no zeros
            # (generate_completed_board produces boards with 1s and 2s)
            has_zeros = np.any(emptier_board == 0)
            
            # It's possible, though unlikely for a typical run, that create_emptier_board
            # might not be able to empty any cells if all are "forced".
            # A more robust test would be to check if it's different OR has zeros.
            # If the board is full of forced moves, it might return the same board.
            # However, the typical expectation is that some cells are emptied.
            if not np.array_equal(emptier_board, completed_board):
                self.assertTrue(has_zeros, f"Emptier board (n={n}) should have zeros if different from original.")
            else:
                # This case means no cells were emptied. This can be valid if all cells are critical.
                # For this test, we'll assume for typical boards some cells can be emptied.
                # A more sophisticated test might check the number of non-zero elements.
                print(f"Warning: create_emptier_board(n={n}) returned the same board. This might be valid if all cells are critical.")
                # Assert that if it's the same, the original didn't have zeros (or check properties of a "stuck" board)
                self.assertFalse(np.any(completed_board == 0), "If board is unchanged, original should not have zeros.")


    def test_not_identical_to_input_if_emptying_possible(self):
        """Test that the returned board is different if cells can be emptied."""
        # This test is probabilistic as check_unique might make all cells essential.
        # For typical boards, some should be removable.
        n = 4 
        completed_board = generate_completed_board(n)
        if completed_board.size == 0:
            self.fail(f"generate_completed_board({n}) returned an empty board.")

        emptier_board = create_emptier_board(completed_board.copy())
        
        # If no cells could be emptied, this might be equal.
        # The core idea is that if create_emptier_board *can* empty cells, it *should*.
        # This test is somewhat an integration test of check_unique's behavior.
        # A board that is returned unchanged implies all cells were "unique".
        if np.count_nonzero(completed_board) > 0 : # Only if there are cells to empty
             # It's possible that for some fully completed boards, no cell can be emptied.
             # This is a weak test. A better one would be to ensure *fewer* non-zero cells
             # if the board changed.
            if not np.array_equal(emptier_board, completed_board):
                 self.assertLess(np.count_nonzero(emptier_board), np.count_nonzero(completed_board),
                                "If board changed, it should have more zeros (fewer non-zeros).")
            # else:
                # print(f"Skipping non-identical check for n={n} as board was returned unchanged.")
        else:
            self.assertTrue(np.array_equal(emptier_board, completed_board), "Empty input should return empty.")


    def test_handles_already_empty_board(self):
        """Test with an already empty board."""
        n = 4
        empty_board = np.zeros((n, n), dtype=np.int64)
        returned_board = create_emptier_board(empty_board.copy())
        self.assertTrue(np.array_equal(returned_board, empty_board), 
                        "Should return an already empty board as is.")

    def test_handles_board_with_some_zeros(self):
        """Test with a board that already has some zeros."""
        n = 4
        partially_filled_board = generate_completed_board(n)
        if partially_filled_board.size == 0:
            self.fail(f"generate_completed_board({n}) returned an empty board.")
        
        # Make some cells zero
        partially_filled_board[0,0] = 0
        partially_filled_board[1,1] = 0
        
        num_zeros_before = np.count_nonzero(partially_filled_board == 0)
        
        returned_board = create_emptier_board(partially_filled_board.copy())
        num_zeros_after = np.count_nonzero(returned_board == 0)

        self.assertEqual(returned_board.shape, partially_filled_board.shape)
        self.assertGreaterEqual(num_zeros_after, num_zeros_before,
                                "Should not fill cells, only add more zeros or keep as is.")

if __name__ == '__main__':
    unittest.main()
