import unittest
import pandas as pd
from pandas.api.types import is_numeric_dtype
from src.data.loaders import load_fbref, load_statsbomb, load_nfl

class TestLoaders(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Runs ONCE before all tests. 
        We load the data here to avoid hitting the APIs multiple times.
        """
        print("Fetching data for tests (this might take a moment)...")
        cls.df_fbref = load_fbref("La Liga", "2023-2024")
        # Initialize the others so we can test them
        # Note: You need valid IDs for StatsBomb, e.g., Men's World Cup 2022 (comp 43, season 106)
        cls.df_sb = load_statsbomb(43, 106) 
        cls.df_nfl = load_nfl([2023], "pbp")

    # --- FBREF TESTS ---
    def test_fbref_valid_data(self):
        """Test the 'Happy Path' for FBref: proper structure and data types."""
        self.assertIsNotNone(self.df_fbref, "FBref DataFrame returned None.")
        self.assertFalse(self.df_fbref.empty, "FBref DataFrame is empty.")
        
        # 1. Extensive Schema Check
        expected_cols = ["Squad", "Age", "Poss", "Performance_Gls", "Performance_Ast"]
        for col in expected_cols:
            self.assertIn(col, self.df_fbref.columns, f"FBref missing column: {col}")
            
        # 2. Data Type Validation
        self.assertTrue(is_numeric_dtype(self.df_fbref["Age"]), "FBref 'Age' should be numeric.")
        
    def test_fbref_invalid_input(self):
        """Test the 'Sad Path': Graceful failure on bad user input."""
        df_invalid = load_fbref("Fake League Name", "2023-2024")
        self.assertIsNone(df_invalid, "FBref should return None for unsupported leagues.")

    # --- STATSBOMB TESTS ---
    def test_statsbomb_valid_data(self):
        """Test the structural integrity of StatsBomb match data."""
        self.assertIsNotNone(self.df_sb, "StatsBomb DataFrame returned None.")
        self.assertFalse(self.df_sb.empty, "StatsBomb DataFrame is empty.")
        
        expected_cols = ["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]
        for col in expected_cols:
            self.assertIn(col, self.df_sb.columns, f"StatsBomb missing column: {col}")

    # --- NFL TESTS ---
    def test_nfl_valid_data(self):
        """Test the structural integrity of NFL play-by-play data."""
        self.assertIsNotNone(self.df_nfl, "NFL DataFrame returned None.")
        self.assertFalse(self.df_nfl.empty, "NFL DataFrame is empty.")
        
        # We will add the expected columns here based on your domain knowledge
        expected_cols = [] 
        for col in expected_cols:
            self.assertIn(col, self.df_nfl.columns, f"NFL missing column: {col}")
            
    def test_nfl_invalid_data_type(self):
        """Test that the NFL loader rejects bad data_type strings."""
        df_invalid = load_nfl([2023], "fake_data_type")
        self.assertIsNone(df_invalid, "NFL should return None for unsupported data types.")

if __name__ == '__main__':
    unittest.main()