import os
import ast
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler

def process_data() -> Dict[str, Any]:
    """
    Loads locally saved CSV files from the data directory,
    aggregates and processes them, normalizes feature columns,
    and creates sequences for LSTM.
    Returns a dictionary containing key DataFrames and the LSTM input (X, y).
    """
    base_local_dir = "data"
    teams_path = os.path.join(base_local_dir, "teams.csv")
    fixtures_path = os.path.join(base_local_dir, "fixtures.csv")
    player_idlist_path = os.path.join(base_local_dir, "player_idlist.csv")
    playerraw_path = os.path.join(base_local_dir, "playerraw.csv")
    players_local_dir = os.path.join(base_local_dir, "players")

    # Load key files
    teams_df = pd.read_csv(teams_path) if os.path.exists(teams_path) else None
    fixtures_df = pd.read_csv(fixtures_path) if os.path.exists(fixtures_path) else None
    player_idlist_df = pd.read_csv(player_idlist_path) if os.path.exists(player_idlist_path) else None
    playerraw_df = pd.read_csv(playerraw_path) if os.path.exists(playerraw_path) else None

    # Aggregate player gameweek data from the players folder
    player_gw_dfs = []
    if os.path.exists(players_local_dir):
        for folder in os.listdir(players_local_dir):
            folder_path = os.path.join(players_local_dir, folder)
            if os.path.isdir(folder_path):
                gw_file = os.path.join(folder_path, "gw.csv")
                if os.path.exists(gw_file):
                    df = pd.read_csv(gw_file)
                    player_gw_dfs.append(df)
        if player_gw_dfs:
            player_gw_df = pd.concat(player_gw_dfs, ignore_index=True)
            print("Aggregated player gameweek data shape:", player_gw_df.shape)
            # Print columns for debugging
            print("Player GW Data columns:", player_gw_df.columns.tolist())
        else:
            player_gw_df = None
    else:
        player_gw_df = None

    # Parse fixture stats (convert nested 'stats' string into Python objects)
    if fixtures_df is not None and 'stats' in fixtures_df.columns:
        def parse_stats(stats_str: str) -> list:
            if pd.isna(stats_str) or stats_str.strip() == "":
                return []
            try:
                return ast.literal_eval(stats_str)
            except Exception as e:
                print("Error parsing stats:", e)
                return []
        fixtures_df['parsed_stats'] = fixtures_df['stats'].apply(parse_stats)

        def extract_home_goals(parsed_stats: list) -> int:
            for stat in parsed_stats:
                if stat.get('identifier') == 'goals_scored':
                    home_stats = stat.get('h', [])
                    return sum(item.get('value', 0) for item in home_stats)
            return 0

        fixtures_df['home_goals_scored'] = fixtures_df['parsed_stats'].apply(extract_home_goals)
        print("\n--- Fixtures with Home Goals Scored ---")
        print(fixtures_df[['id', 'home_goals_scored']].head())

    # Define the required columns for sequence creation
    required_cols = ['player_id', 'gameweek', 'minutes', 'goals', 'assists', 'points']
    
    # Check and rename columns if necessary in player_gw_df
    if player_gw_df is not None:
        # Debug print: list current columns
        print("Initial columns in player_gw_df:", player_gw_df.columns.tolist())
        
        # Rename columns if needed:
        if 'total_points' in player_gw_df.columns and 'points' not in player_gw_df.columns:
            player_gw_df = player_gw_df.rename(columns={'total_points': 'points'})
            print("Renamed 'total_points' to 'points'.")
        if 'mins' in player_gw_df.columns and 'minutes' not in player_gw_df.columns:
            player_gw_df = player_gw_df.rename(columns={'mins': 'minutes'})
            print("Renamed 'mins' to 'minutes'.")
        if 'goals_scored' in player_gw_df.columns and 'goals' not in player_gw_df.columns:
            player_gw_df = player_gw_df.rename(columns={'goals_scored': 'goals'})
            print("Renamed 'goals_scored' to 'goals'.")
        
        # Print columns after renaming
        print("Columns after renaming:", player_gw_df.columns.tolist())
        
        # Check if all required columns are present
        if all(col in player_gw_df.columns for col in required_cols):
            # Normalize the feature columns used for sequence creation
            feature_cols = ['minutes', 'goals', 'assists']
            scaler = StandardScaler()
            player_gw_df[feature_cols] = scaler.fit_transform(player_gw_df[feature_cols])
            print("Normalized feature columns:", feature_cols)
            
            X, y = create_sequences(player_gw_df, seq_length=5,
                                    feature_cols=feature_cols,
                                    target_col='points')
            if X is not None:
                print("\nLSTM Input Sequences Shape:", X.shape)
                print("LSTM Target Shape:", y.shape)
            else:
                print("Sequence creation failed due to insufficient data for some players.")
        else:
            missing = [col for col in required_cols if col not in player_gw_df.columns]
            print("player_gw_df is missing required columns for sequence creation:", missing)
            X, y = None, None
    else:
        print("No player gameweek data available.")
        X, y = None, None

    print("\nData processing complete.")
    return {
        "teams_df": teams_df,
        "fixtures_df": fixtures_df,
        "player_idlist_df": player_idlist_df,
        "playerraw_df": playerraw_df,
        "player_gw_df": player_gw_df,
        "X": X,
        "y": y
    }

def create_sequences(df: pd.DataFrame, seq_length: int = 5, 
                     feature_cols: list = ['minutes', 'goals', 'assists'], 
                     target_col: str = 'points') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Creates sliding-window sequences from player gameweek data.
    Each sequence (shape [seq_length, num_features]) is paired with the target value from the next gameweek.
    """
    sequences = []
    targets = []
    if df is None:
        return None, None
    if 'gameweek' not in df.columns:
        print("Column 'gameweek' not found in player gameweek data. Cannot create sequences.")
        return None, None
    df = df.sort_values(['player_id', 'gameweek'])
    for player in df['player_id'].unique():
        player_data = df[df['player_id'] == player].reset_index(drop=True)
        if len(player_data) <= seq_length:
            continue
        data_array = player_data[feature_cols].values
        target_array = player_data[target_col].values
        for i in range(len(player_data) - seq_length):
            sequences.append(data_array[i:i+seq_length])
            targets.append(target_array[i+seq_length])
    return np.array(sequences), np.array(targets)

if __name__ == "__main__":
    process_data()
