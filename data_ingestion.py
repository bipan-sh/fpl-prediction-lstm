import os
import pandas as pd
import requests
from io import StringIO

# Base URL for raw files from GitHub
raw_base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/"

def load_csv_from_url(relative_path):
    """
    Given a relative path, construct the raw URL, download the CSV,
    and load it into a DataFrame using UTF-8 encoding.
    """
    url = raw_base_url + relative_path
    try:
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = 'utf-8'
        df = pd.read_csv(StringIO(response.text), encoding='utf-8')
        print("Loaded '{}' with shape {}".format(relative_path, df.shape))
        return df
    except Exception as e:
        print("Error loading '{}': {}".format(relative_path, e))
        return None

def save_df_to_local(df, local_path):
    """Save DataFrame to a local CSV file, creating directories if necessary."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False, encoding='utf-8')
    print("Saved file to", local_path)

def ingest_data():
    """
    Downloads required data files from GitHub and saves them locally.
    This includes key files, Understat data, and all players' gameweek data.
    """
    base_local_dir = "data"
    os.makedirs(base_local_dir, exist_ok=True)

    # --- Download key files from the root of the data directory ---
    key_files = ["teams.csv", "fixtures.csv", "player_idlist.csv", "players_raw.csv"]
    for file_name in key_files:
        df = load_csv_from_url(file_name)
        if df is not None:
            local_path = os.path.join(base_local_dir, file_name)
            save_df_to_local(df, local_path)

    # --- Ingest Understat files using the GitHub API ---
    understat_local_dir = os.path.join(base_local_dir, "understat")
    os.makedirs(understat_local_dir, exist_ok=True)
    api_url = "https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/2024-25/understat"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
    except Exception as e:
        print("Error accessing GitHub API for Understat files:", e)
        return
    files = response.json()
    for file in files:
        if file.get('type') == 'file' and file.get('name', '').endswith('.csv'):
            name = file.get('name')
            download_url = file.get('download_url')
            try:
                df = pd.read_csv(download_url, encoding='utf-8')
                local_file_path = os.path.join(understat_local_dir, name)
                save_df_to_local(df, local_file_path)
            except Exception as e:
                safe_name = name.encode('utf-8', 'replace').decode('utf-8')
                print("Error saving Understat file '{}': {}".format(safe_name, e))

    # --- Ingest all players' gameweek data ---
    players_local_dir = os.path.join(base_local_dir, "players")
    os.makedirs(players_local_dir, exist_ok=True)
    # Load the local player_idlist file
    player_idlist_path = os.path.join(base_local_dir, "player_idlist.csv")
    if os.path.exists(player_idlist_path):
        player_idlist_df = pd.read_csv(player_idlist_path)
    else:
        print("Local player_idlist.csv not found.")
        return

    for idx, row in player_idlist_df.iterrows():
        # Construct folder name in the format "FirstName_SecondName_ID"
        folder_name = f"{row['first_name']}_{row['second_name']}_{int(row['id'])}"
        relative_path = f"players/{folder_name}/gw.csv"
        df = load_csv_from_url(relative_path)
        if df is not None:
            # Save the file preserving folder structure: data/players/<folder_name>/gw.csv
            folder_path = os.path.join(players_local_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            local_file_path = os.path.join(folder_path, "gw.csv")
            df['player_id'] = row['id']
            # If a gameweek column is missing, add one (assumes row order reflects gameweeks)
            if 'gameweek' not in df.columns:
                df = df.reset_index().rename(columns={'index': 'gameweek'})
                df['gameweek'] = df['gameweek'] + 1
            save_df_to_local(df, local_file_path)

    print("\nData ingestion complete. All files are saved in the 'data' directory.")

if __name__ == "__main__":
    ingest_data()
