import os
import requests
import pandas as pd

# --- Configuration ---
API_KEY = os.environ.get("SPORTSDATA_IO_API_KEY", "YOUR_API_KEY") 
BASE_URL = "https://api.sportsdata.io/v3/nfl/stats/json"
YEAR = 2023

def get_player_season_stats(year, api_key):
    """
    Fetches season stats for all players for a given year.
    """
    url = f"{BASE_URL}/PlayerSeasonStats/{year}"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def main():
    """
    Main function to fetch and save player stats.
    """
    if API_KEY == "YOUR_API_KEY":
        print("Please set your SportsData.io API key as an environment variable (SPORTSDATA_IO_API_KEY) or directly in the script.")
        return

    print(f"Fetching player stats for the {YEAR} season...")
    player_stats = get_player_season_stats(YEAR, API_KEY)

    if player_stats:
        df = pd.DataFrame(player_stats)
        
        # --- Data Cleaning and Preparation ---
        # Select and rename columns for clarity
        columns_to_keep = {
            'PlayerID': 'player_id',
            'Name': 'player_name',
            'Team': 'team',
            'Position': 'position',
            'Played': 'games_played',
            'PassingYards': 'passing_yards',
            'PassingTouchdowns': 'passing_tds',
            'PassingInterceptions': 'passing_interceptions',
            'RushingYards': 'rushing_yards',
            'RushingTouchdowns': 'rushing_tds',
            'Receptions': 'receptions',
            'ReceivingYards': 'receiving_yards',
            'ReceivingTouchdowns': 'receiving_tds',
            'FumblesLost': 'fumbles_lost'
        }
        df = df[columns_to_keep.keys()].rename(columns=columns_to_keep)

        # Create data directory if it doesn't exist
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        file_path = os.path.join(output_dir, f"player_stats_{YEAR}.csv")
        df.to_csv(file_path, index=False)
        
        print(f"Player stats for the {YEAR} season saved to {file_path}")

if __name__ == "__main__":
    main() 