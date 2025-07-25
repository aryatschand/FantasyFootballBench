import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

def get_pro_football_reference_player_stats(year: int, week: int) -> pd.DataFrame:
    """
    Scrapes weekly player stats for a given year and week from Pro-Football-Reference.
    """
    url = f"https://www.pro-football-reference.com/years/{year}/week_{week}.htm"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Could not retrieve data for week {week}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # The stats are in a commented-out section, so we need to find the comments
    comments = soup.find_all(string=lambda text: isinstance(text, str) and 'passing' in text)
    
    all_dfs = []
    
    for comment in comments:
        comment_soup = BeautifulSoup(comment, 'html.parser')
        
        # Passing stats
        passing_table = comment_soup.find('table', {'id': 'passing'})
        if passing_table:
            df = pd.read_html(str(passing_table))[0]
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            df.rename(columns={'Unnamed: 0_level_0_Player': 'Player'}, inplace=True)
            df['Week'] = week
            df['Year'] = year
            all_dfs.append(df)
            
        # Rushing and Receiving stats
        rushing_and_receiving_table = comment_soup.find('table', {'id': 'rushing_and_receiving'})
        if rushing_and_receiving_table:
            df = pd.read_html(str(rushing_and_receiving_table))[0]
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            df.rename(columns={'Unnamed: 0_level_0_Player': 'Player'}, inplace=True)
            df['Week'] = week
            df['Year'] = year
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    # Merge the dataframes
    merged_df = all_dfs[0]
    if len(all_dfs) > 1:
        for df in all_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on=['Player', 'Week', 'Year'], how='outer')

    return merged_df


def main():
    year = 2023
    all_weekly_data = []

    for week in range(1, 19):
        print(f"Scraping data for {year}, Week {week}...")
        weekly_stats = get_pro_football_reference_player_stats(year, week)
        
        if not weekly_stats.empty:
            all_weekly_data.append(weekly_stats)
        
        time.sleep(2)  # Be a good citizen

    if not all_weekly_data:
        print("No data was scraped. Exiting.")
        return

    # Combine all weekly data
    combined_df = pd.concat(all_weekly_data, ignore_index=True)

    # Clean up player names (remove symbols like '*' and '+')
    combined_df['Player'] = combined_df['Player'].str.replace(r'[\\*+]', '', regex=True)

    # Save to CSV
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"player_stats_{year}.csv")
    combined_df.to_csv(file_path, index=False)
    
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    main() 