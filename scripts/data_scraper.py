import requests
import csv
import re
import os

## Gets API Key from specified path
def read_api_key(path='../../apiKeys/sportsData.txt'):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    with open(abs_path, 'r') as f:
        return f.read().strip()

# HTTP GET request locations
API_KEY = read_api_key()
SEASONS = [2022, 2023, 2024]
WEEKS = list(range(1, 18))
ENDPOINTS = [
    'https://api.sportsdata.io/api/nfl/fantasy/json/Byes/{season}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/Players',
    'https://api.sportsdata.io/api/nfl/fantasy/json/FreeAgents',
    'https://api.sportsdata.io/api/nfl/fantasy/json/Rookies/{season}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/Standings/{season}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/Teams',
    'https://api.sportsdata.io/api/nfl/fantasy/json/Timeframes/all',
    'https://api.sportsdata.io/api/nfl/fantasy/json/FantasyDefenseByGame/{season}/{week}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/FantasyDefenseBySeason/{season}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/PlayerGameStatsByWeek/{season}/{week}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/PlayerSeasonStats/{season}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/FantasyDefenseProjectionsByGame/{season}/{week}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/FantasyDefenseProjectionsBySeason/{season}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/PlayerGameProjectionStatsByWeek/{season}/{week}',
    'https://api.sportsdata.io/api/nfl/fantasy/json/PlayerSeasonProjectionStats/{season}'
]
FIELDS = [
    # Add the fields you want in your CSV, e.g. 'PlayerID', 'Name', 'Team', ...
]

def fetch_json(url):
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Loops through endpoints for seasons and weeks
def expand_endpoints():
    expanded = []
    for endpoint in ENDPOINTS:
        if '{season}' in endpoint and '{week}' in endpoint:
            for season in SEASONS:
                for week in WEEKS:
                    expanded.append(endpoint.format(season=season, week=week))
        elif '{season}' in endpoint:
            for season in SEASONS:
                expanded.append(endpoint.format(season=season))
        elif '{week}' in endpoint:
            for week in WEEKS:
                expanded.append(endpoint.format(week=week))
        else:
            expanded.append(endpoint)
    return expanded

def get_csv_path(url):
    match = re.search(r'/fantasy/json/([^/]+)', url)
    endpoint = match.group(1) if match else 'data'
    season_match = re.search(r'(?:/|=)(20\d{2})(?:/|$)', url)
    week_match = re.search(r'(?:/|=)(\d{1,2})(?:/|$)', url)
    season = season_match.group(1) if season_match else None
    week = week_match.group(1) if week_match else None

    filename = f'{endpoint}.csv'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if season and week:
        dir_path = os.path.join(base_dir, str(season), str(week))
    elif season:
        dir_path = os.path.join(base_dir, str(season))
    else:
        dir_path = os.path.join(base_dir, 'other')
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)

def main():
    urls = expand_endpoints()
    for url in urls:
        print(f'Fetching: {url}')
        try:
            data = fetch_json(url)
            if isinstance(data, dict):
                data = [data]
            if not data:
                print(f'No data for {url}')
                continue
            # Auto-detect fields from the first item
            fields = list(data[0].keys()) if isinstance(data[0], dict) else []
            csv_path = get_csv_path(url)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
            print(f'Data written to {csv_path}')
        except Exception as e:
            print(f'Error fetching {url}: {e}')

if __name__ == '__main__':
    main()