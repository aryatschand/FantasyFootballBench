import pandas as pd
import os

class DataHandler:
    def __init__(self, data_dir="../data"):
        """
        Initializes the DataHandler to work with spreadsheets in the /data folder.
        """
        self.data_dir = os.path.abspath(data_dir)

    def read_spreadsheet(self, file_path):
        """
        Generic function to read data from a spreadsheet.
        """
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                print(f"Spreadsheet not found: {file_path}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading spreadsheet {file_path}: {e}")
            return pd.DataFrame()

    def read_byes(self, year):
        file_path = os.path.join(self.data_dir, str(year), "Byes.csv")
        return self.read_spreadsheet(file_path)

    def read_fantasy_defense_by_season(self, year):
        file_path = os.path.join(self.data_dir, str(year), "FantasyDefenseBySeason.csv")
        return self.read_spreadsheet(file_path)

    def read_fantasy_defense_projections_by_season(self, year):
        file_path = os.path.join(self.data_dir, str(year), "FantasyDefenseProjectionsBySeason.csv")
        return self.read_spreadsheet(file_path)

    def read_player_season_projection_stats(self, year):
        file_path = os.path.join(self.data_dir, str(year), "PlayerSeasonProjectionStats.csv")
        return self.read_spreadsheet(file_path)

    def read_rookies(self, year):
        file_path = os.path.join(self.data_dir, str(year), "Rookies.csv")
        return self.read_spreadsheet(file_path)

    def read_standings(self, year):
        file_path = os.path.join(self.data_dir, str(year), "Standings.csv")
        return self.read_spreadsheet(file_path)

    def read_fantasy_defense_by_game(self, year, week):
        file_path = os.path.join(self.data_dir, str(year), str(week), "FantasyDefenseByGame.csv")
        return self.read_spreadsheet(file_path)

    def read_fantasy_defense_projections_by_game(self, year, week):
        file_path = os.path.join(self.data_dir, str(year), str(week), "FantasyDefenseProjectionsByGame.csv")
        return self.read_spreadsheet(file_path)

    def read_player_game_projection_stats_by_week(self, year, week):
        file_path = os.path.join(self.data_dir, str(year), str(week), "PlayerGameProjectionStatsByWeek.csv")
        return self.read_spreadsheet(file_path)

    def read_player_game_stats_by_week(self, year, week):
        file_path = os.path.join(self.data_dir, str(year), str(week), "PlayerGameStatsByWeek.csv")
        return self.read_spreadsheet(file_path)

    def read_free_agents(self):
        file_path = os.path.join(self.data_dir, "other", "FreeAgents.csv")
        return self.read_spreadsheet(file_path)

    def read_players(self):
        file_path = os.path.join(self.data_dir, "other", "Players.csv")
        return self.read_spreadsheet(file_path)

    def read_teams(self):
        file_path = os.path.join(self.data_dir, "other", "Teams.csv")
        return self.read_spreadsheet(file_path)

    def read_timeframes(self):
        file_path = os.path.join(self.data_dir, "other", "Timeframes.csv")
        return self.read_spreadsheet(file_path)