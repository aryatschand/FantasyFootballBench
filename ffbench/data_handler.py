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

    def generate_natural_language_response(self, player_name, stat_type, time_range):
        """
        Generate a natural language response for a player's stats.

        Args:
            player_name (str): Name of the player (e.g., "Justin Jefferson").
            stat_type (str): Type of stat (e.g., "ReceivingYards").
            time_range (dict): Time range for the stats (e.g., {"start_year": 2022, "end_year": 2024}).

        Returns:
            str: A natural language response summarizing the requested stats.
        """
        start_year = time_range.get("start_year")
        end_year = time_range.get("end_year")

        # Validate input
        if not start_year or not end_year:
            return "Invalid time range provided. Please specify both start and end years."

        # Fetch data for the specified time range
        stats = []
        for year in range(start_year, end_year + 1):
            for week in range(1, 18):  # Assuming 17 weeks in a season
                weekly_stats = self.read_player_game_stats_by_week(year, week)
                if not weekly_stats.empty:
                    player_stats = weekly_stats[weekly_stats["Name"] == player_name]
                    stats.append(player_stats)

        # Combine stats across weeks and years
        if stats:
            combined_stats = pd.concat(stats, ignore_index=True)
        else:
            return f"No data found for {player_name} from {start_year} to {end_year}."

        # Aggregate the requested stat
        if stat_type not in combined_stats.columns:
            return f"Stat type '{stat_type}' not found for {player_name}."

        total_stat = combined_stats[stat_type].sum()

        # Format the response
        response = (
            f"From {start_year} to {end_year}, {player_name} recorded a total of {total_stat} {stat_type}."
        )
        return response