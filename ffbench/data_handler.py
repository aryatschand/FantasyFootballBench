import pandas as pd
import os
from datetime import datetime

class DataHandler:
    def __init__(self, data_dir="../data"):
        """
        Initializes the DataHandler to work with spreadsheets in the /data folder.
        """
        # Get the directory containing this file, then go up one level and to data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up from ffbench to project root
        self.data_dir = os.path.join(project_root, "data")
        self._season_cache = {}  # Cache for season stats
        self._weekly_cache = {}  # Cache for weekly stats

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

    # Core data reading functions (keep only what's needed)
    def read_player_season_stats(self, year):
        file_path = os.path.join(self.data_dir, str(year), "PlayerSeasonStats.csv")
        return self.read_spreadsheet(file_path)

    def read_player_season_projection_stats(self, year):
        file_path = os.path.join(self.data_dir, str(year), "PlayerSeasonProjectionStats.csv")
        return self.read_spreadsheet(file_path)

    def read_player_game_stats_by_week(self, year, week):
        file_path = os.path.join(self.data_dir, str(year), str(week), "PlayerGameStatsByWeek.csv")
        return self.read_spreadsheet(file_path)

    def read_player_game_projection_stats_by_week(self, year, week):
        file_path = os.path.join(self.data_dir, str(year), str(week), "PlayerGameProjectionStatsByWeek.csv")
        return self.read_spreadsheet(file_path)

    def get_player_season_stats(self, player_name, year, current_year=None, current_week=None):
        """
        Get comprehensive season stats for a player in a specific year, formatted as natural language.

        Args:
            player_name (str): Name of the player
            year (int): Year to get stats for
            current_year (int, optional): Current year for data validation
            current_week (int, optional): Current week for data validation

        Returns:
            str or None: Natural language summary of player stats or None if data doesn't exist
        """
        if current_year is None:
            current_year = datetime.now().year
        if current_week is None:
            current_week = 1  # Default to week 1 if not specified

        # Don't return data for future years
        if year > current_year:
            return f"No data available for {player_name} in {year} (future season)."

        # Check if data is cached
        cache_key = f"season_{year}"
        if cache_key not in self._season_cache:
            season_stats = self.read_player_season_stats(year)
            self._season_cache[cache_key] = season_stats
        else:
            season_stats = self._season_cache[cache_key]

        if season_stats.empty:
            return f"No season data available for {year}."

        # Find the player
        player_data = season_stats[season_stats["Name"] == player_name]
        if player_data.empty:
            return f"No data found for {player_name} in {year}."

        # Get player stats
        stats = player_data.iloc[0]

        # Build natural language summary
        position = stats.get("Position", "Unknown")
        team = stats.get("Team", "Unknown")

        summary = f"📊 {player_name} ({position}, {team}) - {year} Season Performance:\n"

        # Games played
        games = stats.get("Played", 0)
        summary += f"• Games Played: {games}\n"

        if position == "QB":
            # QB specific stats
            pass_yds = stats.get("PassingYards", 0)
            pass_tds = stats.get("PassingTouchdowns", 0)
            ints = stats.get("PassingInterceptions", 0)
            rush_yds = stats.get("RushingYards", 0)
            rush_tds = stats.get("RushingTouchdowns", 0)

            summary += f"• Passing: {pass_yds} yards, {pass_tds} TDs, {ints} INTs\n"
            summary += f"• Rushing: {rush_yds} yards, {rush_tds} TDs\n"
            summary += f"• Fantasy Points: {stats.get('FantasyPointsPPR', 0):.1f} (PPR)\n"

        elif position in ["RB", "WR", "TE"]:
            # Skill position stats
            rush_yds = stats.get("RushingYards", 0)
            rush_tds = stats.get("RushingTouchdowns", 0)
            rec = stats.get("Receptions", 0)
            rec_yds = stats.get("ReceivingYards", 0)
            rec_tds = stats.get("ReceivingTouchdowns", 0)

            if position == "RB":
                summary += f"• Rushing: {rush_yds} yards, {rush_tds} TDs\n"
            summary += f"• Receiving: {rec} catches for {rec_yds} yards, {rec_tds} TDs\n"

            total_tds = rush_tds + rec_tds
            summary += f"• Total TDs: {total_tds}\n"
            summary += f"• Fantasy Points: {stats.get('FantasyPointsPPR', 0):.1f} (PPR)\n"

        elif position == "K":
            # Kicker stats
            fg_made = stats.get("FieldGoalsMade", 0)
            fg_att = stats.get("FieldGoalsAttempted", 0)
            pat_made = stats.get("ExtraPointsMade", 0)

            summary += f"• Field Goals: {fg_made}/{fg_att} made\n"
            summary += f"• Extra Points: {pat_made} made\n"
            summary += f"• Fantasy Points: {stats.get('FantasyPointsPPR', 0):.1f} (PPR)\n"

        # Add injury/games missed info if relevant
        if games < 17:  # Assuming 17 game season
            missed_games = 17 - games
            summary += f"• Missed Games: {missed_games}\n"

        return summary

    def get_player_weekly_stats(self, player_name, week, year, current_year=None, current_week=None):
        """
        Get game stats for a player in a specific week and year, formatted as natural language.

        Args:
            player_name (str): Name of the player
            week (int): Week number
            year (int): Year
            current_year (int, optional): Current year for data validation
            current_week (int, optional): Current week for data validation

        Returns:
            str or None: Natural language summary of player weekly stats or None if data doesn't exist
        """
        if current_year is None:
            current_year = datetime.now().year
        if current_week is None:
            current_week = 1

        # Don't return data for future games
        if year > current_year or (year == current_year and week > current_week):
            return f"No data available for {player_name} in Week {week}, {year} (future game)."

        # Check if data is cached
        cache_key = f"weekly_{year}_{week}"
        if cache_key not in self._weekly_cache:
            weekly_stats = self.read_player_game_stats_by_week(year, week)
            self._weekly_cache[cache_key] = weekly_stats
        else:
            weekly_stats = self._weekly_cache[cache_key]

        if weekly_stats.empty:
            return f"No game data available for Week {week}, {year}."

        # Find the player
        player_data = weekly_stats[weekly_stats["Name"] == player_name]
        if player_data.empty:
            return f"No data found for {player_name} in Week {week}, {year}."

        # Get player stats
        stats = player_data.iloc[0]

        # Build natural language summary
        position = stats.get("Position", "Unknown")
        team = stats.get("Team", "Unknown")
        opponent = stats.get("Opponent", "Unknown")
        home_away = "vs" if stats.get("HomeOrAway") == "HOME" else "@"

        summary = f"🏈 {player_name} ({position}, {team}) - Week {week}, {year} {home_away} {opponent}:\n"

        if position == "QB":
            # QB specific stats
            pass_yds = stats.get("PassingYards", 0)
            pass_tds = stats.get("PassingTouchdowns", 0)
            ints = stats.get("PassingInterceptions", 0)
            rush_yds = stats.get("RushingYards", 0)
            rush_tds = stats.get("RushingTouchdowns", 0)

            if pass_yds > 0:
                summary += f"• Passing: {pass_yds} yards, {pass_tds} TDs, {ints} INTs\n"
            if rush_yds > 0:
                summary += f"• Rushing: {rush_yds} yards, {rush_tds} TDs\n"

        elif position in ["RB", "WR", "TE"]:
            # Skill position stats
            rush_yds = stats.get("RushingYards", 0)
            rush_tds = stats.get("RushingTouchdowns", 0)
            rec = stats.get("Receptions", 0)
            rec_yds = stats.get("ReceivingYards", 0)
            rec_tds = stats.get("ReceivingTouchdowns", 0)

            if position == "RB" and rush_yds > 0:
                summary += f"• Rushing: {rush_yds} yards, {rush_tds} TDs\n"
            if rec > 0:
                summary += f"• Receiving: {rec} catches for {rec_yds} yards, {rec_tds} TDs\n"

            total_tds = rush_tds + rec_tds
            if total_tds > 0:
                summary += f"• Total TDs: {total_tds}\n"

        elif position == "K":
            # Kicker stats
            fg_made = stats.get("FieldGoalsMade", 0)
            fg_att = stats.get("FieldGoalsAttempted", 0)
            pat_made = stats.get("ExtraPointsMade", 0)

            if fg_att > 0:
                summary += f"• Field Goals: {fg_made}/{fg_att} made\n"
            if pat_made > 0:
                summary += f"• Extra Points: {pat_made} made\n"

        # Fantasy points
        fantasy_pts = stats.get("FantasyPointsPPR", 0)
        if fantasy_pts > 0:
            summary += f"• Fantasy Points: {fantasy_pts:.1f} (PPR)\n"

        # Check if they played
        played = stats.get("Played", 0)
        if played == 0:
            summary += "• Did not play (DNP)\n"

        return summary

    def get_player_projection_stats(self, player_name, week, year, current_year=None, current_week=None):
        """
        Get projection stats for a player in a specific week and year, formatted as natural language.

        Args:
            player_name (str): Name of the player
            week (int): Week number
            year (int): Year
            current_year (int, optional): Current year for data validation
            current_week (int, optional): Current week for data validation

        Returns:
            str or None: Natural language summary of player projections or None if data doesn't exist
        """
        if current_year is None:
            current_year = datetime.now().year
        if current_week is None:
            current_week = 1

        # Projections are available for future games, so no time validation needed

        # Check cache
        cache_key = f"projection_{year}_{week}"
        if cache_key not in self._weekly_cache:
            projection_stats = self.read_player_game_projection_stats_by_week(year, week)
            self._weekly_cache[cache_key] = projection_stats
        else:
            projection_stats = self._weekly_cache[cache_key]

        if projection_stats.empty:
            return f"No projection data available for Week {week}, {year}."

        # Find the player
        player_data = projection_stats[projection_stats["Name"] == player_name]
        if player_data.empty:
            return f"No projections found for {player_name} in Week {week}, {year}."

        # Get player projections
        projections = player_data.iloc[0]

        # Build natural language summary
        position = projections.get("Position", "Unknown")
        team = projections.get("Team", "Unknown")

        summary = f"🔮 {player_name} ({position}, {team}) - Week {week}, {year} Projections:\n"

        if position == "QB":
            # QB projections
            pass_yds = projections.get("PassingYards", 0)
            pass_tds = projections.get("PassingTouchdowns", 0)
            ints = projections.get("PassingInterceptions", 0)
            rush_yds = projections.get("RushingYards", 0)

            if pass_yds > 0:
                summary += f"• Passing: {pass_yds:.1f} yards, {pass_tds:.1f} TDs, {ints:.1f} INTs\n"
            if rush_yds > 0:
                summary += f"• Rushing: {rush_yds:.1f} yards\n"

        elif position in ["RB", "WR", "TE"]:
            # Skill position projections
            rush_yds = projections.get("RushingYards", 0)
            rec = projections.get("Receptions", 0)
            rec_yds = projections.get("ReceivingYards", 0)

            if position == "RB" and rush_yds > 0:
                summary += f"• Rushing: {rush_yds:.1f} yards\n"
            if rec > 0:
                summary += f"• Receiving: {rec:.1f} catches for {rec_yds:.1f} yards\n"

        # Fantasy points projection
        fantasy_pts = projections.get("FantasyPointsPPR", 0)
        if fantasy_pts > 0:
            summary += f"• Projected Fantasy Points: {fantasy_pts:.1f} (PPR)\n"

        return summary

    def get_all_players_season_stats(self, year, position=None, current_year=None, top_n=10):
        """
        Get season stats for top players in a year, optionally filtered by position, formatted as natural language.

        Args:
            year (int): Year to get stats for
            position (str, optional): Position filter (QB, RB, WR, TE)
            current_year (int, optional): Current year for validation
            top_n (int, optional): Number of top players to return

        Returns:
            str: Natural language summary of top players
        """
        if current_year is None:
            current_year = datetime.now().year

        if year > current_year:
            return f"No data available for {year} (future season)."

        cache_key = f"season_{year}"
        if cache_key not in self._season_cache:
            season_stats = self.read_player_season_stats(year)
            self._season_cache[cache_key] = season_stats
        else:
            season_stats = self._season_cache[cache_key]

        if season_stats.empty:
            return f"No season data available for {year}."

        # Filter by position if specified
        if position:
            season_stats = season_stats[season_stats["Position"] == position]
            position_text = f"{position}s"
        else:
            position_text = "Players"

        # Sort by fantasy points and get top N
        season_stats = season_stats.sort_values("FantasyPointsPPR", ascending=False).head(top_n)

        if season_stats.empty:
            return f"No {position_text.lower()} data found for {year}."

        # Build natural language summary
        summary = f"🏆 Top {len(season_stats)} {position_text} by Fantasy Points - {year} Season:\n\n"

        for i, (_, player) in enumerate(season_stats.iterrows(), 1):
            name = player.get("Name", "Unknown")
            team = player.get("Team", "Unknown")
            position_abbr = player.get("Position", "Unknown")
            fantasy_pts = player.get("FantasyPointsPPR", 0)

            summary += f"{i}. {name} ({position_abbr}, {team}) - {fantasy_pts:.1f} pts\n"

            # Add key stats based on position
            if position_abbr == "QB":
                pass_yds = player.get("PassingYards", 0)
                pass_tds = player.get("PassingTouchdowns", 0)
                rush_yds = player.get("RushingYards", 0)
                summary += f"   • {pass_yds} pass yds, {pass_tds} pass TDs, {rush_yds} rush yds\n"

            elif position_abbr in ["RB", "WR", "TE"]:
                rush_yds = player.get("RushingYards", 0)
                rush_tds = player.get("RushingTouchdowns", 0)
                rec = player.get("Receptions", 0)
                rec_yds = player.get("ReceivingYards", 0)
                rec_tds = player.get("ReceivingTouchdowns", 0)

                if position_abbr == "RB" and rush_yds > 0:
                    summary += f"   • {rush_yds} rush yds, {rush_tds} rush TDs\n"
                if rec > 0:
                    summary += f"   • {rec} rec, {rec_yds} rec yds, {rec_tds} rec TDs\n"

            elif position_abbr == "K":
                fg_made = player.get("FieldGoalsMade", 0)
                pat_made = player.get("ExtraPointsMade", 0)
                summary += f"   • {fg_made} FGs, {pat_made} PATs\n"

            summary += "\n"

        return summary

    def clear_cache(self):
        """Clear all cached data."""
        self._season_cache.clear()
        self._weekly_cache.clear()

    # Backward compatibility aliases
    def get_player_stats_by_year(self, player_name, year, current_year=None, current_week=None):
        """Alias for get_player_season_stats for backward compatibility."""
        return self.get_player_season_stats(player_name, year, current_year, current_week)

    def get_player_stats_by_week(self, player_name, week, year, current_year=None, current_week=None):
        """Alias for get_player_weekly_stats for backward compatibility."""
        return self.get_player_weekly_stats(player_name, week, year, current_year, current_week)