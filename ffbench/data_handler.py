import nfl_data_py as nfl
import pandas as pd
import os

class DataHandler:
    def __init__(self, season, data_dir="ffbench/data_cache"):
        """
        Initializes the DataHandler and pre-fetches all necessary data.
        """
        self.season = season
        self.data_dir = data_dir
        self.historical_data_path = os.path.join(self.data_dir, f"historical_data_{season-1}.parquet")
        self.weekly_data_path_template = os.path.join(self.data_dir, f"weekly_data_{season}_week_{{week}}.parquet")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self._cache_historical_data()
        self._cache_weekly_data()


    def _cache_historical_data(self):
        """
        Caches historical player data up to the simulated season.
        """
        if not os.path.exists(self.historical_data_path):
            print(f"Caching historical data up to {self.season -1}...")
            historical_seasons = range(2018, self.season) # 5 years of history
            df = nfl.import_seasonal_data(list(historical_seasons))
            df.to_parquet(self.historical_data_path)
    
    def _cache_weekly_data(self):
        """
        Caches all weekly data for the simulated season.
        """
        print(f"Caching weekly data for the {self.season} season...")
        for week in range(1, 18): # Cache all 17 weeks of the regular season
            weekly_data_path = self.weekly_data_path_template.format(week=week)
            if not os.path.exists(weekly_data_path):
                df = nfl.import_weekly_data([self.season], [week])
                if not df.empty:
                    df.to_parquet(weekly_data_path)


    def get_historical_player_stats(self):
        """
        Fetches historical player stats from the local cache.
        """
        return pd.read_parquet(self.historical_data_path)

    def get_weekly_player_stats(self, week):
        """
        Fetches weekly player stats for a given week from the local cache.
        """
        weekly_data_path = self.weekly_data_path_template.format(week=week)
        if os.path.exists(weekly_data_path):
            return pd.read_parquet(weekly_data_path)
        else:
            return pd.DataFrame() 