from ffbench.data_handler import DataHandler
from ffbench.core import Player, Team, League
from ffbench.draft import Draft
from ffbench.season import Season
from ffbench.llm import LLMManager

def main():
    # Define the season to simulate
    simulated_season = 2023

    # 1. Initialize the DataHandler
    data_handler = DataHandler(data_dir="./data")
    
    # 2. Create a list of teams, each with its own LLM manager
    # num_teams = 10
    # teams = [
    #     Team(
    #         team_id=i,
    #         manager_name=f"LLM_{i+1}",
    #         llm_manager=LLMManager(model_name=f"gpt-4-turbo-0125"),
    #     )
    #     for i in range(num_teams)
    # ]

    # 3. Fetch historical player data for the draft
    print("\n--- Testing read_player_game_stats_by_week for 2022, Week 1 ---")
    player_stats = data_handler.read_player_game_stats_by_week(2022, 1)  # Load data for the previous season, week 1
    print(player_stats.head())

    # Filter for draftable players
    draftable_players = player_stats[
        player_stats["Position"].isin(["QB", "RB", "WR", "TE"])
    ].head(200)

    # Test reading data from Byes.csv
    print("\n--- Testing read_byes for 2023 ---")
    byes_2023 = data_handler.read_byes(2023)
    print(byes_2023.head())

    # Test reading data from FantasyDefenseBySeason.csv
    print("\n--- Testing read_fantasy_defense_by_season for 2023 ---")
    defense_2023 = data_handler.read_fantasy_defense_by_season(2023)
    print(defense_2023.head())

    # Test reading data from FreeAgents.csv
    print("\n--- Testing read_free_agents ---")
    free_agents = data_handler.read_free_agents()
    print(free_agents.head())

    # Test reading data from Players.csv
    print("\n--- Testing read_players ---")
    players = data_handler.read_players()
    print(players.head())

    # Test reading data from Teams.csv
    print("\n--- Testing read_teams ---")
    teams = data_handler.read_teams()
    print(teams.head())

    # Test reading data from Timeframes.csv
    print("\n--- Testing read_timeframes ---")
    timeframes = data_handler.read_timeframes()
    print(timeframes.head())

    # Test filtering for Justin Jefferson's stats in Week 1 of the 2023 season
    print("\n--- Testing filtering for Justin Jefferson's stats in Week 1, 2023 ---")
    week1_stats_2023 = data_handler.read_player_game_stats_by_week(2023, 1)
    jefferson_stats = week1_stats_2023[week1_stats_2023["Name"] == "Justin Jefferson"]
    print(jefferson_stats)

    # Test extracting Justin Jefferson's yards from Week 1 of the 2023 season
    print("\n--- Testing extracting Justin Jefferson's yards in Week 1, 2023 ---")
    jefferson_yards = jefferson_stats["ReceivingYards"]
    print(jefferson_yards)

    # Test extracting Josh Allen's passing yards from Week 14 of the 2024 season in one line
    print("\n--- Testing extracting Josh Allen's passing yards in Week 14, 2024 (one line) ---")
    allen_passing_yards_2024 = data_handler.read_player_game_stats_by_week(2024, 14)[data_handler.read_player_game_stats_by_week(2024, 14)["Name"] == "Josh Allen"]["PassingYards"]
    print(allen_passing_yards_2024)

    # Test cases for natural language response generation
    print("\n--- Testing natural language response for Justin Jefferson's 2023 Receiving Yards ---")
    response_1 = data_handler.generate_natural_language_response(
        player_name="Justin Jefferson",
        stat_type="ReceivingYards",
        time_range={"start_year": 2023, "end_year": 2023}
    )
    print(response_1)

    print("\n--- Testing natural language response for Josh Allen's Passing Yards from 2022 to 2024 ---")
    response_2 = data_handler.generate_natural_language_response(
        player_name="Josh Allen",
        stat_type="PassingYards",
        time_range={"start_year": 2022, "end_year": 2024}
    )
    print(response_2)

    print("\n--- Testing natural language response for Justin Jefferson's Receiving Stats from 2022 to 2023 ---")
    response_3 = data_handler.generate_natural_language_response(
        player_name="Justin Jefferson",
        stat_type="ReceivingYards",
        time_range={"start_year": 2022, "end_year": 2023}
    )
    print(response_3)

    print("\n--- Testing natural language response for a non-existent stat type ---")
    response_4 = data_handler.generate_natural_language_response(
        player_name="Justin Jefferson",
        stat_type="NonExistentStat",
        time_range={"start_year": 2023, "end_year": 2023}
    )
    print(response_4)

    print("\n--- Testing natural language response with an invalid time range ---")
    response_5 = data_handler.generate_natural_language_response(
        player_name="Justin Jefferson",
        stat_type="ReceivingYards",
        time_range={"start_year": None, "end_year": 2023}
    )
    print(response_5)



if __name__ == "__main__":
    main()