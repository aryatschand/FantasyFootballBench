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

    # 3. Test the new natural language data access functions
    print("\n--- Testing new natural language data access ---")

    # Test player season stats
    print("\n--- Player Season Stats (Natural Language) ---")
    season_stats = data_handler.get_player_season_stats("Josh Allen", 2024, current_year=2024, current_week=10)
    print(season_stats)

    # Test player weekly stats
    print("\n--- Player Weekly Stats (Natural Language) ---")
    weekly_stats = data_handler.get_player_weekly_stats("Josh Allen", 1, 2024, current_year=2024, current_week=10)
    print(weekly_stats)

    # Test player projections
    print("\n--- Player Projections (Natural Language) ---")
    projections = data_handler.get_player_projection_stats("Josh Allen", 11, 2024, current_year=2024, current_week=10)
    print(projections)

    # Test top players
    print("\n--- Top QB Performers (Natural Language) ---")
    top_qbs = data_handler.get_all_players_season_stats(2024, position="QB", current_year=2024, top_n=5)
    print(top_qbs)

    # Test the new weekly stats function for Justin Jefferson
    print("\n--- Justin Jefferson Week 1, 2023 (Natural Language) ---")
    jj_weekly = data_handler.get_player_weekly_stats("Justin Jefferson", 1, 2023, current_year=2024, current_week=10)
    print(jj_weekly)

    # Test the new weekly stats function for Josh Allen
    print("\n--- Josh Allen Week 14, 2024 (Natural Language) ---")
    ja_weekly = data_handler.get_player_weekly_stats("Josh Allen", 14, 2024, current_year=2024, current_week=15)  # Week 15 is current
    print(ja_weekly)

    # Additional tests with different players and scenarios
    print("\n--- Testing with different players ---")

    # Test RB stats
    print("\n--- Christian McCaffrey Season Stats ---")
    cmc_stats = data_handler.get_player_season_stats("Christian McCaffrey", 2024, current_year=2024, current_week=10)
    print(cmc_stats)

    # Test WR stats
    print("\n--- Ja'Marr Chase Season Stats ---")
    chase_stats = data_handler.get_player_season_stats("Ja'Marr Chase", 2024, current_year=2024, current_week=10)
    print(chase_stats)

    # Test TE stats
    print("\n--- Brock Bowers Projections ---")
    bowers_proj = data_handler.get_player_projection_stats("Brock Bowers", 11, 2024, current_year=2024, current_week=10)
    print(bowers_proj)

    print("\n--- All tests completed successfully! ---")



if __name__ == "__main__":
    main()