from ffbench.data_handler import DataHandler
from ffbench.core import Player, Team, League
from ffbench.draft import Draft
from ffbench.season import Season
from ffbench.llm import LLMManager

def main():
    # Define the season to simulate
    simulated_season = 2023

    # 1. Initialize the DataHandler
    data_handler = DataHandler(season=simulated_season)
    
    # 2. Create a list of teams, each with its own LLM manager
    num_teams = 10
    teams = [
        Team(
            team_id=i,
            manager_name=f"LLM_{i+1}",
            llm_manager=LLMManager(model_name=f"gpt-4-turbo-0125"),
        )
        for i in range(num_teams)
    ]

    # 3. Fetch historical player data for the draft
    player_stats = data_handler.get_historical_player_stats()

    # Filter for draftable players
    draftable_players = player_stats[
        player_stats["position"].isin(["QB", "RB", "WR", "TE"])
    ].head(200)

    # 4. Initialize the League and Draft objects
    league = League(season=simulated_season, teams=teams)
    draft = Draft(
        teams=league.teams,
        players=draftable_players,
        historical_data=player_stats,
    )

    # 5. Run the draft
    print("--- Starting Draft ---")
    draft.run_draft()
    print("--- Draft Complete ---")

    # 6. Run the season
    season = Season(league=league, data_handler=data_handler)
    season.run_season()

    # 7. Print final standings
    print("\n--- Final Standings ---")
    sorted_teams = sorted(league.teams, key=lambda t: t.wins, reverse=True)
    for team in sorted_teams:
        print(f"{team.manager_name}: {team.wins}-{team.losses}")


if __name__ == "__main__":
    main() 