from ffbench.data_handler import DataHandler
from ffbench.core import League
from ffbench.scoring import calculate_fantasy_points
from ffbench.trade import Trade
from ffbench.llm import LLMManager

class Season:
    def __init__(self, league, data_handler):
        self.league = league
        self.data_handler = data_handler

    def run_season(self):
        """
        Simulates the entire regular season.
        """
        for week in range(1, 15): # 14 weeks in a standard fantasy regular season
            print(f"--- Week {week} ---")
            self.simulate_week(week)
            self.run_trading_phase(week)

    def simulate_week(self, week):
        """
        Simulates a single week of the season.
        """
        weekly_stats = self.data_handler.get_weekly_player_stats(self.league.season, week)
        matchups = self.league.schedule[week]

        for team1, team2 in matchups:
            # 1. Get starters for each team (LLM interaction)
            team1_starters = team1.llm_manager.choose_starters(team1, week, {team1.team_id: team2.manager_name})
            team2_starters = team2.llm_manager.choose_starters(team2, week, {team2.team_id: team1.manager_name})
            team1.set_starters(week, team1_starters)
            team2.set_starters(week, team2_starters)

            # 2. Calculate scores
            team1_score = self._calculate_team_score(team1_starters, weekly_stats)
            team2_score = self._calculate_team_score(team2_starters, weekly_stats)

            # 3. Update records
            if team1_score > team2_score:
                team1.wins += 1
                team2.losses += 1
            else:
                team1.losses += 1
                team2.wins += 1
            
            print(f"{team1.manager_name} ({team1_score:.2f}) vs {team2.manager_name} ({team2_score:.2f})")

    def run_trading_phase(self, week):
        """
        Runs the trading phase for a given week.
        """
        print(f"--- Trading Phase Week {week} ---")
        all_teams = self.league.teams

        # Create a map of opponents for the week
        opponents_map = {}
        for team1, team2 in self.league.schedule[week]:
            opponents_map[team1] = team2
            opponents_map[team2] = team1

        for team in all_teams:
            if team in opponents_map:
                opponent = opponents_map[team]
                # Decide whether to propose a trade
                trade_details = team.llm_manager.propose_trade(team, opponent)

                if trade_details:
                    # Propose the trade and get a response
                    trade_proposal = Trade(team, opponent, trade_details['offered'], trade_details['requested'])
                    response = opponent.llm_manager.respond_to_trade(opponent, trade_proposal)

                    if response == "accept":
                        trade_proposal.accept()
                        print(f"Trade accepted between {team.manager_name} and {opponent.manager_name}!")
                    else:
                        trade_proposal.reject()
                        print("Trade rejected.")

    def _calculate_team_score(self, starters, weekly_stats):
        """
        Calculates the total score for a team's starters.
        """
        total_score = 0
        for player in starters:
            player_stats = weekly_stats[weekly_stats["player_id"] == player.player_id]
            if not player_stats.empty:
                total_score += calculate_fantasy_points(player_stats.iloc[0])
        return total_score 