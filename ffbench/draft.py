from ffbench.core import Team, Player
from ffbench.llm import LLMManager

class Draft:
    def __init__(self, teams, players, historical_data):
        self.teams = teams
        self.players = players
        self.draft_order = self._generate_draft_order()
        self.historical_data = historical_data

    def _generate_draft_order(self):
        """
        Generates the snake draft order.
        """
        draft_order = []
        num_rounds = 15  # Typical fantasy draft size
        for i in range(num_rounds):
            if (i + 1) % 2 != 0:
                draft_order.extend(self.teams)
            else:
                draft_order.extend(reversed(self.teams))
        return draft_order

    def run_draft(self):
        """
        Runs the snake draft using the LLM manager.
        """
        drafted_player_ids = []
        for i, team in enumerate(self.draft_order):
            available_players = self.players[~self.players['player_id'].isin(drafted_player_ids)]
            
            # Get pick from LLM
            picked_player_df = team.llm_manager.make_draft_pick(team, available_players, self.historical_data)
            picked_player = Player(
                player_id=picked_player_df["player_id"],
                name=picked_player_df["player_name"],
                position=picked_player_df["position"],
            )

            team.add_player(picked_player)
            drafted_player_ids.append(picked_player.player_id)
            
            print(f"Round {i // len(self.teams) + 1}, Pick {i % len(self.teams) + 1}: {team.manager_name} drafts {picked_player.name}") 