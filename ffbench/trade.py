class Trade:
    def __init__(self, proposing_team, receiving_team, players_offered, players_requested):
        self.proposing_team = proposing_team
        self.receiving_team = receiving_team
        self.players_offered = players_offered
        self.players_requested = players_requested
        self.status = "proposed"  # can be "proposed", "accepted", "rejected"

    def accept(self):
        self.status = "accepted"
        self._execute_trade()

    def reject(self):
        self.status = "rejected"

    def _execute_trade(self):
        # Remove players from proposing team's roster and add to receiving team's
        for player in self.players_offered:
            self.proposing_team.remove_player(player)
            self.receiving_team.add_player(player)

        # Remove players from receiving team's roster and add to proposing team's
        for player in self.players_requested:
            self.receiving_team.remove_player(player)
            self.proposing_team.add_player(player) 