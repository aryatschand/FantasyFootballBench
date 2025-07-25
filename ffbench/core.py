import random
import csv
import os

class Player:
    def __init__(self, player_id, name, position):
        self.player_id = player_id
        self.name = name
        self.position = position

class Team:
    def __init__(self, team_id, manager_name, llm_manager):
        self.team_id = team_id
        self.manager_name = manager_name
        self.llm_manager = llm_manager
        self.roster = []
        self.starters = {}  # week -> list of Player
        self.wins = 0
        self.losses = 0
        self.roster_csv_path = f"team_rosters/team_{self.team_id}_roster.csv"
        self._create_roster_csv_if_not_exists()

    def _create_roster_csv_if_not_exists(self):
        os.makedirs(os.path.dirname(self.roster_csv_path), exist_ok=True)
        if not os.path.exists(self.roster_csv_path):
            with open(self.roster_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["player_id", "name", "position"])

    def add_player(self, player):
        self.roster.append(player)
        self.save_roster_to_csv()

    def remove_player(self, player):
        self.roster.remove(player)
        self.save_roster_to_csv()

    def save_roster_to_csv(self):
        with open(self.roster_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["player_id", "name", "position"])
            for p in self.roster:
                writer.writerow([p.player_id, p.name, p.position])

    def set_starters(self, week, starters):
        self.starters[week] = starters

class League:
    def __init__(self, season, teams):
        self.season = season
        self.teams = teams
        self.schedule = self._generate_schedule()

    def _generate_schedule(self):
        """
        Generates a random schedule for the league for 14 weeks.
        """
        num_weeks = 14
        schedule = {week: [] for week in range(1, num_weeks + 1)}
        teams = list(self.teams)

        for week in range(1, num_weeks + 1):
            random.shuffle(teams)
            matchups = []
            for i in range(0, len(teams), 2):
                matchups.append((teams[i], teams[i + 1]))
            schedule[week] = matchups

        return schedule 