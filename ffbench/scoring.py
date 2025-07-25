def calculate_fantasy_points(player_stats):
    """
    Calculates fantasy points for a player based on their weekly stats.
    Uses a standard PPR scoring system.
    """
    points = 0
    # Passing stats
    points += player_stats.get("passing_yards", 0) * 0.04
    points += player_stats.get("passing_tds", 0) * 4
    points -= player_stats.get("interceptions", 0) * 2

    # Rushing stats
    points += player_stats.get("rushing_yards", 0) * 0.1
    points += player_stats.get("rushing_tds", 0) * 6

    # Receiving stats
    points += player_stats.get("receptions", 0) * 1
    points += player_stats.get("receiving_yards", 0) * 0.1
    points += player_stats.get("receiving_tds", 0) * 6

    # Other
    points -= player_stats.get("fumbles_lost", 0) * 2

    return points 