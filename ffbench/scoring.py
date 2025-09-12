def calculate_fantasy_points(player_stats, scoring_system=None):
    """
    Calculates fantasy points for a player based on their weekly stats.
    Uses a configurable scoring system.

    Args:
        player_stats (dict): Dictionary of player stats (e.g., {"passing_yards": 250, "passing_tds": 2}).
        scoring_system (dict, optional): Dictionary mapping stat names to points per unit.
            Defaults to standard PPR scoring if None.

    Returns:
        float: Total fantasy points.
    """
    if scoring_system is None:
        scoring_system = {
            "passing_yards": 0.04,
            "passing_tds": 4,
            "interceptions": -2,
            "rushing_yards": 0.1,
            "rushing_tds": 6,
            "receptions": 1,
            "receiving_yards": 0.1,
            "receiving_tds": 6,
            "fumbles_lost": -2
        }

    points = 0
    # Passing stats
    points += player_stats.get("passing_yards", 0) * scoring_system.get("passing_yards", 0)
    points += player_stats.get("passing_tds", 0) * scoring_system.get("passing_tds", 0)
    points += player_stats.get("interceptions", 0) * scoring_system.get("interceptions", 0)

    # Rushing stats
    points += player_stats.get("rushing_yards", 0) * scoring_system.get("rushing_yards", 0)
    points += player_stats.get("rushing_tds", 0) * scoring_system.get("rushing_tds", 0)

    # Receiving stats
    points += player_stats.get("receptions", 0) * scoring_system.get("receptions", 0)
    points += player_stats.get("receiving_yards", 0) * scoring_system.get("receiving_yards", 0)
    points += player_stats.get("receiving_tds", 0) * scoring_system.get("receiving_tds", 0)

    # Other
    points += player_stats.get("fumbles_lost", 0) * scoring_system.get("fumbles_lost", 0)

    return points 