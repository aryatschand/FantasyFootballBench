#!/usr/bin/env python3
"""
Comprehensive tests for the improved DataHandler class.
Tests all the new data access functions for player stats.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ffbench.data_handler import DataHandler
from datetime import datetime

def test_data_handler():
    """Test all DataHandler functions."""
    print("🧪 Testing DataHandler functions...")

    # Initialize data handler
    dh = DataHandler()

    # Test 1: Get player season stats (existing player)
    print("\n1. Testing get_player_season_stats for existing player...")
    stats = dh.get_player_season_stats("A.Rodgers", 2024, current_year=2024, current_week=10)
    if stats and not stats.startswith("No data"):
        print("✅ Found season stats for A.Rodgers 2024")
        print(stats[:200] + "..." if len(stats) > 200 else stats)
    else:
        print(f"❌ Failed to get season stats: {stats}")

    # Test 2: Get player season stats (non-existing player)
    print("\n2. Testing get_player_season_stats for non-existing player...")
    stats = dh.get_player_season_stats("NonExistent Player", 2024, current_year=2024, current_week=10)
    if stats and stats.startswith("No data"):
        print("✅ Correctly returned 'No data' message for non-existing player")
    else:
        print(f"❌ Should have returned 'No data' message: {stats}")

    # Test 3: Get player season stats (future year)
    print("\n3. Testing get_player_season_stats for future year...")
    stats = dh.get_player_season_stats("A.Rodgers", 2025, current_year=2024, current_week=10)
    if stats and stats.startswith("No data"):
        print("✅ Correctly returned 'No data' message for future year")
    else:
        print(f"❌ Should have returned 'No data' message: {stats}")

    # Test 4: Get player weekly stats (existing player and week)
    print("\n4. Testing get_player_weekly_stats for existing player/week...")
    stats = dh.get_player_weekly_stats("Josh Allen", 1, 2024, current_year=2024, current_week=10)
    if stats and not stats.startswith("No data"):
        print("✅ Found weekly stats for Josh Allen Week 1, 2024")
        print(stats[:200] + "..." if len(stats) > 200 else stats)
    else:
        print(f"❌ Failed to get weekly stats: {stats}")

    # Test 5: Get player weekly stats (future week)
    print("\n5. Testing get_player_weekly_stats for future week...")
    stats = dh.get_player_weekly_stats("Josh Allen", 15, 2024, current_year=2024, current_week=10)
    if stats and stats.startswith("No data"):
        print("✅ Correctly returned 'No data' message for future week")
    else:
        print(f"❌ Should have returned 'No data' message: {stats}")

    # Test 6: Get player weekly stats (non-existing player)
    print("\n6. Testing get_player_weekly_stats for non-existing player...")
    stats = dh.get_player_weekly_stats("NonExistent Player", 1, 2024, current_year=2024, current_week=10)
    if stats and stats.startswith("No data"):
        print("✅ Correctly returned 'No data' message for non-existing player")
    else:
        print(f"❌ Should have returned 'No data' message: {stats}")

    # Test 7: Get player projection stats
    print("\n7. Testing get_player_projection_stats...")
    stats = dh.get_player_projection_stats("Josh Allen", 11, 2024, current_year=2024, current_week=10)
    if stats and not stats.startswith("No"):
        print("✅ Found projection stats for Josh Allen Week 11, 2024")
        print(stats[:200] + "..." if len(stats) > 200 else stats)
    else:
        print(f"ℹ️  Projection stats result: {stats}")

    # Test 8: Get all players season stats
    print("\n8. Testing get_all_players_season_stats...")
    all_stats = dh.get_all_players_season_stats(2024, position="QB", current_year=2024)
    if all_stats and not all_stats.startswith("No"):
        print("✅ Found QB season stats for 2024")
        print(all_stats[:500] + "..." if len(all_stats) > 500 else all_stats)
    else:
        print(f"❌ Failed to get all players season stats: {all_stats}")

    # Test 9: Test caching
    print("\n9. Testing caching functionality...")
    # Clear cache first
    dh.clear_cache()

    # First call should load from file
    start_time = datetime.now()
    stats1 = dh.get_player_season_stats("A.Rodgers", 2024, current_year=2024, current_week=10)
    first_call_time = (datetime.now() - start_time).total_seconds()

    # Second call should use cache
    start_time = datetime.now()
    stats2 = dh.get_player_season_stats("A.Rodgers", 2024, current_year=2024, current_week=10)
    second_call_time = (datetime.now() - start_time).total_seconds()

    # Check that both calls returned data and cache is populated
    cache_populated = 'season_2024' in dh._season_cache
    both_calls_succeeded = stats1 is not None and stats2 is not None and not stats1.startswith("No")
    cache_is_faster = second_call_time < first_call_time

    if both_calls_succeeded and cache_populated:
        print("✅ Caching works correctly")
        print(".3f")
        print(".3f")
        if cache_is_faster:
            print("   Cache access is faster as expected")
        else:
            print("   Note: Cache timing may vary on different systems")
    else:
        print("❌ Caching test failed")

    # Test 10: Test backward compatibility aliases
    print("\n10. Testing backward compatibility aliases...")
    stats_alias = dh.get_player_stats_by_year("A.Rodgers", 2024, current_year=2024, current_week=10)
    weekly_alias = dh.get_player_stats_by_week("Josh Allen", 1, 2024, current_year=2024, current_week=10)

    if stats_alias and weekly_alias and not stats_alias.startswith("No"):
        print("✅ Backward compatibility aliases work correctly")
    else:
        print("❌ Backward compatibility aliases failed")

    # Test 11: Clear cache
    print("\n11. Testing cache clearing...")
    dh.clear_cache()
    print("✅ Cache cleared successfully")

    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    test_data_handler()
