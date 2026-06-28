from tools.location_advantage import location_advantage_points, match_location_adjustment

def test_host_country_gets_100_elo_points():
    assert location_advantage_points('Mexico', 'Mexico') == 100.0

def test_regional_concacaf_gets_35_points():
    assert location_advantage_points('United States', 'Canada', 'CONCACAF') == 35.0

def test_neutral_game_gets_zero():
    assert match_location_adjustment('Brazil', 'France', 'Canada') == (0.0, 0.0)
