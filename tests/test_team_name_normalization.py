from tools.team_name_normalization import canonical_team_name, normalized_team_key

def test_minimum_aliases():
    assert canonical_team_name("Korea Republic") == "South Korea"
    assert canonical_team_name("Cape Verde") == "Cabo Verde"
    assert canonical_team_name("Ivory Coast") == "Côte d'Ivoire"
    assert canonical_team_name("Cote d Ivoire") == "Côte d'Ivoire"
    assert canonical_team_name("Bosnia & Herzegovina") == "Bosnia and Herzegovina"
    assert canonical_team_name("Czechia") == "Czech Republic"
    assert canonical_team_name("Türkiye") == "Turkey"
    assert canonical_team_name("USA") == "United States"
    assert canonical_team_name("United States of America") == "United States"

def test_normalized_keys_are_stable():
    assert normalized_team_key("U.S.A.") == normalized_team_key("United States")
