# api/tests/test_counties.py
def test_counties_returns_all_with_community(client):
    resp = client.get("/api/v1/counties")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 5  # TEST_FIPS count
    item = data[0]
    assert "county_fips" in item
    assert "state_abbr" in item
    assert "community_id" in item
    assert isinstance(item["community_id"], int)
