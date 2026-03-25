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


class TestCountyDetail:
    def test_returns_county_detail(self, client):
        resp = client.get("/api/v1/counties/01001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["county_fips"] == "01001"
        assert data["county_name"] == "Autauga County, AL"
        assert data["state_abbr"] == "AL"
        assert data["dominant_type"] == 2
        assert data["super_type"] == 1
        assert data["type_display_name"] == "Suburban Moderate"
        assert data["super_type_display_name"] == "Suburban & Moderate"

    def test_has_narrative(self, client):
        resp = client.get("/api/v1/counties/01001")
        data = resp.json()
        assert data["narrative"] == "A suburban type."

    def test_has_pred_dem_share(self, client):
        resp = client.get("/api/v1/counties/01001")
        data = resp.json()
        # All test predictions are 0.42
        assert data["pred_dem_share"] is not None
        assert abs(data["pred_dem_share"] - 0.42) < 0.01

    def test_has_demographics(self, client):
        resp = client.get("/api/v1/counties/01001")
        data = resp.json()
        assert "demographics" in data
        demo = data["demographics"]
        assert isinstance(demo, dict)
        assert "median_hh_income" in demo
        assert demo["median_hh_income"] == 68000.0
        assert "pct_bachelors_plus" in demo
        assert "pct_white_nh" in demo

    def test_has_sibling_counties(self, client):
        resp = client.get("/api/v1/counties/01001")
        data = resp.json()
        assert "sibling_counties" in data
        siblings = data["sibling_counties"]
        assert isinstance(siblings, list)
        # 01001 is type 2 — no other county in fixture is type 2, so empty
        # Actually check: county_type_map = 12001:3, 12003:0, 13001:1, 13003:0, 01001:2
        assert len(siblings) == 0

    def test_sibling_counties_populated(self, client):
        # 12003 and 13003 are both type 0 — they should be siblings
        resp = client.get("/api/v1/counties/12003")
        data = resp.json()
        siblings = data["sibling_counties"]
        assert len(siblings) == 1
        assert siblings[0]["county_fips"] == "13003"

    def test_404_unknown_fips(self, client):
        resp = client.get("/api/v1/counties/99999")
        assert resp.status_code == 404

    def test_demographics_excludes_county_fips(self, client):
        resp = client.get("/api/v1/counties/01001")
        data = resp.json()
        assert "county_fips" not in data["demographics"]
