# api/tests/test_types.py
"""Tests for the type-primary endpoints: /types, /types/{id}, /super-types."""


class TestListTypes:
    def test_returns_all_types(self, client):
        resp = client.get("/api/v1/types")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4  # 4 types in test fixture

    def test_type_has_required_fields(self, client):
        resp = client.get("/api/v1/types")
        item = resp.json()[0]
        assert "type_id" in item
        assert "super_type_id" in item
        assert "display_name" in item
        assert "n_counties" in item
        assert isinstance(item["display_name"], str)

    def test_types_have_county_counts(self, client):
        resp = client.get("/api/v1/types")
        data = resp.json()
        total_counties = sum(t["n_counties"] for t in data)
        # Each county is assigned to exactly one dominant type
        assert total_counties == 5  # 5 test counties


class TestGetType:
    def test_returns_type_detail(self, client):
        resp = client.get("/api/v1/types/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type_id"] == 0
        assert data["super_type_id"] == 0
        assert data["display_name"] == "Rural Conservative"
        assert "counties" in data
        assert isinstance(data["counties"], list)
        assert len(data["counties"]) > 0
        assert "demographics" in data

    def test_type_detail_counties_have_names(self, client):
        resp = client.get("/api/v1/types/0")
        data = resp.json()
        for county in data["counties"]:
            assert isinstance(county, dict)
            assert "county_fips" in county
            assert "county_name" in county
            assert "state_abbr" in county
            assert len(county["county_fips"]) == 5

    def test_type_detail_has_shift_profile(self, client):
        resp = client.get("/api/v1/types/0")
        data = resp.json()
        # shift_profile may be null or a dict
        assert "shift_profile" in data
        if data["shift_profile"] is not None:
            assert isinstance(data["shift_profile"], dict)

    def test_type_404_unknown(self, client):
        resp = client.get("/api/v1/types/999")
        assert resp.status_code == 404


class TestListSuperTypes:
    def test_returns_super_types(self, client):
        resp = client.get("/api/v1/super-types")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2  # 2 super-types in test fixture

    def test_super_type_has_member_ids(self, client):
        resp = client.get("/api/v1/super-types")
        data = resp.json()
        for st in data:
            assert "super_type_id" in st
            assert "display_name" in st
            assert "member_type_ids" in st
            assert "n_counties" in st
            assert isinstance(st["member_type_ids"], list)
            assert len(st["member_type_ids"]) > 0

    def test_super_type_member_ids_correct(self, client):
        resp = client.get("/api/v1/super-types")
        data = resp.json()
        st_map = {st["super_type_id"]: st for st in data}
        # super-type 0 has types 0, 1
        assert sorted(st_map[0]["member_type_ids"]) == [0, 1]
        # super-type 1 has types 2, 3
        assert sorted(st_map[1]["member_type_ids"]) == [2, 3]


class TestCountiesWithTypes:
    def test_counties_include_type_fields(self, client):
        resp = client.get("/api/v1/counties")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5
        for item in data:
            assert "dominant_type" in item
            assert "super_type" in item
            # All test counties have type assignments
            assert item["dominant_type"] is not None
            assert item["super_type"] is not None

    def test_county_type_values_correct(self, client):
        resp = client.get("/api/v1/counties")
        data = resp.json()
        county_map = {c["county_fips"]: c for c in data}
        # 12001 -> type 3, super 1
        assert county_map["12001"]["dominant_type"] == 3
        assert county_map["12001"]["super_type"] == 1
        # 12003 -> type 0, super 0
        assert county_map["12003"]["dominant_type"] == 0
        assert county_map["12003"]["super_type"] == 0
