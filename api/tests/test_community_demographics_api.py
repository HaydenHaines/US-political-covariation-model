# api/tests/test_community_demographics_api.py
"""API tests for the demographics field added to /communities/{id}."""


def test_community_detail_has_demographics_field(client):
    """GET /communities/{id} must include a 'demographics' key in the response."""
    resp = client.get("/api/v1/communities/0")
    assert resp.status_code == 200
    data = resp.json()
    # demographics may be null when the test DB has no community_profiles table,
    # but the key must always be present in the serialised response.
    assert "demographics" in data


def test_community_detail_demographics_null_when_no_profiles_table(client):
    """Demographics is None when community_profiles table is absent (test fixture).

    The real DuckDB has this table; the in-memory test fixture does not.
    The router must fall back gracefully rather than raising a 500.
    """
    resp = client.get("/api/v1/communities/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["demographics"] is None
