# api/tests/test_meta.py
def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db"] == "connected"
    assert data["contract"] == "ok"


def test_health_reports_degraded_without_types(client_no_types):
    """Health endpoint reports degraded when contract tables are missing."""
    resp = client_no_types.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["contract"] == "degraded"


def test_model_version_returns_fields(client):
    resp = client.get("/api/v1/model/version")
    assert resp.status_code == 200
    data = resp.json()
    assert "version_id" in data
    assert "k" in data
    assert data["k"] == 3  # TEST_K
    assert "holdout_r" in data
    assert "created_at" in data
