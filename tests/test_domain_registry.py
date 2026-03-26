"""Tests for DomainSpec registry and error types."""
from src.db.domains import DomainSpec, DomainIngestionError, REGISTRY


def test_domain_spec_fields():
    spec = DomainSpec(
        name="test",
        tables=["foo", "bar"],
        description="A test domain",
    )
    assert spec.name == "test"
    assert spec.tables == ["foo", "bar"]
    assert spec.active is True
    assert spec.version_key == "version_id"


def test_registry_has_four_domains():
    names = [d.name for d in REGISTRY]
    assert set(names) == {"model", "polling", "candidate", "runtime"}


def test_candidate_is_inactive():
    candidate = next(d for d in REGISTRY if d.name == "candidate")
    assert candidate.active is False


def test_domain_ingestion_error_message():
    err = DomainIngestionError("model", "/path/to/file.parquet", "bad row 3")
    assert "model" in str(err)
    assert "bad row 3" in str(err)
