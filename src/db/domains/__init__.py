"""Data domain registry for the WetherVane pipeline."""
from __future__ import annotations
from dataclasses import dataclass


class DomainIngestionError(Exception):
    """Raised when a domain's source data fails validation or is missing."""

    def __init__(self, domain: str, path: str, reason: str) -> None:
        self.domain = domain
        self.path = path
        self.reason = reason
        super().__init__(f"[{domain}] {path}: {reason}")


@dataclass
class DomainSpec:
    name: str               # "model" | "polling" | "candidate" | "runtime"
    tables: list[str]       # DuckDB tables this domain owns
    description: str
    active: bool = True     # False = reserved, skip on build
    version_key: str = "version_id"  # discriminator column name (docs only)


REGISTRY: list[DomainSpec] = [
    # NOTE: The table list here must stay in sync with DOMAIN_SPEC in model.py.
    # model.py owns the authoritative list used by ingest(); this entry is
    # for REGISTRY discovery only.
    DomainSpec(
        name="model",
        tables=[
            "type_scores", "type_covariance", "type_priors",
            "ridge_county_priors", "hac_state_weights", "hac_county_weights",
        ],
        description="KMeans type scores, covariance matrix, priors, ridge predictions, HAC fallback weights",
        version_key="version_id",
    ),
    DomainSpec(
        name="polling",
        tables=["polls", "poll_crosstabs", "poll_notes"],
        description="Poll rows ingested from CSV; crosstabs and quality notes queryable via SQL",
        version_key="cycle",
    ),
    DomainSpec(
        name="candidate",
        tables=[],
        description="Politician stats: CTOV, district fit scores, career composites",
        active=False,
        version_key="version_id",
    ),
    DomainSpec(
        name="runtime",
        tables=[],
        description="User what-ifs and recalculate inputs — always API request bodies, never persisted",
        active=False,
        version_key="n/a",
    ),
]
