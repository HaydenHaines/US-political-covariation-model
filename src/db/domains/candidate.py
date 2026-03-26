"""Candidate domain — reserved for political sabermetrics silo.

active=False: no ingest() function; DomainSpec is registered to
establish the domain boundary for future implementation.
"""
from src.db.domains import DomainSpec

DOMAIN_SPEC = DomainSpec(
    name="candidate",
    tables=[],  # TBD when sabermetrics pipeline is implemented
    description="Politician stats: CTOV, district fit scores, career composites",
    active=False,
    version_key="version_id",
)
