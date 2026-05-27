"""Tests for the poll coverage diagnostics API endpoint."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routers.pollsters as pollsters_mod


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(pollsters_mod.router, prefix="/api/v1")
    return TestClient(app)


def _sample_report() -> dict:
    return {
        "metadata": {
            "total_polls": 12,
            "polls_with_xt_data": 2,
            "polls_analyzed": 1,
            "active_xt_columns": ["xt_race_black", "xt_education_college"],
            "mappable_xt_columns": ["xt_race_black", "xt_education_college"],
            "oversample_threshold": 1.2,
            "undersample_threshold": 0.8,
        },
        "summary": {
            "by_group": {
                "xt_race_black": {
                    "label": "Black",
                    "n_undersampled": 1,
                    "n_oversampled": 0,
                    "n_representative": 0,
                    "n_total_polls": 1,
                    "top_affected_types": [
                        {
                            "type_label": "4: Urban Black Belt",
                            "n_races_affected": 1,
                        }
                    ],
                }
            },
            "undersampled_ranking": [
                {
                    "group": "xt_race_black",
                    "label": "Black",
                    "n_polls_undersampled": 1,
                }
            ],
        },
        "per_poll_results": [
            {
                "race": "2026 GA Governor",
                "state": "GA",
                "pollster": "Example Polling",
                "date": "2026-05-01",
                "n_sample": 800,
                "n_groups_analyzed": 1,
                "n_undersampled": 1,
                "n_oversampled": 0,
                "gaps": [
                    {
                        "demographic_group": "xt_race_black",
                        "label": "Black",
                        "poll_share": 0.18,
                        "population_share": 0.32,
                        "ratio": 0.5625,
                        "status": "undersampled",
                        "affected_types": [
                            {
                                "type_id": 4,
                                "display_name": "Urban Black Belt",
                                "group_share": 0.54,
                                "state_weight": 0.21,
                                "exposure": 0.39,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_poll_coverage_report_success_includes_diagnostics(tmp_path, monkeypatch):
    report_path = tmp_path / "poll_coverage_report.json"
    report_path.write_text(json.dumps(_sample_report()))
    monkeypatch.setattr(pollsters_mod, "_COVERAGE_REPORT_PATH", report_path)

    response = _client().get("/api/v1/pollsters/coverage")

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["polls_analyzed"] == 1
    assert "xt_race_black" in data["summary"]["by_group"]
    assert data["summary"]["undersampled_ranking"][0] == {
        "group": "xt_race_black",
        "label": "Black",
        "n_polls_undersampled": 1,
    }
    gap = data["per_poll_results"][0]["gaps"][0]
    assert gap["affected_types"] == [
        {
            "type_id": 4,
            "display_name": "Urban Black Belt",
            "group_share": 0.54,
            "state_weight": 0.21,
            "exposure": 0.39,
        }
    ]


def test_poll_coverage_report_missing_file_returns_503(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing_poll_coverage_report.json"
    monkeypatch.setattr(pollsters_mod, "_COVERAGE_REPORT_PATH", missing_path)

    response = _client().get("/api/v1/pollsters/coverage")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Poll coverage diagnostics report not yet generated. "
        "Run: uv run python scripts/analyze_poll_coverage.py"
    )
