from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from industrial_tsfm.platform.api import create_app


def test_dashboard_is_served_from_platform_root(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "results", tmp_path / "workspace"))
    response = client.get("/")

    assert response.status_code == 200
    assert "IndusTSFM Studio" in response.text
    assert "Industrial Time-Series Intelligence" in response.text
    assert "/v1/capabilities" in response.text
