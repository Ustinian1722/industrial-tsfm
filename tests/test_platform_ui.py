from industrial_tsfm.platform.ui import render_dashboard


def test_dashboard_surfaces_analytics_and_runtime_boundaries() -> None:
    html = render_dashboard()

    assert "IndusTSFM Studio" in html
    assert "Industrial analytics" in html
    assert "analyticsCount" in html
    assert "variable discovery is a transparent screening heuristic" in html
    assert "Live MQTT/OPC-UA execution" in html
