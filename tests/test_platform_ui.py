from industrial_tsfm.platform.ui import render_dashboard


def test_dashboard_surfaces_analytics_and_runtime_boundaries() -> None:
    html = render_dashboard()

    assert "IndusTSFM Studio" in html
    assert "Industrial analytics" in html
    assert "analyticsCount" in html
    assert "variable discovery is a transparent screening heuristic" in html
    assert "OPC-UA Live" in html
    assert "/v1/opcua/runtimes" in html
    assert "browse/read/subscribe only" in html
    assert "OPC-UA writes, setpoint changes and closed-loop control are not exposed" in html
    assert "strictly read-only OPC-UA live acquisition" in html
    assert "MQTT remains contract-only" in html
