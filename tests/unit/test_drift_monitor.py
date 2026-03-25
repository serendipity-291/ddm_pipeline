import drift_monitor
import pytest


@pytest.mark.unit
def test_check_ks_drift_returns_none_when_data_insufficient(monkeypatch):
    calls = {"n": 0}

    def fake_get_rms_data(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [1.0] * 20  # baseline too short (<50)
        return [1.0] * 30

    monkeypatch.setattr(drift_monitor, "get_rms_data", fake_get_rms_data)
    assert drift_monitor.check_ks_drift(query_api=None) is None


@pytest.mark.unit
def test_check_ks_drift_returns_pvalue(monkeypatch):
    calls = {"n": 0}

    def fake_get_rms_data(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return [0.1] * 60
        return [0.2] * 20

    class FakeStats:
        @staticmethod
        def ks_2samp(a, b):
            return 0.99, 0.0123

    monkeypatch.setattr(drift_monitor, "get_rms_data", fake_get_rms_data)
    monkeypatch.setattr(drift_monitor, "stats", FakeStats)

    p_value = drift_monitor.check_ks_drift(query_api=None)
    assert p_value == 0.0123
