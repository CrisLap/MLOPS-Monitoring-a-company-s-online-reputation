from fastapi.testclient import TestClient
from app.main import app
import re

client = TestClient(app)


def _parse_counter_value(metrics_text, name, label=None):
    """
    Extracts a numeric value from Prometheus-style metrics text.
    Searches for a metric by name, optionally filtered by label,
    and returns its float value or None if not found.
    """
    # find line starting with name (and optional label) and parse float value
    # if label provided, looks for name{label="value"}
    if label is None:
        pattern = rf"^{name}\s+(?P<val>[0-9.+eE-]+)$"
    else:
        pattern = rf"^{name}\{{label=\"{label}\"\}}\s+(?P<val>[0-9.+eE-]+)$"
    for line in metrics_text.splitlines():
        m = re.match(pattern, line)
        if m:
            return float(m.group("val"))
    return None


def test_metrics_endpoint_and_predict_updates_counters():
    """
    Tests that the /metrics endpoint and prediction updates counters correctly.
    Verifies that total request and per-label prediction counters are
    incrementedafter a call to the /predict endpoint.
    """
    # fetch metrics before
    before = client.get("/metrics")
    assert before.status_code == 200
    tb = before.text

    # call predict
    r = client.post("/predict", json={"text": "I love this product"})
    assert r.status_code == 200

    af = client.get("/metrics")
    assert af.status_code == 200
    ta = af.text

    # parse total requests
    bval = _parse_counter_value(tb, "sentiment_requests_total")
    aval = _parse_counter_value(ta, "sentiment_requests_total")

    # if previously not present, treat as 0
    if bval is None:
        bval = 0.0
    assert aval is not None and aval >= bval + 1

    # check per-label counter increment for returned label
    lab = r.json()["label"]
    b_label = _parse_counter_value(tb, "sentiment_predictions_total", label=lab)
    a_label = _parse_counter_value(ta, "sentiment_predictions_total", label=lab)
    if b_label is None:
        b_label = 0.0
    assert a_label is not None and a_label >= b_label + 1
