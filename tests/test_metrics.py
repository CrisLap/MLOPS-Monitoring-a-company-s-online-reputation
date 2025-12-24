from app import metrics


def test_request_count_counter_increment():
    """
    Tests that the REQUEST_COUNT counter increments correctly.
    Verifies that calling inc() on the counter increases its
    internal value by one.
    """
    before = metrics.REQUEST_COUNT._value.get()
    metrics.REQUEST_COUNT.inc()
    after = metrics.REQUEST_COUNT._value.get()
    assert after == before + 1


def _get_hist_sum_and_count(h):
    """
    Retrieves the sum and count from a Prometheus histogram metric.
    Iterates over collected samples to extract values of '_sum' and '_count'.
    """
    s = 0.0
    c = 0.0
    for metric in h.collect():
        for sample in metric.samples:
            # sample is a namedtuple with (name, labels, value)
            if sample.name.endswith("_sum"):
                s = sample.value
            if sample.name.endswith("_count"):
                c = sample.value
    return s, c


def test_sentiment_counter_labels_and_histogram():
    """
    Tests labeled counters and histogram observation updates correctly.
    Verifies that labeled counters increment as expected and that
    histogram sum and count reflect observed values.
    """
    # labeled counter
    before_pos = metrics.SENTIMENT_COUNTER.labels("positive")._value.get()
    metrics.SENTIMENT_COUNTER.labels("positive").inc()
    after_pos = metrics.SENTIMENT_COUNTER.labels("positive")._value.get()
    assert after_pos == before_pos + 1

    # histogram observe via public collect API
    before_sum, before_count = _get_hist_sum_and_count(metrics.REQUEST_LATENCY)
    metrics.REQUEST_LATENCY.observe(0.123)
    after_sum, after_count = _get_hist_sum_and_count(metrics.REQUEST_LATENCY)

    assert after_sum >= before_sum + 0.123
    assert after_count >= before_count + 1
