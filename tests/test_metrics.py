from term_ai.experiment.metrics import accuracy, expected_calibration_error, macro_f1, summarize_predictions


def test_metric_summary_contract():
    predictions = [
        {"label": "A", "prediction": "A", "confidence": 0.9, "latency_ms": 10, "task_type": "Synonym Selection"},
        {"label": "B", "prediction": "A", "confidence": 0.6, "latency_ms": 20, "task_type": "Synonym Selection"},
        {"label": "B", "prediction": "B", "confidence": 0.8, "latency_ms": 30, "task_type": "Antonym Selection"},
    ]
    summary = summarize_predictions(predictions)
    assert summary["n"] == 3
    assert summary["accuracy"] == accuracy(["A", "B", "B"], ["A", "A", "B"])
    assert summary["macro_f1"] == macro_f1(["A", "B", "B"], ["A", "A", "B"])
    assert summary["ece"] == expected_calibration_error([1, 0, 1], [0.9, 0.6, 0.8])
    assert summary["latency_p95"] == 30.0
