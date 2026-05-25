from term_ai.experiment.metrics import (
    accuracy,
    compare_prediction_sets,
    expected_calibration_error,
    macro_f1,
    mcnemar_test,
    summarize_predictions,
)


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


def test_paired_statistics_compare_common_items():
    labels = ["A", "B", "C"]
    pred_a = ["A", "B", "A"]
    pred_b = ["B", "B", "C"]
    mcnemar = mcnemar_test(labels, pred_a, pred_b)
    assert mcnemar["b"] == 1
    assert mcnemar["c"] == 1

    rows_a = [
        {"item_id": "1", "label": "A", "prediction": "A"},
        {"item_id": "2", "label": "B", "prediction": "B"},
    ]
    rows_b = [
        {"item_id": "1", "label": "A", "prediction": "B"},
        {"item_id": "2", "label": "B", "prediction": "B"},
    ]
    result = compare_prediction_sets(rows_a, rows_b, samples=10)
    assert result["n_common"] == 2
    assert "paired_bootstrap_accuracy_delta" in result
