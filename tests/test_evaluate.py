import json
import os
from pathlib import Path
from training.train import train


def test_train_metrics_generation():
    """Verify that the training process generates a valid metrics JSON file."""
    # Define output directory for the test
    test_output_dir = Path("test_models")
    os.environ["OUTPUT_DIR"] = str(test_output_dir)
    metrics_path = test_output_dir / "metrics.json"

    # Run training with minimal epochs for speed
    train(epoch=1, lr=0.1, dim=10)

    # Check if metrics file exists
    assert metrics_path.exists(), "Metrics JSON file was not created"

    # Validate metrics content
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    expected_keys = ["test_samples", "precision", "recall", "f1_score"]
    for key in expected_keys:
        assert key in metrics, f"Metric {key} missing from JSON"
        if key != "test_samples":
            assert 0 <= metrics[key] <= 1, f"Metric {key} is out of valid range [0, 1]"

    # Cleanup
    if metrics_path.exists():
        os.remove(metrics_path)
    if (test_output_dir / "sentiment_ft.bin").exists():
        os.remove(test_output_dir / "sentiment_ft.bin")
    test_output_dir.rmdir()
