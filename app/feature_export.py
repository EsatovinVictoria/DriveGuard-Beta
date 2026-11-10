"""DriveGuard Feature Export Module.
Saves extracted Improved HOG feature vectors and comprehensive metadata to CSV and JSON.

Strict Requirements:
- Format: sample_id, source_file, subject_id, label, label_id, hog_000, hog_001, ...
- Strict ground truth preservation from folder labels ('drowsy' vs 'notdrowsy').
- Full validation: zero NaN or Infinity values permitted.
- Complete auditing: failed extractions are logged with explicit error reasons in reports.
- Output artifacts saved to outputs/features/ for downstream Naïve Bayes modeling.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from app.config import (
    FEATURES_DIR,
    REPORTS_DIR,
)


class FeatureExporter:
    """Manages tabular CSV export and audit reporting for feature vectors."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir).resolve() if output_dir else FEATURES_DIR
        self.reports_dir = REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[Dict[str, Any]] = []
        self.metadata_records: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []
        self.expected_dim: Optional[int] = None

    def add_sample(
        self,
        sample_id: Union[str, int],
        source_file: str,
        label: str,
        label_id: int,
        feature_vector: List[int],
        subject_id: Optional[str] = None,
        scenario: Optional[str] = None,
        frame_index: Optional[int] = None,
        processing_time_ms: float = 0.0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a successful feature extraction sample."""
        # Sanity check on vector
        if feature_vector is None or len(feature_vector) == 0:
            self.add_failure(
                sample_id=sample_id,
                source_file=source_file,
                label=label,
                reason="Feature vector is empty or None",
            )
            return False

        # Validate no NaN or Inf
        arr = np.array(feature_vector, dtype=float)
        if np.isnan(arr).any() or np.isinf(arr).any():
            self.add_failure(
                sample_id=sample_id,
                source_file=source_file,
                label=label,
                reason="Feature vector contains NaN or Infinite values",
            )
            return False

        dim = len(feature_vector)
        if self.expected_dim is None:
            self.expected_dim = dim
        elif dim != self.expected_dim:
            self.add_failure(
                sample_id=sample_id,
                source_file=source_file,
                label=label,
                reason=f"Dimension mismatch: expected {self.expected_dim}, got {dim}",
            )
            return False

        record = {
            "sample_id": sample_id,
            "source_file": source_file,
            "subject_id": subject_id if subject_id is not None else "None",
            "label": label,
            "label_id": label_id,
            "feature_vector": feature_vector,
        }
        self.records.append(record)

        meta = {
            "sample_id": sample_id,
            "source_file": source_file,
            "subject_id": subject_id,
            "label": label,
            "label_id": label_id,
            "scenario": scenario,
            "frame_index": frame_index,
            "feature_dimension": dim,
            "processing_time_ms": processing_time_ms,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        self.metadata_records.append(meta)
        return True

    def add_failure(
        self,
        sample_id: Union[str, int],
        source_file: str,
        label: str,
        reason: str,
        subject_id: Optional[str] = None,
    ) -> None:
        """Record an extraction failure for audit transparency."""
        self.failures.append({
            "sample_id": sample_id,
            "source_file": source_file,
            "subject_id": subject_id,
            "label": label,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    def export_csv(
        self,
        features_filename: str = "hog_features.csv",
        metadata_filename: str = "feature_metadata.csv",
        report_filename: str = "extraction_report.json",
    ) -> Dict[str, Any]:
        """Write features and audit report to disk."""
        if not self.records:
            return {
                "status": "warning",
                "message": "No valid records to export",
                "exported_count": 0,
                "failure_count": len(self.failures),
            }

        features_path = self.output_dir / features_filename
        metadata_path = self.output_dir / metadata_filename
        report_path = self.output_dir / report_filename

        dim = self.expected_dim or len(self.records[0]["feature_vector"])
        hog_headers = [f"hog_{i:04d}" for i in range(dim)]
        full_headers = ["sample_id", "source_file", "subject_id", "label", "label_id"] + hog_headers

        # 1. Write hog_features.csv
        with open(features_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(full_headers)
            for r in self.records:
                row = [
                    r["sample_id"],
                    r["source_file"],
                    r["subject_id"],
                    r["label"],
                    r["label_id"],
                ] + r["feature_vector"]
                writer.writerow(row)

        # 2. Write feature_metadata.csv
        if self.metadata_records:
            meta_keys = list(self.metadata_records[0].keys())
            with open(metadata_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=meta_keys)
                writer.writeheader()
                writer.writerows(self.metadata_records)

        # 3. Write extraction_report.json
        drowsy_count = sum(1 for r in self.records if r["label"] == "drowsy")
        notdrowsy_count = sum(1 for r in self.records if r["label"] == "notdrowsy")

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "features_file": str(features_path),
            "metadata_file": str(metadata_path),
            "total_processed": len(self.records) + len(self.failures),
            "total_success": len(self.records),
            "total_failed": len(self.failures),
            "class_distribution": {
                "drowsy": drowsy_count,
                "notdrowsy": notdrowsy_count,
            },
            "feature_dimension": dim,
            "failures": self.failures,
        }

        with open(report_path, mode="w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        return {
            "status": "success",
            "features_csv": str(features_path),
            "metadata_csv": str(metadata_path),
            "report_json": str(report_path),
            "total_success": len(self.records),
            "total_failed": len(self.failures),
            "dimension": dim,
        }
