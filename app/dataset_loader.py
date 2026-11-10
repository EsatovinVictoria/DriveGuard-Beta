"""DriveGuard Dataset Loader Module.
Responsible for scanning, indexing, parsing metadata, and loading samples from
the NTHU-DDD (Driver Drowsiness Detection) dataset.

Strictly follows methodology rules:
- Ground truth labels originate purely from folder classification ('drowsy' vs 'notdrowsy').
- Original files are strictly treated as read-only.
- Parses subject ID and video scenario to prevent train/test data leakage.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.config import (
    CLASS_LABELS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    get_dataset_root,
)


class DatasetSample:
    """Represents a single data sample with associated metadata."""

    def __init__(
        self,
        file_path: Union[str, Path],
        label: str,
        label_id: int,
        subject_id: Optional[str] = None,
        source_file: Optional[str] = None,
        file_type: str = "image",
        scenario: Optional[str] = None,
        frame_index: Optional[int] = None,
    ):
        self.file_path = Path(file_path).resolve()
        self.label = label
        self.label_id = label_id
        self.subject_id = subject_id
        self.source_file = source_file or self.file_path.name
        self.file_type = file_type
        self.scenario = scenario
        self.frame_index = frame_index

    def to_dict(self) -> Dict[str, Union[str, int, Optional[str], Optional[int]]]:
        """Convert sample metadata to standard dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "label": self.label,
            "label_id": self.label_id,
            "subject_id": self.subject_id,
            "source_file": self.source_file,
            "file_type": self.file_type,
            "scenario": self.scenario,
            "frame_index": self.frame_index,
        }

    def load_image(self) -> Optional[np.ndarray]:
        """Load image in BGR format with error checking."""
        if not self.file_path.exists():
            return None
        img = cv2.imread(str(self.file_path))
        return img


class NTHUDatasetLoader:
    """Scanner and manager for NTHU-DDD dataset."""

    def __init__(self, dataset_root: Optional[Union[str, Path]] = None):
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else get_dataset_root()
        self.samples: List[DatasetSample] = []
        self.unreadable_files: List[str] = []
        self._is_indexed = False

    def scan(self, max_samples_per_class: Optional[int] = None) -> Dict[str, Union[int, Dict[str, int]]]:
        """Scan dataset directory, validate structure, and index metadata."""
        self.samples.clear()
        self.unreadable_files.clear()

        if not self.dataset_root.exists() or not self.dataset_root.is_dir():
            return {
                "status": "error",
                "message": f"Dataset root not found: {self.dataset_root}",
                "total_samples": 0,
                "counts": {"drowsy": 0, "notdrowsy": 0},
            }

        counts = {"drowsy": 0, "notdrowsy": 0}

        for folder_name, label_id in CLASS_LABELS.items():
            folder_path = self.dataset_root / folder_name
            if not folder_path.exists() or not folder_path.is_dir():
                continue

            # Iterate over files in the class folder
            class_count = 0
            with os.scandir(folder_path) as it:
                for entry in it:
                    if not entry.is_file():
                        continue

                    file_ext = Path(entry.name).suffix.lower()
                    if file_ext in SUPPORTED_IMAGE_EXTENSIONS:
                        file_type = "image"
                    elif file_ext in SUPPORTED_VIDEO_EXTENSIONS:
                        file_type = "video"
                    else:
                        continue

                    # Parse metadata from NTHU filename structure
                    # Example: 001_glasses_sleepyCombination_1003_drowsy.jpg
                    subject_id, scenario, frame_index = self._parse_filename_metadata(entry.name)

                    sample = DatasetSample(
                        file_path=entry.path,
                        label=folder_name,
                        label_id=label_id,
                        subject_id=subject_id,
                        source_file=entry.name,
                        file_type=file_type,
                        scenario=scenario,
                        frame_index=frame_index,
                    )
                    self.samples.append(sample)
                    class_count += 1

                    if max_samples_per_class and class_count >= max_samples_per_class:
                        break

            counts[folder_name] = class_count

        self._is_indexed = True
        return {
            "status": "success",
            "dataset_root": str(self.dataset_root),
            "total_samples": len(self.samples),
            "counts": counts,
        }

    def _parse_filename_metadata(self, filename: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Extract subject_id, scenario, and frame_index from filename if valid."""
        stem = Path(filename).stem
        parts = stem.split("_")

        subject_id: Optional[str] = None
        scenario: Optional[str] = None
        frame_index: Optional[int] = None

        # Check standard NTHU pattern: [subject]_[scenario...]_[frame]_[label]
        if len(parts) >= 3:
            # Check if first part looks like subject ID (digits)
            if re.match(r"^\d+$", parts[0]):
                subject_id = parts[0]

            # Check if second to last is frame number
            if re.match(r"^\d+$", parts[-2]):
                frame_index = int(parts[-2])
                scenario = "_".join(parts[1:-2])
            elif re.match(r"^\d+$", parts[-1]):
                frame_index = int(parts[-1])
                scenario = "_".join(parts[1:-1])
            else:
                scenario = "_".join(parts[1:-1])

        return subject_id, scenario, frame_index

    def get_summary(self) -> Dict[str, Union[str, int, Dict[str, int], List[str]]]:
        """Return dataset statistics and metadata summary."""
        if not self._is_indexed:
            self.scan()

        counts = {"drowsy": 0, "notdrowsy": 0}
        subjects = set()
        scenarios = set()

        for s in self.samples:
            counts[s.label] = counts.get(s.label, 0) + 1
            if s.subject_id:
                subjects.add(s.subject_id)
            if s.scenario:
                scenarios.add(s.scenario)

        return {
            "dataset_root": str(self.dataset_root),
            "total_samples": len(self.samples),
            "counts": counts,
            "unique_subjects": sorted(list(subjects)),
            "subject_count": len(subjects),
            "scenarios": sorted(list(scenarios)),
            "unreadable_count": len(self.unreadable_files),
        }

    def get_samples(
        self,
        label: Optional[str] = None,
        subject_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Union[str, int, Optional[str], Optional[int]]]]:
        """Retrieve paginated sample metadata for UI exploration."""
        if not self._is_indexed:
            self.scan()

        filtered = self.samples
        if label:
            filtered = [s for s in filtered if s.label == label]
        if subject_id:
            filtered = [s for s in filtered if s.subject_id == subject_id]

        paged = filtered[offset : offset + limit]
        return [s.to_dict() for s in paged]

    def get_sample_by_index(self, index: int) -> Optional[DatasetSample]:
        """Fetch a specific sample by its scan index."""
        if not self._is_indexed:
            self.scan()
        if 0 <= index < len(self.samples):
            return self.samples[index]
        return None

    def find_sample_by_path(self, file_path: Union[str, Path]) -> Optional[DatasetSample]:
        """Find sample matching a given path."""
        target_path = Path(file_path).resolve()
        for s in self.samples:
            if s.file_path == target_path:
                return s
        return None
