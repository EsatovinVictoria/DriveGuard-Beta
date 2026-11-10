"""DriveGuard Experimental Backend Server.
Powered by FastAPI, OpenCV, and NumPy.
Serves computer vision experimental endpoints and local web visualization.
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    BASE_DIR,
    FEATURES_DIR,
    OUTPUTS_DIR,
)
from app.dataset_loader import NTHUDatasetLoader
from app.feature_export import FeatureExporter
from app.pipeline import get_pipeline
from app.visualization import image_to_base64_data_url

# Initialize FastAPI App
app = FastAPI(
    title="DriveGuard Computer Vision Backend",
    description="Experimental pipeline based on Bakheet & Al-Hamadi (Brain Sciences 2021)",
    version="0.1.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State Managers
dataset_loader = NTHUDatasetLoader()
feature_exporter = FeatureExporter()
pipeline = get_pipeline()

# Mount static files
WEB_DIR = BASE_DIR / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# Pydantic Request Models
class ProcessImageRequest(BaseModel):
    image_base64: Optional[str] = None
    file_path: Optional[str] = None
    sample_metadata: Optional[Dict[str, Any]] = None
    eye_mode: str = "APPROXIMATION_HAAR"
    manual_box: Optional[Dict[str, int]] = None


class LoadSampleRequest(BaseModel):
    file_path: str


class BatchExtractRequest(BaseModel):
    max_samples: int = 50
    label_filter: Optional[str] = None


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve web user interface."""
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


@app.get("/api/health")
async def health_check():
    """Verify backend and OpenCV availability."""
    return {
        "status": "healthy",
        "service": "DriveGuard-CV",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }


@app.get("/api/dataset/summary")
async def get_dataset_summary():
    """Get NTHU-DDD dataset metadata, counts, and scan status."""
    summary = dataset_loader.get_summary()
    return summary


@app.get("/api/dataset/samples")
async def get_dataset_samples(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    label: Optional[str] = None,
    subject_id: Optional[str] = None,
):
    """Retrieve paginated dataset samples for the UI dropdown."""
    samples = dataset_loader.get_samples(
        label=label,
        subject_id=subject_id,
        limit=limit,
        offset=offset,
    )
    return samples


@app.post("/api/load-sample-image")
async def load_sample_image(req: LoadSampleRequest):
    """Load an image file from the dataset and return base64 preview."""
    target_path = Path(req.file_path).resolve()
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    img = cv2.imread(str(target_path))
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image file")

    h, w = img.shape[:2]
    data_url = image_to_base64_data_url(img)

    return {
        "status": "success",
        "filename": target_path.name,
        "width": w,
        "height": h,
        "image_base64": data_url,
    }


@app.post("/api/process-image")
async def process_image_endpoint(req: ProcessImageRequest):
    """Execute complete CV pipeline (Preprocessing -> Face -> Eye -> HOG)."""
    image: Optional[np.ndarray] = None
    source_name = "unknown"

    # Source 1: File path
    if req.file_path:
        p = Path(req.file_path).resolve()
        if p.exists():
            image = cv2.imread(str(p))
            source_name = p.name

    # Source 2: Base64 string
    if image is None and req.image_base64:
        try:
            b64_str = req.image_base64
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            source_name = "uploaded_image"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="No valid image could be decoded")

    # Run pipeline
    result = pipeline.process_frame(
        frame=image,
        eye_mode=req.eye_mode,
        manual_eye_box=req.manual_box,
        include_visualizations=True,
    )

    # If successful and metadata is present, buffer into feature exporter
    if result["status"] == "success" and req.sample_metadata:
        meta = req.sample_metadata
        feature_exporter.add_sample(
            sample_id=meta.get("frame_index") or meta.get("source_file", source_name),
            source_file=meta.get("source_file", source_name),
            label=meta.get("label", "unknown"),
            label_id=meta.get("label_id", -1),
            feature_vector=result["hog_features"]["feature_vector"],
            subject_id=meta.get("subject_id"),
            scenario=meta.get("scenario"),
            frame_index=meta.get("frame_index"),
            processing_time_ms=result["processing_time_ms"],
        )

    return result


@app.post("/api/extract-features")
async def extract_features_batch(req: BatchExtractRequest):
    """Run batch feature extraction on indexed dataset samples."""
    samples = dataset_loader.get_samples(label=req.label_filter, limit=req.max_samples)
    if not samples:
        return {"status": "warning", "message": "No dataset samples matched filter"}

    success_count = 0
    fail_count = 0

    for s in samples:
        img_path = Path(s["file_path"])
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        res = pipeline.process_frame(img, include_visualizations=False)
        if res["status"] == "success":
            feature_exporter.add_sample(
                sample_id=s.get("frame_index") or s.get("source_file"),
                source_file=s.get("source_file"),
                label=s.get("label"),
                label_id=s.get("label_id"),
                feature_vector=res["hog_features"]["feature_vector"],
                subject_id=s.get("subject_id"),
                scenario=s.get("scenario"),
                frame_index=s.get("frame_index"),
                processing_time_ms=res["processing_time_ms"],
            )
            success_count += 1
        else:
            feature_exporter.add_failure(
                sample_id=s.get("frame_index") or s.get("source_file"),
                source_file=s.get("source_file"),
                label=s.get("label"),
                reason=res.get("message", "Pipeline failure"),
                subject_id=s.get("subject_id"),
            )
            fail_count += 1

    # Auto-export CSV files
    export_res = feature_exporter.export_csv()
    return {
        "status": "success",
        "processed_samples": success_count + fail_count,
        "success_count": success_count,
        "fail_count": fail_count,
        "export_details": export_res,
    }


@app.post("/api/feature-export")
async def trigger_feature_export():
    """Trigger disk export of all buffered features into CSV and JSON reports."""
    # If no records in exporter yet, run a small automatic batch of 20 samples (10 drowsy, 10 notdrowsy)
    if not feature_exporter.records:
        samples = dataset_loader.get_samples(limit=20)
        for s in samples:
            p = Path(s["file_path"])
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            res = pipeline.process_frame(img, include_visualizations=False)
            if res["status"] == "success":
                feature_exporter.add_sample(
                    sample_id=s.get("frame_index") or s.get("source_file"),
                    source_file=s.get("source_file"),
                    label=s.get("label"),
                    label_id=s.get("label_id"),
                    feature_vector=res["hog_features"]["feature_vector"],
                    subject_id=s.get("subject_id"),
                    scenario=s.get("scenario"),
                    frame_index=s.get("frame_index"),
                    processing_time_ms=res["processing_time_ms"],
                )

    export_res = feature_exporter.export_csv()
    return export_res


if __name__ == "__main__":
    import uvicorn
    # Initial scan of dataset on startup
    print("DriveGuard: Indexing initial dataset samples...")
    dataset_loader.scan()
    print("DriveGuard: Dataset indexing complete. Launching FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
