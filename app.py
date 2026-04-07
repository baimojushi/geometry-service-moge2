import io
import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

from moge.model.v2 import MoGeModel


PORT = int(os.getenv("PORT", "8000"))
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_NAME = os.getenv("MODEL_NAME", "Ruicheng/moge-2-vitb-normal")
MODEL_DIR = os.getenv("MODEL_DIR", "").strip()
JOBS_DIR = Path(os.getenv("JOBS_DIR", "/app/data/jobs"))
AUTO_DOWNLOAD_ON_START = os.getenv("AUTO_DOWNLOAD_ON_START", "false").lower() == "true"

app = FastAPI(title="moge2-geometry-service", version="1.0.0")
model = None


class AnalyzeRequest(BaseModel):
    image_url: str
    max_side: int = Field(default=768, ge=256, le=1536)
    resolution_level: int = Field(default=5, ge=0, le=9)


def resize_keep_ratio(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def save_depth_preview(depth: np.ndarray, out_path: Path) -> None:
    valid = np.isfinite(depth)
    if not valid.any():
        preview = np.zeros_like(depth, dtype=np.uint8)
    else:
        d = depth.copy()
        d[~valid] = np.nan
        lo = np.nanpercentile(d, 2)
        hi = np.nanpercentile(d, 98)
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((d - lo) / (hi - lo), 0, 1)
        norm = np.nan_to_num(norm, nan=0.0)
        preview = (norm * 255).astype(np.uint8)
    Image.fromarray(preview).save(out_path)


def save_normal_preview(normal: np.ndarray, out_path: Path) -> None:
    vis = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(vis).save(out_path)


def save_mask_preview(mask: np.ndarray, out_path: Path) -> None:
    vis = mask.astype(np.uint8) * 255
    Image.fromarray(vis).save(out_path)


def load_model() -> MoGeModel:
    target = MODEL_DIR if MODEL_DIR else MODEL_NAME
    if MODEL_DIR and not Path(MODEL_DIR).exists():
        if AUTO_DOWNLOAD_ON_START:
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        else:
            raise RuntimeError(f"MODEL_DIR does not exist: {MODEL_DIR}")
    model_instance = MoGeModel.from_pretrained(target).to(torch.device(DEVICE))
    model_instance.eval()
    return model_instance


@app.on_event("startup")
def startup_event():
    global model
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "device": DEVICE,
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR or None,
    }


@app.get("/readyz")
def readyz():
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"ready": True}


@app.get("/v1/meta")
def meta():
    return {
        "service": "moge2-geometry-service",
        "device": DEVICE,
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR or None,
        "jobs_dir": str(JOBS_DIR),
    }


def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    np_img = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).permute(2, 0, 1).to(torch.device(DEVICE))


def _build_result(job_id: str, img: Image.Image, intrinsics: np.ndarray, normal_exists: bool):
    result = {
        "job_id": job_id,
        "image_size": {"width": img.size[0], "height": img.size[1]},
        "files": {
            "depth_npy": f"/artifacts/{job_id}/depth.npy",
            "mask_npy": f"/artifacts/{job_id}/mask.npy",
            "intrinsics_npy": f"/artifacts/{job_id}/intrinsics.npy",
            "depth_preview": f"/artifacts/{job_id}/depth_preview.png",
            "mask_preview": f"/artifacts/{job_id}/mask_preview.png",
        },
        "intrinsics": intrinsics.tolist(),
    }
    if normal_exists:
        result["files"]["normal_npy"] = f"/artifacts/{job_id}/normal.npy"
        result["files"]["normal_preview"] = f"/artifacts/{job_id}/normal_preview.png"
    return result


def _persist_output(img: Image.Image, output: dict) -> dict:
    depth = output["depth"].detach().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy()
    intrinsics = output["intrinsics"].detach().cpu().numpy()
    normal = output["normal"].detach().cpu().numpy() if "normal" in output else None

    job_id = f"geo_{uuid.uuid4().hex[:12]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    np.save(job_dir / "depth.npy", depth)
    np.save(job_dir / "mask.npy", mask)
    np.save(job_dir / "intrinsics.npy", intrinsics)

    save_depth_preview(depth, job_dir / "depth_preview.png")
    save_mask_preview(mask, job_dir / "mask_preview.png")

    if normal is not None:
        np.save(job_dir / "normal.npy", normal)
        save_normal_preview(normal, job_dir / "normal_preview.png")

    return _build_result(job_id, img, intrinsics, normal is not None)


def _run_inference(img: Image.Image, resolution_level: int) -> dict:
    tensor = _image_to_tensor(img)
    with torch.inference_mode():
        output = model.infer(tensor, resolution_level=resolution_level)
    return _persist_output(img, output)


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest):
    try:
        resp = requests.get(req.image_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"download failed: {e}")

    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    img = resize_keep_ratio(img, req.max_side)
    return _run_inference(img, req.resolution_level)


@app.post("/v1/analyze-upload")
async def analyze_upload(
    file: UploadFile = File(...),
    max_side: int = 768,
    resolution_level: int = 5,
):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    img = resize_keep_ratio(img, max_side)
    return _run_inference(img, resolution_level)


@app.get("/artifacts/{job_id}/{filename}")
def get_artifact(job_id: str, filename: str):
    path = JOBS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")

    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".npy"):
        media_type = "application/octet-stream"
    else:
        media_type = "application/octet-stream"

    return FileResponse(str(path), media_type=media_type, filename=filename)
