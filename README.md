# MoGe2 Geometry Service

A deployable MoGe-2 inference service based on FastAPI. This package is prepared for Tencent Cloud Super Node style container deployment with a persistent model cache volume and an init-container download step.

## What is included

- `app.py`: FastAPI inference service
- `scripts/download_model.sh`: model preparation script
- `scripts/start.sh`: service startup script
- `deploy/tencent-supernode/*.yaml`: Kubernetes manifests for Tencent Cloud style deployment
- `Dockerfile`: container image definition

## Service features

- Loads `moge.model.v2.MoGeModel`
- Supports remote URL inference through `/v1/analyze`
- Supports direct file upload inference through `/v1/analyze-upload`
- Saves `depth.npy`, `mask.npy`, `intrinsics.npy`
- Saves preview images for depth, mask, and optional normal outputs
- Exposes `/healthz`, `/readyz`, `/v1/meta`

## Environment variables

| Name | Default | Description |
|---|---|---|
| `PORT` | `8000` | Service port |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `MODEL_NAME` | `Ruicheng/moge-2-vitb-normal` | Hugging Face repo for MoGe-2 |
| `MODEL_DIR` | `/models/moge2` | Mounted local model directory |
| `HF_HOME` | `/models/.hf` | Hugging Face cache root |
| `HUGGINGFACE_HUB_CACHE` | `/models/.hf/hub` | Hub cache location |
| `TORCH_HOME` | `/models/.torch` | Torch cache location |
| `JOBS_DIR` | `/app/data/jobs` | Output artifact directory |
| `AUTO_DOWNLOAD_ON_START` | `false` | Enable auto-download inside main container |

## Local run

```bash
docker build -t moge2-geometry-service:local .
docker run --rm -p 8000:8000 \
  -e DEVICE=cpu \
  -e MODEL_NAME=Ruicheng/moge-2-vitb-normal \
  -e AUTO_DOWNLOAD_ON_START=true \
  -v $(pwd)/models:/models \
  moge2-geometry-service:local
```

Open:

```bash
http://127.0.0.1:8000/docs
```

## API examples

### Analyze by image URL

```bash
curl -X POST http://127.0.0.1:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/demo.jpg",
    "max_side": 768,
    "resolution_level": 5
  }'
```

### Analyze by file upload

```bash
curl -X POST "http://127.0.0.1:8000/v1/analyze-upload?max_side=768&resolution_level=5" \
  -F "file=@demo.jpg"
```

### Fetch artifact

```bash
curl -O http://127.0.0.1:8000/artifacts/<job_id>/depth.npy
```

## Recommended deployment flow

1. Build the image.
2. Push it to your image registry.
3. Create namespace and PVC.
4. Deploy the app with the init container enabled.
5. Let the init container prepare the model into the mounted volume.
6. Start the main service container and load the model from the volume.

## Kubernetes apply order

```bash
kubectl apply -f deploy/tencent-supernode/namespace.yaml
kubectl apply -f deploy/tencent-supernode/pvc.yaml
kubectl apply -f deploy/tencent-supernode/deployment.yaml
kubectl apply -f deploy/tencent-supernode/service.yaml
```

Apply ingress when needed:

```bash
kubectl apply -f deploy/tencent-supernode/ingress-example.yaml
```

## Notes for Tencent Cloud Super Node

- Replace `your-registry.example.com/moge2-geometry-service:latest` with your real image address.
- Keep the model volume mounted at `/models`.
- Use a shared storage class such as CFS when you want model cache persistence across pod recreation.
- Keep model preparation in `initContainers` so the main app does not mix download failures with service startup failures.

## Suggested GitHub repo layout

```text
moge2-geometry-service/
├─ app.py
├─ Dockerfile
├─ requirements.txt
├─ README.md
├─ scripts/
│  ├─ download_model.sh
│  └─ start.sh
└─ deploy/
   └─ tencent-supernode/
      ├─ namespace.yaml
      ├─ pvc.yaml
      ├─ deployment.yaml
      ├─ service.yaml
      └─ ingress-example.yaml
```
