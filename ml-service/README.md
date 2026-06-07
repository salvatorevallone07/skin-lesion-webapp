# ML inference service (Python / FastAPI)

Reuses the exact computer-vision pipeline from the original repository
(preprocess -> segment -> 20 hand-crafted features -> MLP classifier).

## Why this exists
The original model is NOT an image-to-class CNN. Inference requires OpenCV + PyTorch,
so it cannot run natively inside the Java/Tomcat backend. This service wraps the
pipeline behind a small HTTP API that Spring calls.

## Endpoints
- `GET  /health`  – liveness + whether a trained model is loaded
- `GET  /info`    – model metadata
- `POST /predict` – multipart field `file` (image) -> JSON result

## Training / exporting the model
The repository ships WITHOUT trained weights. Generate them in one of two ways.

### Option B – manual download (no Kaggle API key)
Download the dataset from your browser (you only need to be signed into a free
Kaggle account) at
<https://www.kaggle.com/datasets/andrewmvd/isic-2019>, extract the archive, then
mount the folder and point `train.py` at it:

```bash
docker compose run --rm \
  -v /absolute/path/to/isic-2019:/data/isic-2019 \
  ml-service python train.py --data-dir /data/isic-2019 --epochs 100
```

The folder must contain `ISIC_2019_Training_Input/ISIC_2019_Training_Input/*.jpg`
and `ISIC_2019_Training_GroundTruth.csv`.

### Option A – automatic download via kagglehub (needs API key)
```bash
docker compose run --rm \
  -e KAGGLE_USERNAME=your_user -e KAGGLE_KEY=your_key \
  ml-service python train.py --epochs 100
```

Either way the script writes `model/model.pt` + `model/normalization.json`.
Restart the service and it loads them.

> Without these artifacts the service still answers `/predict` using an UNTRAINED
> model so the whole stack is demonstrable, but predictions are flagged
> `model_trained: false` and are NOT meaningful.
