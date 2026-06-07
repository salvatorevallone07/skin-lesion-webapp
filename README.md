# Skin lesion classification web app

Single-page web app to classify a dermatoscopic image as **melanocytic nevus** or
**melanoma**, wrapping the computer-vision model from
[Project-computer-vision---skin-lesion-images](https://github.com/salvatorevallone07/Project-computer-vision---skin-lesion-images).

## Architecture

Three containers orchestrated by Docker Compose:

| Service      | Stack                                   | Role |
|--------------|-----------------------------------------|------|
| `tomcat`     | Java 25 · Spring Boot 4 · Tomcat 11     | REST API + serves the Angular SPA (WAR at context root) |
| `postgres`   | PostgreSQL 17                           | Stores analyses + extracted features (image bytes on a volume, only the path in the DB) |
| `ml-service` | Python · FastAPI · OpenCV · PyTorch     | Runs the original pipeline and returns the prediction |

```
Browser ──► Tomcat (Spring + Angular) ──HTTP──► ml-service (FastAPI)
                  │                                   │
                  ▼                                   ▼
              Postgres                         pipeline + MLP
            (metadata)                      (preprocess→segment→
        + image on volume                    20 features→classify)
```

### Why a third (Python) container?
The original model is **not** an image-to-class CNN. Inference requires the exact
OpenCV + PyTorch pipeline (denoise → Otsu segmentation → 5 shape + 6 color + 9
texture features → z-score normalization → MLP 20→128→64→32→1). Re-implementing
this in Java would risk diverging from the trained feature distribution, so the
pipeline runs unchanged in a small Python service that Spring calls over HTTP.

## Data model
- `ml_model` — metadata of each model version (framework, input_dim=20, threshold, accuracy).
- `analysis` — one row per upload + inference (predicted class, probability, status, latency, image path).
- `analysis_feature` — the 20 extracted features per analysis (name, value, index).

See `db/schema.sql` for the reference DDL (created automatically by Hibernate at runtime).

## Quick start

```bash
cp .env.example .env          # optionally edit credentials
docker compose up --build
```

Then open <http://localhost:8080>.

> On first run no trained weights exist, so the `ml-service` answers with an
> **untrained** model and the UI shows a warning. The full stack works end-to-end;
> predictions become meaningful only after training (below).

## Training the model (generate real weights)

The original repository ships without saved weights. Generate them once.

**Manual download (no Kaggle API key)** — download from your browser at
<https://www.kaggle.com/datasets/andrewmvd/isic-2019> (a free Kaggle login is
enough), extract it, then mount the folder:

```bash
docker compose run --rm \
  -v /absolute/path/to/isic-2019:/data/isic-2019 \
  ml-service python train.py --data-dir /data/isic-2019 --epochs 100

docker compose restart ml-service
```

**Automatic download (needs Kaggle API key)**:

```bash
docker compose run --rm \
  -e KAGGLE_USERNAME=your_user -e KAGGLE_KEY=your_key \
  ml-service python train.py --epochs 100
```

Either way the script writes `ml-service/model/model.pt` and
`ml-service/model/normalization.json` (persisted on a volume) which the service
loads automatically.

## API

| Method | Path                          | Description |
|--------|-------------------------------|-------------|
| POST   | `/api/analyses`               | multipart `file` → run inference, persist, return result |
| GET    | `/api/analyses/{id}`          | fetch a stored analysis |
| GET    | `/api/analyses/{id}/image`    | stream the uploaded image |

## Local development (without Docker)

- **ml-service**: `pip install -r ml-service/requirements.txt && uvicorn app:app --reload` (from `ml-service/`)
- **backend**: `cd backend && mvn spring-boot:run` (needs JDK 25, a reachable Postgres and ml-service)
- **frontend**: `cd frontend && npm install && npm start` (Angular dev server on :4200, proxies `/api` to :8080)

## Version notes
- Java 25 + Spring Boot 4.0 are required because Tomcat 11 implements Servlet 6.1 /
  Jakarta EE 11 (Spring Boot 3.5 only supports Tomcat 10.1).
- Angular 21 requires Node.js ≥ 22.22 or ≥ 24.13 (handled by the `node:24` build image).
- If a Docker base-image tag is unavailable in your registry mirror (e.g.
  `tomcat:11.0-jdk25-temurin` or `maven:3.9-eclipse-temurin-25`), substitute the
  closest published tag in `backend/Dockerfile`.