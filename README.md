# Potato Disease Detector

**A production-ready, portfolio-grade app for detecting potato leaf diseases.**

**Author:** Monkey D. Luffy

## Highlights
- Transfer learning using **MobileNetV2** (TensorFlow / Keras)
- FastAPI backend with `/predict`, `/health`, `/version`
- React frontend (Tailwind-ready) with file preview and results
- Docker & Docker Compose for local dev; ready for Render deployment
- GitHub Actions CI scaffold for lint/build/test
- MIT License — fully original codebase

## Quickstart (local, Docker Compose)
1. Copy or train a model: `training/train.py` produces `api/models/best_model.h5`.
2. Build and run:
```bash
docker compose up --build
```
3. Open frontend: http://localhost:3000 (proxy to API at 8000)

## Training & Notebook
See `training/potato_disease_training.ipynb` and `training/train.py` for transfer-learning code using MobileNetV2.

## Deployment
The project includes `docker-compose.yml`. To deploy to Render, create two services (web for API, static for frontend) using the provided Dockerfiles or use Render's Docker support.

## Structure
- `training/` — training script, notebook, requirements
- `api/` — FastAPI app, models, Dockerfile
- `frontend/` — React app with Tailwind-ready config
- `.github/workflows/` — CI workflow

## Credits
Original idea inspired by public potato disease projects. This implementation is an original rewrite by the author.

# potato_disease_detector
