# Cardio AI — API image. Serves the FastAPI backend (api/main.py).
# Note: this image expects models/ to already contain trained artifacts —
# either bake them in via `COPY models/ ./models/` (as done below, since
# they're committed to the repo — see README) or mount a volume at runtime.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
