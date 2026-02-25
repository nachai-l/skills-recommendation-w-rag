FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps (faiss sometimes needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl ca-certificates libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run uses PORT
ENV PORT=8080
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]