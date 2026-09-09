# Bonus: containerization. Build with:
#   docker build -t ppe-detection-api .
# Run with:
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models ppe-detection-api

FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV/torch image ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
