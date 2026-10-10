FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run API
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
