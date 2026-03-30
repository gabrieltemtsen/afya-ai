FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (always invalidates cache on code changes)
COPY backend/ ./backend/
COPY frontend/ ./frontend/
RUN echo "build=$(date -u +%Y%m%d%H%M%S)" > /app/.buildinfo

ENV PORT=8000
EXPOSE 8000

CMD ["python", "backend/main.py"]
