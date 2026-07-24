# Uses a well-established, stable Python tag rather than matching this
# repo's local dev venv version exactly — Docker gives an isolated
# environment regardless, and requirements.txt doesn't depend on
# anything version-specific to 3.14 (what the local .venv uses).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

# Overridden per-service by docker-compose.yml's `command:` — this default
# just makes `docker run` on the image alone do something sensible.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
