FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY alembic.ini .
COPY alembic/ alembic/
RUN uv sync --frozen --no-dev --extra bayes
ENV GANYAN_MODEL_DIR=/app/models FLASK_HOST=0.0.0.0 GANYAN_SKIP_SCHEDULER=1 GANYAN_SKIP_LAUNCH_REFRESH=1
VOLUME ["/app/models"]
EXPOSE 5003
CMD ["uv", "run", "--frozen", "--no-sync", "python", "-c", "from ganyan.web.app import run; run()"]
