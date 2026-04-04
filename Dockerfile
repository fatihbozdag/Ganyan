FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

COPY alembic.ini .
COPY alembic/ alembic/
COPY src/ src/

EXPOSE 5003

CMD ["uv", "run", "python", "-c", "from ganyan.web.app import run; run()"]
