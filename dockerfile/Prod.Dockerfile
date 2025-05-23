FROM python:3.12-slim-bookworm AS builder

RUN mkdir app
WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN pip install --no-cache-dir poetry \
  && poetry config virtualenvs.in-project true \
  && poetry install --no-root --only main --no-interaction --no-ansi \
  && rm -rf $(poetry config cache-dir)/{cache,artifacts}

FROM python:3.12-slim-bookworm

COPY --from=builder /app /app

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"


COPY src/ ./src/

#CMD ["python", "-m",\
#  "uvicorn",\
#  "src.app:app",\
#  "--host", "0.0.0.0",\
#  "--port", "5000" ]

# или для gradio
 CMD ["python", "src/gradio_app.py" ]

