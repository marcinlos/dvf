# syntax=docker/dockerfile:1
FROM python:3.12

WORKDIR /code

# Use global environment, not virtual envs
ENV POETRY_VIRTUALENVS_CREATE=false

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install curl
RUN --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && apt-get install -y --no-install-recommends \
    curl

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -

RUN --mount=type=bind,target=.,source=. \
    poetry install

CMD ["/bin/bash"]
