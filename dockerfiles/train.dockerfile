# FROM sets the base image (in this case, some ubuntu or arch installation with python 3.12 and uv)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# RUN just runs commands
# so we just install gcc and build-essential
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# COPY copies files from host into the image
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY data/ data/

# WORKDIR sets the working dir to whatever
# here, we set it to the root dir of this project
WORKDIR /

# run minimal uv sync, using the exact version from the lock, without caching the packages, and installs only the packages, not the project itself
RUN uv sync --locked --no-cache --no-install-project

# ENTRYPOINT defines what command is run AFTER BUILDING. I.e. when running "docker run <image>"
# in this case: uv run src/mlops_proj/train.py
ENTRYPOINT ["uv", "run", "src/mlops_proj/train.py"]