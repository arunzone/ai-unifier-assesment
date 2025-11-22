# Python Docker Development Environment

A simple, reproducible Docker-based Python 3.12 development environment.

## Why Docker?

- **Host stays clean**: No Python installations or pip packages on your machine
- **Reproducible**: Everyone on the team gets the exact same environment
- **Isolated**: Project dependencies don't conflict with other projects

## Quick Start

### 1. Build the development container

```bash
docker compose -f docker-compose.dev.yml build app
```

### 2. Open a shell in the container

```bash
docker compose -f docker-compose.dev.yml run --rm app bash
```

## Development Commands

All commands are run inside the container shell.

### Run all checks (lint + type check + tests)
  tox

### Format code
  tox -e format

### Run tests only
  tox -e test

### Build package
  tox -e build


## Project Structure

```
.
├── src/
│   └── app.py              # Main application code
├── tests/
│   └── test_app.py         # Unit tests
├── Dockerfile.dev          # Development container definition
├── docker-compose.dev.yml  # Docker Compose configuration
├── requirements.txt        # Python dependencies
├── .dockerignore           # Files excluded from Docker builds
└── README.md               # This file
```

## Adding Dependencies

1. Add packages to `setup.cfg`
2. Rebuild the container: `docker compose -f docker-compose.dev.yml build app`

## How It Works

- **Volume mount**: Your local code is mounted into `/app` in the container
- **Live edits**: Changes you make on your host are immediately available in the container
- **No host pollution**: All Python packages live inside the container only
