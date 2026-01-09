## Knowledge Graph Project – Development Environment

This repository is set up to be fully reproducible via a Dev Container (Docker + Conda) and does not require you to manage Python or system dependencies manually on your host.

### Prerequisites

- **Docker** installed and running
- **Dev Containers / Remote - Containers** extension (or equivalent)

### Recommended: Open in Dev Container

1. Clone the repository:

```bash
git clone <YOUR-REPO-URL> kg
cd kg
```

2. Open the folder in VS Code or Cursor.
3. When prompted, choose **“Reopen in Container”** (or run the command: **Dev Containers: Rebuild and Reopen in Container**).

This will:

- Build the image using `.devcontainer/Dockerfile` (based on `continuumio/miniconda3`)
- Create and activate the Conda environment `py311`
- Install Python dependencies from `requirements.txt` (inside the container)
- Download spaCy model `en_core_web_lg` and required NLTK data
- Set the working directory to `/workspaces/kg`

Once the container is ready, your terminal inside the editor will automatically use the `py311` environment.

### Manual Docker Usage (without Dev Containers extension)

From the repository root:

```bash
docker build -t kg-dev -f .devcontainer/Dockerfile .
docker run --gpus=all --runtime=nvidia \
  -v "$PWD":/workspaces/kg \
  -w /workspaces/kg \
  -it kg-dev
```

Inside the container, the `py311` Conda environment is already activated and all dependencies are installed.

### Running the Project

From within the Dev Container (or inside the running Docker container), run your usual commands, for example:

```bash
python -m src.app
```

Replace this with the actual entrypoint you use (e.g. API server, scripts, or notebooks).


