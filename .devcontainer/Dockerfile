FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/kg

# Upgrade pip
RUN pip install --upgrade pip

# Install Service Dependencies
# Copy requirements from both services to ensure the environment is complete
COPY services/graphgen/requirements.txt /tmp/pip-tmp/graphgen-reqs.txt
COPY services/graphrag/requirements.txt /tmp/pip-tmp/graphrag-reqs.txt

# Install them (GraphGen first as it usually has heavier deps like torch)
RUN pip install --no-cache-dir -r /tmp/pip-tmp/graphgen-reqs.txt \
    && pip install --no-cache-dir -r /tmp/pip-tmp/graphrag-reqs.txt

# Install Root/Dev Dependencies
COPY requirements.txt /tmp/pip-tmp/dev-reqs.txt
RUN pip install --no-cache-dir -r /tmp/pip-tmp/dev-reqs.txt \
    && rm -rf /tmp/pip-tmp

# Setup NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords');"

# Setup Spacy
RUN python -m spacy download en_core_web_lg