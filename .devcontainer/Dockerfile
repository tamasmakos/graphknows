# .devcontainer/Dockerfile
FROM continuumio/miniconda3:latest

# Update existing packages and install fundamental build tools
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y build-essential curl git

# Create Python 3.11 environment and install major packages via conda
# This ensures proper dependency resolution for complex packages
RUN conda create -n py311 python=3.11 -y && \
    conda install -n py311 -c conda-forge -c pytorch --yes \
        numpy scipy pandas scikit-learn \
        pytorch torchvision torchaudio cpuonly \
        openai anthropic \
        pip setuptools wheel

# Activate conda environment for subsequent commands
SHELL ["conda", "run", "-n", "py311", "/bin/bash", "-c"]

# Set working directory before copying files
WORKDIR /workspaces/kg

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install remaining packages from requirements.txt using pip
# Use legacy resolver to handle dependency conflicts more leniently
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --use-deprecated=legacy-resolver -r /requirements.txt

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"

# Validate environment
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print('PyTorch installation successful')"

# Set the default conda environment and configure shell
RUN echo "conda activate py311" >> ~/.bashrc && \
    echo "conda activate py311" >> /etc/bash.bashrc && \
    echo "export PATH=/opt/conda/envs/py311/bin:$PATH" >> ~/.bashrc && \
    echo "export PATH=/opt/conda/envs/py311/bin:$PATH" >> /etc/bash.bashrc

# Set the default shell to use conda
ENV SHELL=/bin/bash
ENV CONDA_DEFAULT_ENV=py311
ENV PATH=/opt/conda/envs/py311/bin:$PATH
