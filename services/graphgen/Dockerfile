FROM python:3.11-slim
# Install system build deps for science libs
RUN apt-get update && apt-get install -y build-essential curl git
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords');"
COPY src/ ./src
ENTRYPOINT ["python", "src/kg_main.py"]
