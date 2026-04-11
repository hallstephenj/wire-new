FROM python:3.12-slim
WORKDIR /app

# Install dependencies first (cached layer — only re-runs when pyproject.toml changes)
COPY pyproject.toml .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "fastapi>=0.115.0" "uvicorn[standard]>=0.34.0" \
        "feedparser>=6.0.0" "httpx>=0.28.0" "pyyaml>=6.0" "apscheduler>=3.10.0" \
        "scikit-learn>=1.6.0" "anthropic>=0.43.0" "python-dotenv>=1.0.0" \
        "jinja2>=3.1.0" "beautifulsoup4>=4.12.0" "sentence-transformers>=3.0.0" \
        "supabase>=2.0.0" "python-jose[cryptography]>=3.3.0" "stripe>=7.0.0"

# Copy code and install the package itself (no deps, already installed above)
COPY . .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000
CMD ["python", "-m", "wire"]
