FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .
EXPOSE 8000
CMD ["python", "-m", "wire"]
