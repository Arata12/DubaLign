FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg mkvtoolnix \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt
COPY sync.py engine.py spectro.py fix.py web.py timeline.py ./
COPY static ./static
ENV MEDIA_DIR=/media OUTPUT_DIR=/output WORK_DIR=/work MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
CMD ["uvicorn", "web:app", "--host", "0.0.0.0", "--port", "7341"]
