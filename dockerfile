FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# FFmpegとffprobeのインストール
RUN apt-get update && \
    apt-get install -y ffmpeg

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "gijirokusakuseikun_forGPT.py"]
