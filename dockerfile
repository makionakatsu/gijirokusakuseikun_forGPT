# ベースイメージを選択
FROM python:3.9

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係をインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# ffmpegとffprobeをインストール
RUN apt-get update && apt-get install -y ffmpeg

# アプリケーションコードをコピー
COPY . .

# Streamlitを実行
CMD ["streamlit", "run", "gijirokusakuseikun_forGPT.py"]
