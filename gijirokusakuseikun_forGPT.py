import os
import openai
import streamlit as st
import io
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.effects import normalize, strip_silence
import numpy as np
import math
from tempfile import NamedTemporaryFile

st.title("議事録作成くん ver0.5")

openai_api_key = st.text_input("Enter OpenAI API Key:")
openai.api_key = openai_api_key

uploaded_file_obj = st.file_uploader("Choose an audio file", type=["m4a", "mp3", "webm", "mp4", "mpga", "wav", "mpeg"], key="audio_file")
if uploaded_file_obj:
    file_ext = uploaded_file_obj.name.split(".")[-1]  # ファイルの拡張子を取得
else:
    file_ext = None

custom_prompt = st.text_input("Enter custom prompt (leave empty to use default):")
transcription = None


def transcribe_audio(uploaded_file_obj):
    if uploaded_file_obj is None:
        raise ValueError("No file has been uploaded.")

    audio_data = uploaded_file_obj.getvalue()
    file_ext = uploaded_file_obj.name.split(".")[-1]

    file_ext = file_ext.lower()

    if file_ext not in ["m4a", "mp3", "webm", "mp4", "mpga", "wav", "mpeg"]:
        raise ValueError(f"Unsupported file format: {file_ext}")

    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=file_ext)
    audio_length_sec = len(audio_segment) / 1000
    file_size = len(audio_data)

    TARGET_FILE_SIZE = 25_000_000

    if file_size > TARGET_FILE_SIZE:
        target_kbps = int(math.floor(TARGET_FILE_SIZE * 8 / audio_length_sec / 1000 * 0.95))

        if target_kbps < 8:
            raise ValueError(f"{target_kbps=} is not supported.")

    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

    if audio_segment.duration_seconds * audio_segment.frame_width > TARGET_FILE_SIZE:
        chunk_length_ms = int(TARGET_FILE_SIZE / audio_segment.frame_width * 1000)
        audio_chunks = make_chunks(audio_segment, chunk_length_ms)
    else:
        audio_chunks = [audio_segment]

    transcriptions = []
    for chunk in audio_chunks:
        with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            if file_size > TARGET_FILE_SIZE:
                print(f"Exporting chunk with bitrate={target_kbps}k")
                chunk.export(temp_file.name, format="mp3", bitrate=f"{target_kbps}k")
            else:
                print(f"Exporting chunk without changing bitrate")
                chunk.export(temp_file.name, format="mp3")
            temp_file.flush()

            # Noise reduction
            # ノイズゲート処理を適用
            silence_threshold = -50.0  # 無音とみなすしきい値を設定（適宜調整してください）
            padding_duration = 100  # パディングの期間（ミリ秒）
            noise_reduced_chunk = chunk.strip_silence(silence_thresh=silence_threshold, padding=padding_duration)

            with open(temp_file.name, "wb") as file_obj:
                noise_reduced_chunk.export(file_obj, format="mp3")
                file_obj.flush()

                with open(file_obj.name, "rb") as file_obj:
                    transcript = openai.Audio.transcribe(model="whisper-1", language='ja', file=file_obj)

            os.unlink(temp_file.name)
            transcriptions.append(transcript.text)

    transcription = " ".join(transcriptions)
    return transcription



def summarize_text(transcription, custom_prompt, max_tokens=4096):
    if not custom_prompt:
        custom_prompt = f"以下の文章を200字程度で要約してください。出力用フォーマットは、「要約」「ポジティブな意見」「ネガティブな意見」「ネクストアクション」とし、要約以外は箇条書きで記載してください。"

    # 1. 与えられたトランスクリプションを、GPT-3.5-turbo モデルが処理できるサイズのチャンクに分割
    text_chunks = []
    start = 0
    while start < len(transcription):
        end = start + max_tokens
        if end > len(transcription):
            end = len(transcription)
        text_chunks.append(transcription[start:end])   
        start = end

    # 2. 分割された各チャンクに対して、MECEに基づく要約を行う
    summarized_chunks = []
    for chunk in text_chunks:
        chunk_prompt = f"このテキストをできるだけ重要な箇所を漏らさないように短くしてください: {chunk}"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは優秀なアシスタントです。MECEの思考法での要約が得意です"},
                {"role": "user", "content": chunk_prompt},
            ],
        )
        summarized_chunk = response["choices"][0]["message"]["content"]
        summarized_chunks.append(summarized_chunk)

    # 3. 次に各チャンクごとの要約をまとめ、一つの文章に整える
    summarized_text = " ".join(summarized_chunks)

    # 4. まとめられた文章をカスタムプロンプトまたは基本プロンプトに則り要約を行う
    final_prompt = f"{custom_prompt}: {summarized_text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "あなたは優秀なアシスタントです。MECEの思考法での要約が得意です"},
            {"role": "user", "content": final_prompt},
        ],
    )
    final_summary = response["choices"][0]["message"]["content"]

    return final_summary


if st.button("Summarize"):
    if uploaded_file_obj is not None and openai_api_key:
        st.write("Transcribing audio...")
        transcription = transcribe_audio(uploaded_file_obj)
        st.write("Transcription:")
        st.write(transcription)

        st.write("Summarizing text...")
        summary = summarize_text(transcription, custom_prompt)
        st.write("Summary:")
        st.write(summary)
    else:
        st.error("Please upload an audio file and enter your OpenAI API key.")
