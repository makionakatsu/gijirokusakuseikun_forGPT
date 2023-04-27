import os
import openai
import streamlit as st
import io
from pydub import AudioSegment
from pydub.utils import make_chunks
import math
from tempfile import NamedTemporaryFile

st.title("MinuteGenerator ver0.8")

openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")  # type 引数を "password" に設定して目隠し
openai.api_key = openai_api_key


language_options = {"Japanese":"ja", "English":"en", "Chinese": "zh"}
selected_language = st.selectbox("Choose a language:", options=list(language_options.keys()))

uploaded_file_obj = st.file_uploader("Choose an audio file", type=["m4a", "mp3", "webm", "mp4", "mpga", "wav", "mpeg"], key="audio_file")
if uploaded_file_obj:
    file_ext = uploaded_file_obj.name.split(".")[-1]  # ファイルの拡張子を取得
else:
    file_ext = None

custom_prompt = st.text_area("Enter custom prompt (leave empty to use default):", height=100)
transcription = None


def transcribe_audio(uploaded_file_obj, language):
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

    # 1. プログレスバーを追加
    progress_bar = st.progress(0)

    transcriptions = []
    for i, chunk in enumerate(audio_chunks):
        with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            if file_size > TARGET_FILE_SIZE:
                print(f"Exporting chunk with bitrate={target_kbps}k")
                chunk.export(temp_file.name, format="mp3", bitrate=f"{target_kbps}k")
            else:
                print(f"Exporting chunk without changing bitrate")
                chunk.export(temp_file.name, format="mp3")
            temp_file.flush()

            # ノイズゲート処理を適用
            silence_threshold = -50.0  # 無音とみなすしきい値を設定（適宜調整してください）
            padding_duration = 100  # パディングの期間（ミリ秒）
            noise_reduced_chunk = chunk.strip_silence(silence_thresh=silence_threshold, padding=padding_duration)

            with open(temp_file.name, "wb") as file_obj:
                noise_reduced_chunk.export(file_obj, format="mp3")
                file_obj.flush()

                with open(temp_file.name, "rb") as file_obj:
                    transcript = openai.Audio.transcribe(model="whisper-1", language = language, file=file_obj)

            os.unlink(temp_file.name)
            transcriptions.append(transcript.text)
            
            # 2. プログレスバーを更新
            progress_bar.progress((i + 1) / len(audio_chunks))

    transcription = " ".join(transcriptions)
    return transcription


def summarize_text(transcription, custom_prompt, max_tokens=3000):

    # デバッグ用コード
    if openai.api_key is None:
        st.error("API key is not set in the summarize_text function.")
    else:
        st.write("API key is set in the summarize_text function.")

    if not custom_prompt:
        custom_prompt = f"""
        以下のテキストについて、注意事項に則って短い文章で要約して出力してください。
        #️# 注意事項 ##
        ・議題ごとのまとまりで要約する
        ・ダブりや不足のないようにする
        ・重要箇所は重要であることを明示する
        ・5W1Hの内容が含まれる場合は必ず出力に反映する
        ## 出力フォーマット ##
        ・出力はMarkdown方式とする
        ・要約は、200字以内とする。
        ・箇条書きとする。
        ・フォーマットは以下とする。
        ####議題
        ####要約
        ####決定事項(結論)
        ####主な意見
        #####ポジティブな意見
        #####ネガティブな意見
        ####NextAction
        """

    # 1. 与えられたトランスクリプションを、GPT-3.5-turbo モデルが処理できるサイズのチャンクに分割
    st.write(f"Transcription Length: {len(transcription)}")
    st.write(f"Max Tokens: {max_tokens}")
    text_chunks = []
    start = 0
    while start < len(transcription):
        end = start + max_tokens
        if end > len(transcription):
            end = len(transcription)
        text_chunks.append(transcription[start:end])   
        start = end

    st.write(f"Number of Chunks: {len(text_chunks)}")

    # プログレスバーを追加
    progress_bar = st.progress(0)

    # 2. 分割された各チャンクに対して、MECEに基づく要約を行う
    summarized_chunks = []
    for i, chunk in enumerate(text_chunks):
        chunk_prompt = f"""
        以下のテキストを内容を変えないように、できる限りTokenを節約してください。
        5W1Hの情報が入っている場合は、必ず文中に残してください。
        テキスト: {chunk}
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """
                 あなたは、MECEや5W1Hの思考法を用いることが得意です。
                 仕事の要約に使える文章を出力します。
                 """
                },
                {"role": "user", "content": chunk_prompt},
            ],
        )
        summarized_chunk = response["choices"][0]["message"]["content"]
        summarized_chunks.append(summarized_chunk)
        
        # プログレスバーを更新
        progress_bar.progress((i + 1) / (len(text_chunks) + 1))

    # 3. 次に各チャンクごとの要約をまとめ、一つの文章に整える
    summarized_text = " ".join(summarized_chunks)
    
    # 4. まとめられた文章をカスタムプロンプトまたは基本プロンプトに則り要約を行う
    final_prompt = f"{custom_prompt}: {summarized_text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """
             あなたは、MECEや5W1Hの思考法を用いることが得意です。
             仕事の要約に使えるわかりやすい文章を出力します。
             """
            },
            {"role": "user", "content": final_prompt},
        ],
    )
    final_summary = response["choices"][0]["message"]["content"]

    # final_summaryの処理が完了した後にプログレスバーを更新
    progress_bar.progress(1)
    
    return final_summary


if st.button("Summarize"):
    if uploaded_file_obj is not None and openai_api_key:
        st.write("Transcribing audio...")
        transcription = transcribe_audio(uploaded_file_obj, language_options[selected_language])
        st.write("Transcription:")
        st.text_area("", transcription, height=200)
        st.download_button(
            label="Download Transcription.txt",
            data=transcription.encode("utf-8"),
            file_name="transcription.txt",
            mime="text/plain",
        )
        st.write("Summarizing text...")
        summary = summarize_text(transcription, custom_prompt)
        st.write("Summary:")
        st.write(summary)
        st.download_button(
            label="Download Summary .txt",
            data=summary.encode("utf-8"),
            file_name="summary.txt",
            mime="text/plain",
        )
    else:
        st.error("Please upload an audio file and enter your OpenAI API key.")



twitter_url = "https://twitter.com/makio_study"
tw_text ="Send feedback"
st.markdown(f"[{tw_text}]({twitter_url})", unsafe_allow_html=False)


url = "https://www.amazon.jp/hz/wishlist/ls/MZ8DL267UFX5?ref_=wl_share"
button_text = "Supporting indie developers"

st.markdown(f"[{button_text}]({url})", unsafe_allow_html=False)
