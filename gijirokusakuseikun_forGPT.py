import os
from openai import OpenAI
import streamlit as st
import io
from pydub import AudioSegment
from pydub.utils import make_chunks
import math
from tempfile import NamedTemporaryFile

st.title("MinuteGenerator")

# OpenAI API Keyの入力
openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")  # type 引数を "password" に設定して目隠し
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    st.error("Please enter your OpenAI API key.")

# 言語の選択肢
language_options = {"Japanese": "ja", "English": "en", "Chinese": "zh"}
selected_language = st.selectbox("Choose a language:", options=list(language_options.keys()))

# 音声ファイルのアップロード
uploaded_file_obj = st.file_uploader("Choose an audio file", type=["m4a", "mp3", "webm", "mp4", "mpga", "wav", "mpeg"], key="audio_file")
if uploaded_file_obj:
    file_ext = uploaded_file_obj.name.split(".")[-1].lower()  # ファイルの拡張子を取得し小文字に変換

# カスタムプロンプトの入力
custom_prompt = st.text_area("Enter custom prompt (leave empty to use default):", height=100)
transcription = None

# 音声ファイルをチャンクごとに処理する関数
def process_audio_chunk(chunk, target_kbps):
    with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        if target_kbps:
            chunk.export(temp_file.name, format="mp3", bitrate=f"{target_kbps}k")
        else:
            chunk.export(temp_file.name, format="mp3")
        temp_file.flush()
        return temp_file.name

# 音声チャンクを転記する関数
def transcribe_chunk(file_path, language):
    with open(file_path, "rb") as file_obj:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=file_obj, language = language)
    os.unlink(file_path)    
    return transcript.text

# 音声ファイルを転記するメイン関数
def transcribe_audio(uploaded_file_obj, language):
    if uploaded_file_obj is None:
        raise ValueError("No file has been uploaded.")

    audio_data = uploaded_file_obj.getvalue()
    file_ext = uploaded_file_obj.name.split(".")[-1].lower()

    if file_ext not in ["m4a", "mp3", "webm", "mp4", "mpga", "wav", "mpeg"]:
        st.error(f"Unsupported file format: {file_ext}")
        return

    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=file_ext)
    audio_length_sec = len(audio_segment) / 1000
    file_size = len(audio_data)
    TARGET_FILE_SIZE = 25_000_000

    if file_size > TARGET_FILE_SIZE:
        target_kbps = int(math.floor(TARGET_FILE_SIZE * 8 / audio_length_sec / 1000 * 0.95))
        if target_kbps < 8:
            st.error(f"{target_kbps=} is not supported.")
            return
    else:
        target_kbps = None

    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    if audio_segment.duration_seconds * audio_segment.frame_width > TARGET_FILE_SIZE:
        chunk_length_ms = int(TARGET_FILE_SIZE / audio_segment.frame_width * 1000)
        audio_chunks = make_chunks(audio_segment, chunk_length_ms)
    else:
        audio_chunks = [audio_segment]

    # プログレスバーを追加
    progress_bar = st.progress(0)

    transcriptions = []
    for i, chunk in enumerate(audio_chunks):
        file_path = process_audio_chunk(chunk, target_kbps)
        transcript = transcribe_chunk(file_path, language)
        transcriptions.append(transcript)
        progress_bar.progress((i + 1) / len(audio_chunks))

    transcription = " ".join(transcriptions)
    return transcription

# テキストを要約する関数
def summarize_text(transcription, custom_prompt, max_tokens=3000):
    if not client.api_key:
        st.error("API key is not set.")
        return

    if not custom_prompt:
        custom_prompt = """
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

    text_chunks = [transcription[i:i + max_tokens] for i in range(0, len(transcription), max_tokens)]
    summarized_chunks = []
    
    # Transcriptionの長さを確認
    st.write("Transcription Length:", len(transcription))

    progress_bar = st.progress(0)
    for i, chunk in enumerate(text_chunks):         
        prompt = f"{custom_prompt}\n\nテキスト: {chunk}"
        completion = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])

        if hasattr(completion.choices[0].message, 'content'):
            content = completion.choices[0].message.content
            summarized_chunks.append(content)
        else:
            st.error("No 'content' in message")
            continue
        
        progress_bar.progress((i + 1) / len(text_chunks))

    # NoneやEmptyをフィルタリングしてから結合
    summarized_chunks = [chunk for chunk in summarized_chunks if chunk]    
    return " ".join(summarized_chunks)

# サマリー生成ボタンの動作
if st.button("Summarize"):
    if uploaded_file_obj and openai_api_key:
        st.write("Transcribing audio...")
        transcription = transcribe_audio(uploaded_file_obj, language_options[selected_language])
        st.write("###Transcription:")
        st.text_area("", transcription, height=200)
        st.download_button("Download Transcription.txt", transcription.encode("utf-8"), "transcription.txt")
        st.write("Summarizing text...")
        summary = summarize_text(transcription, custom_prompt)
        st.write("###Summary:")
        st.text_area("", summary, height=200)
        st.download_button("Download Summary.txt", summary.encode("utf-8"), "summary.txt")
    else:
        st.error("Please upload an audio file and enter your OpenAI API key.")

# フィードバック用リンク
twitter_url = "https://twitter.com/makio_study"
tw_text = "Send feedback"
st.markdown(f"[{tw_text}]({twitter_url})", unsafe_allow_html=False)

# Buy Me A CoffeeボタンのHTMLコード
html_code = '<a href="https://www.buymeacoffee.com/makionakatsu" target="_blank">' \
            '<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" ' \
            'alt="Buy Me A Coffee" height="60" width="250"></a>'
st.markdown(html_code, unsafe_allow_html=True)