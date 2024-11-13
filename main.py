import requests
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import time

SUMMARY_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
QA_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
summary_headers = {"Authorization": "Bearer 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"}
qa_headers = {"Authorization": "Bearer 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"}

def query(api_url, headers, payload):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

def chunk_text(text, max_length=512):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

def generate_summary(text):
    summaries = []
    total_chunks = len(list(chunk_text(text)))
    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_time = time.time()

    for i, chunk in enumerate(chunk_text(text)):
        chunk_length = len(chunk.split())
        min_len, max_len = (1, 60) if chunk_length <= 150 else (70, 130) if chunk_length <= 300 else (130, 280)

        output = query(SUMMARY_API_URL, summary_headers, {
            "inputs": chunk,
            "parameters": {"min_length": min_len, "max_length": max_len}
        })
        summaries.append(output[0]['summary_text'])
        progress_bar.progress((i + 1) / total_chunks)
        progress_text.text(f"Progress: {int((i + 1) / total_chunks * 100)}%")

    progress_bar.empty()
    progress_text.empty()
    elapsed_time = time.time() - start_time
    return ' '.join(summaries), elapsed_time

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.query:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    return parsed_url.path.split('/')[-1] if '/' in parsed_url.path else None

def get_transcript(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return "Error: Could not extract video ID from URL", None
        transcript_data = YouTubeTranscriptApi.list_transcripts(video_id).find_manually_created_transcript(['en']).fetch()
        return " ".join(entry['text'] for entry in transcript_data), video_id
    except Exception as e:
        return f"Error: {str(e)}", None

# Streamlit Interface with Gradient Title
st.markdown(
    """
    <h1 style="background: -webkit-linear-gradient(#a0eaff, #85d1e5); -webkit-background-clip: text; color: transparent;">
        TubeIntel: Summarize Video and Ask Questions about the Video
    </h1>
    """,
    unsafe_allow_html=True
)

video_url = st.text_input("YouTube Video URL")
choice = st.radio("Choose an action:", ('Generate Summary', 'Ask Questions'))

if video_url:
    transcript, video_id = get_transcript(video_url)
    if video_id:
        if choice == 'Generate Summary':
            st.write("Generating summary...")
            summary, elapsed_time = generate_summary(transcript)
            st.success("Summary generated successfully!")
            st.subheader("Summary")
            st.text_area("Video Summary", value=summary, height=300, max_chars=5000)
            st.write(f"Time taken to generate summary: {elapsed_time:.2f} seconds")
            st.download_button("Download Summary", data=summary, file_name=f"summary_{video_id}.txt", mime="text/plain")

        elif choice == 'Ask Questions':
            st.subheader("Ask Questions About the Video")
            question = st.text_input("Enter your question:")

            if st.button("Get Answer"):
                if question:
                    output = query(QA_API_URL, qa_headers, {
                        "inputs": {"question": question, "context": transcript}
                    })
                    answer = output.get("answer", "No answer found.")
                    confidence = output.get("score", 0) * 100  # Convert to percentage

                    # Confidence color-coding
                    confidence_color = "green" if confidence > 70 else "yellow" if confidence >= 50 else "red"
                    # Display answer and confidence with color-coded confidence
                    st.write(f"Answer: {answer}")
                    st.markdown(
                        f"<span style='color:{confidence_color}; font-weight:bold;'>Confidence: {confidence:.2f}%</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.write("Please enter a question.")
    else:
        st.error(transcript)  # Display error if transcript extraction fails
else:
    st.write("Please enter a valid YouTube URL.")