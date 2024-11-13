import requests
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import time
import socket
import requests

# Force requests to use IPv4
original_getaddrinfo = socket.getaddrinfo

def ipv4_getaddrinfo(*args, **kwargs):
    return [info for info in original_getaddrinfo(*args, **kwargs) if info[0] == socket.AF_INET]

socket.getaddrinfo = ipv4_getaddrinfo

SUMMARY_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
QA_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
summary_headers = {"Authorization": "Bearer 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"}
qa_headers = {"Authorization": "Bearer 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"}

def query(api_url, headers, payload):
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

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
        if output and isinstance(output, list) and len(output) > 0:
            summaries.append(output[0]['summary_text'])
        else:
            st.warning(f"Unexpected API response for chunk {i+1}")
        progress_bar.progress((i + 1) / total_chunks)
        progress_text.text(f"Progress: {int((i + 1) / total_chunks * 100)}%")

    progress_bar.empty()
    progress_text.empty()
    elapsed_time = time.time() - start_time
    return ' '.join(summaries), elapsed_time

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            p = parse_qs(parsed_url.query)
            return p['v'][0]
        if parsed_url.path[:7] == '/embed/':
            return parsed_url.path.split('/')[2]
        if parsed_url.path[:3] == '/v/':
            return parsed_url.path.split('/')[2]
    return None

from youtube_transcript_api._errors import TooManyRequests
import time

def get_transcript(url, retries=3, wait_time=5):
    video_id = extract_video_id(url)
    if not video_id:
        return "Error: Could not extract video ID from URL", None

    for attempt in range(retries):
        try:
            transcript_data = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(['en']).fetch()
            return " ".join(entry['text'] for entry in transcript_data), video_id
        except TooManyRequests:
            if attempt < retries - 1:
                time.sleep(wait_time)  # Wait before retrying
                wait_time *= 2  # Exponential backoff
            else:
                return "Error: Rate limit exceeded. Please try again later.", None
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
            if summary:
                st.success("Summary generated successfully!")
                st.subheader("Summary")
                st.text_area("Video Summary", value=summary, height=300, max_chars=5000)
                st.write(f"Time taken to generate summary: {elapsed_time:.2f} seconds")
                st.download_button("Download Summary", data=summary, file_name=f"summary_{video_id}.txt", mime="text/plain")
            else:
                st.error("Failed to generate summary. Please try again.")

        elif choice == 'Ask Questions':
            st.subheader("Ask Questions About the Video")
            question = st.text_input("Enter your question:")

            if st.button("Get Answer"):
                if question:
                    output = query(QA_API_URL, qa_headers, {
                        "inputs": {"question": question, "context": transcript}
                    })
                    if output:
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
                        st.error("Failed to get an answer. Please try again.")
                else:
                    st.write("Please enter a question.")
    else:
        st.error(transcript)  # Display error if transcript extraction fails
else:
    st.write("Please enter a valid YouTube URL.")

# Add a button to test API connection
if st.sidebar.button("Test API Connection"):
    test_response = query(SUMMARY_API_URL, summary_headers, {"inputs": "Test connection"})
    if test_response:
        st.sidebar.success("API connection successful!")
    else:
        st.sidebar.error("API connection failed. Check your network connection.")