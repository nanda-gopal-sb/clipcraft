import os
import tempfile
import streamlit as st


from services.dialouge_search.dialouge_service import DialougeService
from services.face_detection.face_clipper_service import FaceDetectionService
from services.prompt_search.prompt_search_service import PromptService

dialouge_service = DialougeService()
face_service = FaceDetectionService()
prompt_service = PromptService()

def face_clipper():
    string = face_service.health()
    st.title(string)


def video_prompt():
    '''
    string = prompt_service.health()
    st.title(string)
    '''
    st.title("Prompt-Based Clip Extractor")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    prompt = st.text_input("Enter prompt")

    if uploaded_video is not None and prompt:
        with open(uploaded_video.name, "wb") as f:
            f.write(uploaded_video.read())

        if st.button("Extract Clips"):
            with st.spinner("Processing video..."):
                result = prompt_service.run(uploaded_video.name, prompt)

            st.success("Clips extracted successfully!")
            st.write("Top matching scenes:")
            st.write(result)




def dialouge_search():
    st.header("Dialogue Search")

    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "mkv"])
    if uploaded_video:
        video_path = save_uploaded_file(uploaded_video)
        st.session_state["video_path"] = video_path
        st.success(f"File saved: {video_path}")

        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                segments = dialouge_service.dialouge_transcribe(video_path)
                st.session_state["transcript_segments"] = segments
                st.write("Transcription Completed")

    segments = st.session_state.get("transcript_segments", [])
    video_path = st.session_state.get("video_path", None)
    query = st.text_input("Search transcript for", "")

    if st.button("Search"):
        if not segments:
            st.warning("Transcribe first.")
        else:
            matches = dialouge_service.search_transcript(segments, query)
            st.session_state["matched_segments"] = matches
            st.write("Matches:", matches)

    matched_segments = st.session_state.get("matched_segments", [])
    
    if matched_segments and video_path:
        st.subheader("Clip extraction")
        selected_idx = st.selectbox("Select a match to extract", range(len(matched_segments)), 
                                     format_func=lambda i: f"{i}: {matched_segments[i]['text'][:50]}")
        selected_match = matched_segments[selected_idx]
        start = selected_match["start"]
        end = selected_match["end"]
        
        st.write(f"Selected: {selected_match['text']}")
        st.write(f"Start: {start}s, End: {end}s")
        
        out_name = st.text_input("Output filename", "clip.mp4")
        if st.button("Extract Clip"):
            with st.spinner("Extracting clip..."):
                out_path = os.path.join(tempfile.gettempdir(), out_name)
                result = dialouge_service.dialouge_extract_clip(video_path, start, end, out_path)
                st.success(f"Saved clip: {result}")
                st.video(result)

def save_uploaded_file(uploaded_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def main():
    st.title("ClipCraft UI")
    pg = st.navigation([
        st.Page(dialouge_search, title="Dialogue Search"),
        st.Page(face_clipper, title="Face Detection"), 
        st.Page(video_prompt, title="Prompt Search")
    ])
    pg.run()

if __name__ == "__main__":
    main()