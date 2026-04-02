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
    string = dialouge_service.health()
    st.title(string)

pg = st.navigation([
    st.Page(face_clipper, title="Face Detection and Clipping"),
    st.Page(video_prompt, title="Clip Extractor"),
    st.Page(dialouge_search, title="Dialouge Searcher")
])

pg.run()