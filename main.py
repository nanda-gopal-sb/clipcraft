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
    string = prompt_service.health()
    st.title(string)


def dialouge_search():
    string = dialouge_service.health()
    st.title(string)

pg = st.navigation([
    st.Page(face_clipper, title="Face Detection and Clipping"),
    st.Page(video_prompt, title="Clip Extractor"),
    st.Page(dialouge_search, title="Dialouge Searcher")
])

pg.run()