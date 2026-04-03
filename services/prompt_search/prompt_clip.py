import os
import subprocess
import whisper
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import base64
import requests
from services.utils import Utils

utils = Utils()

'''
def transcribe_audio(video):
    model = whisper.load_model("base")
    result = model.transcribe(video)
    print("Transcript done")
    return result["segments"]
    '''
'''
def detect_scenes(video):
    video_stream = open_video(video)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video_stream)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for start, end in scene_list:
        scenes.append((start.get_seconds(), end.get_seconds()))

    print("Scenes:", scenes)
    return scenes
    '''

def get_scene_transcript(scene, transcript_segments):
    scene_start, scene_end = scene
    matched_text = []

    for segment in transcript_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]

        if seg_start < scene_end and seg_end > scene_start:
            matched_text.append(segment["text"])

    return " ".join(matched_text).strip()


def describe_scene(scene, video):
    start, end = scene
    middle_time = (start + end) / 2

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, middle_time * 1000)
    success, frame = cap.read()
    cap.release()

    if not success:
        return "Could not read frame"

    frame_path = "temp_frame.jpg"
    cv2.imwrite(frame_path, frame)

    with open(frame_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "llava",
        "prompt": "Describe what is visually happening in this scene in one short sentence.",
        "images": [image_base64],
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    description = response.json()["response"].strip()

    print("Description:", description)
    return description

def score_scene(prompt, description, transcript_text):
    payload = {
        "model": "mistral",
        "prompt": f"""
User Prompt: {prompt}

Scene Description: {description}

Scene Transcription: {transcript_text}

You are scoring how well this video scene matches the user's prompt.

Use BOTH:
1. the visual scene description
2. the spoken dialogue / transcription

Compare them together against the user's prompt.

Give only a single number from 0 to 10.

Scoring rules:
- 0 = completely unrelated
- 5 = somewhat relevant
- 10 = perfect match

Only return the number.
""",
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    score_text = response.json()["response"].strip()

    try:
        score = float(score_text)
    except:
        score = 0

    print("Score:", score)
    return score

def extract_clips(video, best_scenes):
    os.makedirs("clips", exist_ok=True)

    for i, scene in enumerate(best_scenes, start=1):
        start, end = scene
        output_file = f"clips/clip_{i}.mp4"

        command = [
            "ffmpeg",
            "-i", video,
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            output_file
        ]

        subprocess.run(command, check=True)

    print("Clips extracted successfully.")

def run_prompt_clipper(video_path, prompt):
    transcript = Utils.transcribe_video(video_path)
    scenes = Utils.detect_scenes(video_path)

    scored_scenes = []

    for scene in scenes:
        description = describe_scene(scene, video_path)
        transcript_text = get_scene_transcript(scene, transcript)
        score = score_scene(prompt, description, transcript_text)
        scored_scenes.append((scene, score))

    scored_scenes.sort(key=lambda x: x[1], reverse=True)
    best_scenes = [scene for scene, score in scored_scenes[:3]]
    extract_clips(video_path, best_scenes)

    return best_scenes

'''
video_path = input("Enter video path: ")
prompt = input("Enter prompt: ")

transcript = transcribe_audio(video_path)
scenes = detect_scenes(video_path)

scored_scenes = []

for scene in scenes:
    description = describe_scene(scene, video_path)
    score = score_scene(prompt, description, transcript_text)
    scored_scenes.append((scene, score))

scored_scenes.sort(key=lambda x: x[1], reverse=True)
best_scenes = [scene for scene, score in scored_scenes[:3]]
extract_clips(video_path, best_scenes)
'''
