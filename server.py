import os
import tempfile
import zipfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse

from services.dialouge_search.dialouge_service import DialougeService
from services.face_detection.face_clipper_service import FaceDetectionService
from services.prompt_search.prompt_search_service import PromptService

app = FastAPI()

dialouge_service = DialougeService()
face_service = FaceDetectionService()
prompt_service = PromptService()



def save_upload_to_temp(upload: UploadFile, suffix: str = "") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or f"_{upload.filename}")
    tmp.write(upload.file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def zip_files(file_paths: list[str], zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            if os.path.exists(fp):
                zf.write(fp, arcname=os.path.basename(fp))
    return zip_path


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "ClipCraft API"}



@app.post("/api/dialogue-search", summary="Search video dialogue and extract matching clips")
async def dialogue_search(
    video: UploadFile = File(..., description="Video file (mp4, mov, mkv)"),
    query: str = Form(..., description="Text to search for in the transcript"),
):
    """
    1. Transcribes the uploaded video.
    2. Searches the transcript for the given query.
    3. Extracts a clip for **every** matching segment.
    4. Returns a ZIP of the extracted clips.
    """
    video_path = save_upload_to_temp(video, suffix=".mp4")
    temp_files = [video_path]
    print(video_path)

    try:
        segments = dialouge_service.dialouge_transcribe(video_path)
        if not segments:
            raise HTTPException(status_code=422, detail="Transcription produced no segments.")

        matches = dialouge_service.search_transcript(segments, query)
        if not matches:
            raise HTTPException(status_code=404, detail="No matching dialogue found for the given query.")

        clip_paths = []
        for i, match in enumerate(matches):
            out_path = os.path.join(tempfile.gettempdir(), f"dialogue_clip_{i}.mp4")
            result = dialouge_service.dialouge_extract_clip(
                video_path, match["start"], match["end"], out_path
            )
            clip_paths.append(result)
            temp_files.append(result)

        if len(clip_paths) == 1:
            return FileResponse(
                clip_paths[0],
                media_type="video/mp4",
                filename="dialogue_clip.mp4",
            )

        # Multiple clips → ZIP
        zip_path = os.path.join(tempfile.gettempdir(), "dialogue_clips.zip")
        zip_files(clip_paths, zip_path)
        temp_files.append(zip_path)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="dialogue_clips.zip",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/face-detection", summary="Detect face scenes and extract clips")
async def face_detection(
    video: UploadFile = File(..., description="Video file (mp4, mov, mkv)"),
    reference_image: Optional[UploadFile] = File(
        None, description="Optional reference face image (jpg, png)"
    ),
):
    """
    1. Detects scenes in the uploaded video.
    2. Finds scenes containing faces (or a specific reference face).
    3. Extracts clips for matching face scenes.
    4. Returns a ZIP of the extracted clips.
    """
    video_path = save_upload_to_temp(video, suffix=".mp4")
    temp_files = [video_path]
    ref_path = None

    try:
        if reference_image is not None:
            ref_path = save_upload_to_temp(reference_image, suffix=".jpg")
            temp_files.append(ref_path)

        # Detect scenes
        scenes = face_service.detect_scenes(video_path)
        if not scenes:
            raise HTTPException(status_code=422, detail="No scenes detected in the video.")

        if ref_path:
            face_scenes = face_service.get_scenes_with_reference(video_path, scenes, ref_path)
        else:
            face_scenes = face_service.get_face_scenes(video_path, scenes)

        if not face_scenes:
            raise HTTPException(status_code=404, detail="No face scenes found in the video.")

        clip_paths = face_service.extract_clips(video_path, face_scenes)
        temp_files.extend(clip_paths)

        if not clip_paths:
            raise HTTPException(status_code=500, detail="Clip extraction failed.")

        if len(clip_paths) == 1:
            return FileResponse(
                clip_paths[0],
                media_type="video/mp4",
                filename="face_clip.mp4",
            )

        zip_path = os.path.join(tempfile.gettempdir(), "face_clips.zip")
        zip_files(clip_paths, zip_path)
        temp_files.append(zip_path)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="face_clips.zip",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/prompt-search", summary="Extract clips matching a text prompt")
async def prompt_search(
    video: UploadFile = File(..., description="Video file (mp4, mov, mkv)"),
    prompt: str = Form(..., description="Describe what you're looking for"),
):
    """
    1. Analyses scenes with vision + transcript.
    2. Scores each scene against the prompt.
    3. Extracts the top-matching clips.
    4. Returns a ZIP of the extracted clips.
    """
    video_path = save_upload_to_temp(video, suffix=".mp4")
    temp_files = [video_path]

    try:
        best_scenes = prompt_service.run(video_path, prompt)

        if not best_scenes:
            raise HTTPException(status_code=404, detail="No matching scenes found for the prompt.")

        clip_dir = "clips"
        clip_paths = sorted(
            [os.path.join(clip_dir, f) for f in os.listdir(clip_dir) if f.endswith(".mp4")]
        )

        if not clip_paths:
            raise HTTPException(status_code=500, detail="Clip extraction produced no files.")

        if len(clip_paths) == 1:
            return FileResponse(
                clip_paths[0],
                media_type="video/mp4",
                filename="prompt_clip.mp4",
            )

        zip_path = os.path.join(tempfile.gettempdir(), "prompt_clips.zip")
        zip_files(clip_paths, zip_path)
        temp_files.append(zip_path)

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="prompt_clips.zip",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
