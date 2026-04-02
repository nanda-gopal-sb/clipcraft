import os
import subprocess
import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from deepface import DeepFace
for services.utils.utils import utils

# ── Face detection setup ──────────────────────────────────────────────────────
# Uses OpenCV's built-in Haar cascade (no extra model downloads needed).
class face_detection:
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
 


def get_face_scenes(video, scenes):
    cap = cv2.VideoCapture(video)
    face_scenes = []
    
    for start, end in scenes:
        has_face = False
        timestamps = [start + (end - start) * i / 4 for i in range(5)]
        
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                has_face = True
                break
                
        if has_face:
            face_scenes.append((start, end))

    cap.release()
    return face_scenes


def get_scenes_with_reference(video, scenes, ref_img_path):
    cap = cv2.VideoCapture(video)
    matching_scenes = []
    
    for start, end in scenes:
        has_match = False
        timestamps = [start + (end - start) * i / 4 for i in range(5)]
        
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Crop face with a slight margin
                margin = int(w * 0.2)
                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)
                
                face_crop = frame[y1:y2, x1:x2]
                
                try:
                    # Verify using DeepFace with the Haar cascade crop
                    res = DeepFace.verify(
                        img1_path=ref_img_path,
                        img2_path=face_crop,
                        enforce_detection=False,
                    )
                    if res.get("verified", False):
                        has_match = True
                        break
                except Exception:
                    pass
                    
            if has_match:
                break
                
        if has_match:
            matching_scenes.append((start, end))

    cap.release()
    return matching_scenes
 

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    video = input("Enter video path: ").strip()
    ref_img = input("Enter reference image path (leave blank to just find any faces): ").strip()

    if not os.path.isfile(video):
        print("Video not found.")
        return

    if ref_img and not os.path.isfile(ref_img):
        print("Reference image not found.")
        return

    print("Detecting scenes...")
    scenes = self.utils.detect_scenes(video)

    if ref_img:
        print(f"Finding scenes containing the person from {ref_img}...")
        face_scenes = get_scenes_with_reference(video, scenes, ref_img)
        print(f"Found {len(face_scenes)} scenes matching the reference image.")
    else:
        print("Finding scenes with any faces...")
        face_scenes = get_face_scenes(video, scenes)
        print(f"Found {len(face_scenes)} scenes with faces.")

    if face_scenes:
        print("Extracting clips...")
        self.utils.extract_clips(video, face_scenes)

if __name__ == "__main__":
    main()
