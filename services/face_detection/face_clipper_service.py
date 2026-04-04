import cv2
import face_recognition
from services.utils.utils import Utils


class FaceDetectionService:
    def __init__(self):
        self.utils = Utils()
    
    def detect_scenes(self, video):
        return self.utils.detect_scenes(video)    
    
    def get_face_scenes(self, video, scenes):
        cap = cv2.VideoCapture(video)
        face_scenes = []
        
        for start, end in scenes:
            has_face = False
            # Sample 5 points across the scene duration
            timestamps = [start + (end - start) * i / 4 for i in range(5)]
            
            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret: continue
                
                # Convert BGR (OpenCV) to RGB (face_recognition)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect face locations
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if len(face_locations) > 0:
                    has_face = True
                    break
                    
            if has_face:
                face_scenes.append((start, end))
    
        cap.release()
        return face_scenes
    
    def get_scenes_with_reference(self, video, scenes, ref_img_path):
        # 1. Load and encode the reference image once
        ref_image = face_recognition.load_image_file(ref_img_path)
        ref_encodings = face_recognition.face_encodings(ref_image)
        
        if not ref_encodings:
            print("No face found in reference image.")
            return []
            
        target_encoding = ref_encodings[0]
        cap = cv2.VideoCapture(video)
        matching_scenes = []
        
        for start, end in scenes:
            has_match = False
            timestamps = [start + (end - start) * i / 4 for i in range(5)]
            
            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret: continue
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect and encode faces in the current frame
                current_encodings = face_recognition.face_encodings(rgb_frame)
                
                # Check if any face in the frame matches the reference
                if current_encodings:
                    matches = face_recognition.compare_faces(current_encodings, target_encoding, tolerance=0.6)
                    if True in matches:
                        has_match = True
                        break
                        
            if has_match:
                matching_scenes.append((start, end))
    
        cap.release()
        return matching_scenes
    
    def extract_clips(self, video, best_scenes):
        return self.utils.extract_clips(video, best_scenes)