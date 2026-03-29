import ffmpy
import threading



class FaceDetectionService: 
    health_string = "Face Detection Service is UP"

    def __init__(self):
       pass
       
    def health(self):
        return self.health_string
    
    def segemnt_video(video_path):
        pass