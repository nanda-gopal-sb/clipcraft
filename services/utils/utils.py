import tempfile
import ffmpeg
import whisper
import os

class Utils:
    def __init__(self):
        pass

    def transcribe_video(video_path) -> list:   
        """
        "text": segment["text"].strip(),
        "start": segment["start"],
        "end": segment["end"]
        """          
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
        segments = []
        try:
            ffmpeg.input(video_path).output(temp_audio_path, ac=1, ar=16000, loglevel="error").overwrite_output().run()
            
            model = whisper.load_model("base")
            result = model.transcribe(temp_audio_path)
            
            for segment in result["segments"]:
                segments.append({
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
        return segments

    def extract_clip(video_path, start, end, output_path) -> str:
         # Need to make changes such that it can take a list of timestamps

        start_time = max(0, start - 0.5)
        end_time = end + 0.5
        duration = end_time - start_time
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        (
            ffmpeg
            .input(video_path, ss=start_time)
            .output(output_path, t=duration, loglevel="error")
            .overwrite_output()
            .run()
        )
        return output_path