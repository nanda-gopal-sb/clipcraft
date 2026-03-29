class DialougeService: 
    health_string = "Dialouge Detection Service is UP"

    # Contructor
    def __init__(self): 
        pass 
    

    def health(self):
        segmented_video = segment_video()
        return self.health_string