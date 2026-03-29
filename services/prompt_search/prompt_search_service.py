class PromptService: 
    health_string = "Prompt Service is UP"

    def __init__(self):
        pass 
       
    def health(self):
        return self.health_string