import pyttsx3

SAVE_PATH = "./voutput"


class ttsModule:
    
    def __init__(self, text):
        self.engine = pyttsx3.init()
        self.text = text

    def speak(self, save=False, save_path=SAVE_PATH):
        if save:
            self.engine.save_to_file(self.text, f"{save_path}/talk.mp3")
        self.engine.say(self.text)
        self.engine.runAndWait()

    def get_current_property(self):
        rate = self.engine.getProperty("rate")
        volume = self.engine.getProperty("volume")
        voice = self.engine.getProperty("voices")
        return rate, volume, voice

    def change_property(self, rate=125, volume=1.0, gender="m"):
        # gender select "m" or "f"
        if gender == "m":
            g_id = 0
        else:
            g_id = 1
        voices = self.engine.getProperty("voices")
        
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self.engine.setProperty("voice", voices[g_id].id)


if __name__ == "__main__":
    engine = ttsModule(text="Hello fucker")
    engine.change_property(rate=100, volume=0.8, gender="f")
    engine.speak()
