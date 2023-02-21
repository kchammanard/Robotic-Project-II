# This requires internet connection to work, if required to work without internet use the other file
# if error use playsound 1.2.2

import os
import playsound
from gtts import gTTS

SAVE_PATH = "./voutput"

class gttsModule:

    def __init__(self, text, language="en", save_path=SAVE_PATH, accent="com"):
        self.engine = gTTS(text, tld=accent, lang=language, slow=False)
        self.path = save_path
        self.engine.save(f"{self.path}/talk.mp3")

    def speak(self):
        playsound.playsound(f"{self.path}/talk.mp3")

if __name__ == "__main__":
    engine = gttsModule(text="hello fucker", accent="com")
    engine.speak()
