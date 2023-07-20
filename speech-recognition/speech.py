import speech_recognition as sr

def record_audio():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source, timeout=5)  # Record audio for a maximum of 5 seconds

    return audio

def speech_to_text(audio):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    try:
        # Perform speech recognition
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == "__main__":
    audio_data = record_audio()
    if audio_data:
        text_output = speech_to_text(audio_data)
        if text_output:
            print("You said: ", text_output)
