import pyttsx3
from playsound import playsound
from googletrans import Translator


def translate_to_bengali(text):
    translator = Translator()
    translation = translator.translate(text, dest='bn')
    return translation.text


def text_to_speech(text, lang='en'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust the speaking rate if needed
    voices = engine.getProperty('voices')
    # Choose the appropriate Bengali voice (replace 'your_voice_name' with the desired voice)
    engine.setProperty('voice', voices[1].id)  # You may need to experiment with available voices

    engine.say(text)
    engine.save_to_file(text, 'output.mp3')  # Save the speech to a file
    engine.runAndWait()


def play_sound():
    # Assuming the file 'output.mp3' has been saved by the text-to-speech engine
    playsound('output.mp3')


if __name__ == "__main__":
    # Example usage
    translated_text = translate_to_bengali('My Friend Bad')
    text_to_speech(translated_text, lang='bn')
    play_sound()
