# Feed audio data to collect OSINT

import SpeechRecognition as sr

listOfAudioFiles = ["boy18.wav", "Comey.wav"]
keywords = ["Tweeter", "LinkedIn", "Facebook", "Instagram"]

def transcribeAudioFile (audioFile):
    r = sr.Recognizer()
    with sr.AudioFile(audioFile) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UknownValueError:
        print("Google speech recognition cannot understand audio")
        return ""
    except sr.requestError as e:
        print("Could not request results from speech recognition service; {0}".format(e))
        return ""

corpus = []
for audioFile in listOfAudioFiles:
    corpus[transcribeAudioFile(audioFile)] = audioFile

print(corpus)

for keyword in keywords:
    for text in corpus:
        if keyword in text:
            print("keyword " + keyword + " found in audio " + "\""+corpus[text] + "\"")
