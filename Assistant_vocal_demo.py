import os
import pyttsx3
import numpy as np
import sounddevice as sd
import time

#os.system('cls')


### INITIALISATION SYNTHÈSE VOCALE ###
engine = pyttsx3.init()  # Initialiser le narrateur
engine.setProperty('rate', 150)  # Régler la vitesse
engine.setProperty('volume', 1.0)  # Régler le volume
voices = engine.getProperty('voices')  # Récupère les voix disponibles
engine.setProperty('voice', voices[0].id)  # Choisir une voix (0 français et 1 anglais)


### INITIALISATION INDICES SONORES ###
fs = 44100
duration = 0.5
duration2 = 0.5
freq1 = 1000  # Fréquence fondamentale
freq2 = 2000  # Première harmonique
freq3 = 3000  # Deuxième harmonique

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
envelope = np.exp(-5 * t)  # Réduction du volume
ding = envelope * (np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.3 * np.sin(2 * np.pi * freq3 * t))
ding /= np.max(np.abs(ding))  # Normaliser le signal pour éviter la saturation

t2 = np.linspace(0, duration2, int(fs * duration2), endpoint=False)
envelope = np.exp(-5 * t2)
ding2 = envelope * (np.sin(2 * np.pi * freq1 * t2) + 0.5 * np.sin(2 * np.pi * freq2 * t2) + 0.3 * np.sin(2 * np.pi * freq3 * t2))
ding2 /= np.max(np.abs(ding))  # Normaliser le signal pour éviter la saturation


### PARAMÈTRES D'ENTRÉE ###
is_crosswalk = True
is_still_in_sight = False

'''engine.say("Bienvenue")
engine.runAndWait()
engine.stop()'''


### CODE PRINCIPAL ###
# Fonction pour déclarer la présence d'un passage piéton
def PositionToSpeech(is_crosswalk) :

    if is_crosswalk:
        sd.play(ding, samplerate=fs)
        time.sleep(0.2)
        sd.play(ding2, samplerate=fs)
        sd.wait()
        is_still_in_sight == True
    
    elif is_still_in_sight == False :
        engine.say("Recherche")
        engine.runAndWait()
        engine.stop()

    



PositionToSpeech(is_crosswalk)




'''
# Exemple 1 : Text to Speech
import pyttsx3

engine = pyttsx3.init()  # Initialiser le narrateur
engine.setProperty('rate', 150)  # Régler la vitesse
engine.setProperty('volume', 1.0)  # Régler le volume
voices = engine.getProperty('voices')  # Récupère les voix disponibles
engine.setProperty('voice', voices[0].id)  # Choisir une voix (0 français et 1 anglais)
engine.say("Test")  # Charger une phrase
engine.runAndWait()  # Déclancher la lecture
engine.stop()  # Arrêter la lecture
'''

'''
# Exemple 2 : générer un son
# import numpy as np
# import sounddevice as sd

# Paramètres de l'onde sinusoïdale
fs = 44100  # fréquence d'échantillonnage
duration = 0.3  # durée en secondes
freq = 440  # fréquence de la note (en Hz)

# Générer un signal sinusoïdal
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
left_channel = 0.5 * np.sin(2 * np.pi * freq * t)  # Canal gauche
right_channel = 0.5 * np.sin(2 * np.pi * freq * t + np.pi / 4)  # Canal droit avec phase décalée
# Empiler les deux canaux pour créer un signal stéréo
stereo_signal = np.vstack((left_channel, right_channel)).T
# Jouer le son stéréo
sd.play(stereo_signal, samplerate=fs)
sd.wait()
'''

'''
# Exemple 3 : enregistrer un fichier audio
from pydub import AudioSegment
from pydub.generators import Sine

# Générer un signal sinusoïdal (par exemple 440 Hz, fréquence de la note A)
signal = Sine(440).to_audio_segment(duration=1000)  # 1000 ms = 1 seconde
# Appliquer une spatialisation en stéréo (en modifiant l'intensité de chaque canal)
left = signal - 10  # Canal gauche, un peu plus bas en volume
right = signal + 10  # Canal droit, un peu plus haut en volume
# Combiner les deux canaux pour créer un effet stéréo
stereo_signal = AudioSegment.from_mono_audiosegments(left, right)
'''

'''
# Exemple 4 : lire un fichier audio
import pygame

pygame.mixer.init()
sound = pygame.mixer.Sound('sound_file.wav')  # Charger un fichier sonore
sound.set_volume(0.5, 0.5)  # Volume égal pour les deux canaux : va de -1.0 (entièrement à gauche) à 1.0 (entièrement à droite)
sound.play()  # Jouer le son
pygame.time.delay(1000)  # Attendre que le son se termine
'''

print("Terminé")