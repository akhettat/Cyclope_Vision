# Importation des librairies pour le modèle et l'affichage de l'entrainement

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# On clone le yolov5 si on ne l'a pas fait
if not Path("yolov5").exists():
    os.system("git clone https://github.com/ultralytics/yolov5.git")
    os.system("pip install -r yolov5/requirements.txt")

# Test de la disponibilité du GPU
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# Importation du module d'entrainement de notre modèle 
from yolov5 import train 


data_yaml = "path/to/your/dataset.yaml"  # Chemein vers notre dataset (format .yaml avec annotations dans un fichier .txt)  https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
weights = "yolov5s.pt" # Matrice des poids si préentrainé (à voir) 
epochs = 50  # Nombre d'époques
batch_size = 16  # Taille du lot
img_size = 640  # Taille des images (par défaut : 640)

# Lancer l'entraînement
train.run(
    data=data_yaml,  # Données d'entraînement
    weights=weights,  # Modèle pré-entraîné
    epochs=epochs,  # Nombre d'époques
    batch_size=batch_size,  # Taille des lots
    imgsz=img_size,  # Taille des images
)