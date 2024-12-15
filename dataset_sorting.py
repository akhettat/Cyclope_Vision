import os
import cv2
import numpy as np
from shutil import copy2

os.system('cls')


# --- Charger et préparer les dossiers ---
# Dossier du dataset non trié
global_images_dir = r"D:\SUP\ET5\Project Cyclope\Code Python\global_Dataset\train\images"
global_labels_dir = r"D:\SUP\ET5\Project Cyclope\Code Python\global_Dataset\train\labels"

# Chemins pour sauvegarder les images triées
images_dir = r"D:\SUP\ET5\Project Cyclope\Code Python\Dataset\train\images" 
labels_dir = r"D:\SUP\ET5\Project Cyclope\Code Python\Dataset\train\labels"

# Créer les dossiers s'ils n'existent pas
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

wanted_class = 3.0 # La classe que l'on souhaite sauvegarder

# --- Code principal ---
for filename in os.listdir(global_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"): # Sélectionner seulement les fichiers images

        # Chemin de l'image
        global_image_path = os.path.join(global_images_dir, filename)
        image = cv2.imread(global_image_path) # Charger l'image

        # Correspondance du fichier label
        label_filename = os.path.splitext(filename)[0] + ".txt"                 # Associer le fichier .txt
        global_label_path = os.path.join(global_labels_dir, label_filename)     # Chemin complet du fichier label

        # Vérifier que le fichier de labels existe sinon continuer
        if not os.path.exists(global_label_path):
            continue

        # Vérifier si le label est le bon
        contains_label = False
        with open(global_label_path, 'r') as f:
            labels = []
            for line in f.readlines():
                label = list(map(float, line.split()))  # Convertir chaque ligne en liste de valeurs float
                if label[0] == wanted_class:  # Vérifier si la classe est la valeur souhaitée
                    contains_label = True
                labels.append(label)
        
        if contains_label:
            # Sauvegarder l'image triée
            image_path = os.path.join(images_dir, "aug_" + filename)
            cv2.imwrite(image_path, image)

            # Sauvegarder le label trié
            label_path = os.path.join(labels_dir, "aug_" + label_filename)
            with open(label_path, 'w') as f:
                for labels in labels:
                    f.write(" ".join(map(str, label)) + "\n") # Réécrire les labels dans le fichier
