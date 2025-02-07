import os
import cv2
import numpy as np
import random
from shutil import copy2

os.system('cls')


# --- Charger et préparer les dossiers ---
images_dir = r"C:\Users\esteb\Desktop\Dataset\valid\images"  # Dossier des images
labels_dir = r"C:\Users\esteb\Desktop\Dataset\valid\labels"  # Dossier des labels

# Chemins pour sauvegarder les données augmentées
augmented_images_dir = r"C:\Users\esteb\Desktop\AugmentedDataset\valid\images"
augmented_labels_dir = r"C:\Users\esteb\Desktop\AugmentedDataset\valid\labels"

# Créer les dossiers augmentés s'ils n'existent pas
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)


# --- Fonctions d'augmentation ---
# Flip horizontal
def augment_flip(image, labels) :

    image = cv2.flip(image, 1)      # Appliquer un flip horizontal
    for label in labels:
        label[1] = 1.0 - label[1]   # Mise à jour des labels

    return image, labels


# Zoom aléatoire
def augment_zoom(image, labels) :

    h_img, w_img = image.shape[:2]          # Obtenir la hauteur et la largeur de l'image
    zoom_scale = 0
    while not (0.90 <= zoom_scale <= 1.10) :
        zoom_scale = random.uniform(0.7, 1.3)   # Zoom entre 70 et 130%
    zoom_matrix = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), 0, zoom_scale)  # Matrice de transformation
    image = cv2.warpAffine(image, zoom_matrix, (w_img, h_img))  # Appliquer le zoom

    # Mise à jour des labels
    for label in labels:
        label[3] *= zoom_scale  # Ajuster la largeur
        label[4] *= zoom_scale  # Ajuster la hauteur

    return image, labels


#Modification de la luminosité/contraste
def augment_brightness(image) :

    contrast = random.uniform(0.5, 1.5)    # Contraste entre 0.5 et 1.5
    brightness = 0
    while brightness < 5 :
        brightness = random.uniform(-20, 20)    # Luminosité entre -20 et 20, sup a 5 ou inf a -5
    
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness) # Appliquer les changements
    #image = cv2.convertScaleAbs(image, beta=brightness) # Appliquer les changements

    return image


# Flou gaussien
def augment_blur(image) :

    kernel_size = random.choice([3, 5, 7]) # Noyau aléatoire
    if kernel_size != 0:
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) # Appliquer les changements

    return image


# Bruit gaussien
def augment_noise(image):
    
    mean = 0
    sigma = random.uniform(0.5, 2.0) # Valeur de l'écart-type entre 0 et 0.37
    gauss_noise = np.random.normal(mean, sigma, image.shape).astype('uint8')

    noisy_image = cv2.add(image, gauss_noise) # Ajouter le bruit
    image = noisy_image

    return image


# Parcourir les images
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"): # Sélectionner seulement les fichiers images

        # Chemin de l'image
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path) # Charger l'image

        # Correspondance du fichier label
        label_filename = os.path.splitext(filename)[0] + ".txt"     # Associer le fichier .txt
        label_path = os.path.join(labels_dir, label_filename)       # Chemin complet du fichier label

        # Vérifier que le fichier de labels existe sinon continuer
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            labels = [list(map(float, line.split())) for line in f.readlines()]
                

        # Aucune augmentation
        augmented_image = image
        augmented_labels = labels

        # Sauvegarder l'image augmentée
        augmented_image_path = os.path.join(augmented_images_dir, "aug_0_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        # Sauvegarder les labels augmentés
        augmented_label_path = os.path.join(augmented_labels_dir, "aug_0_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")  # Réécrire les labels dans le fichier


        # Augmentation 1 : Enchainement total
        augmented_image = image
        augmented_labels = labels
        augmented_image, augmented_labels = augment_flip(augmented_image, augmented_labels)
        augmented_image, augmented_labels = augment_zoom(augmented_image, augmented_labels)
        augmented_image = augment_brightness(augmented_image)
        augmented_image = augment_blur(augmented_image)
        augmented_image = augment_noise(augmented_image)

        # Sauvegarder l'image augmentée
        augmented_image_path = os.path.join(augmented_images_dir, "aug_1_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        # Sauvegarder les labels augmentés
        augmented_label_path = os.path.join(augmented_labels_dir, "aug_1_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")  # Réécrire les labels dans le fichier


        # Augmentation 2 : Flip
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_flip(augmented_image, augmented_labels)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_2_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_2_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 3 : Zoom
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_zoom(augmented_image, augmented_labels)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_3_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_3_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 4 : Brightness
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image = augment_brightness(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_4_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_4_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 5 : Blur
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image = augment_blur(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_5_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_5_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 6 : Noise
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image = augment_noise(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_6_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_6_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")

"""
        # Augmentation 7 : Flip + Zoom
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_flip(augmented_image, augmented_labels)
        augmented_image, augmented_labels = augment_zoom(augmented_image, augmented_labels)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_7_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_7_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 8 : Flip + Brightness
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_flip(augmented_image, augmented_labels)
        augmented_image = augment_brightness(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_8_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_8_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 9 : Flip + Blur
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_flip(augmented_image, augmented_labels)
        augmented_image = augment_blur(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_9_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_9_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 10 : Flip + Noise
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_flip(augmented_image, augmented_labels)
        augmented_image = augment_noise(augmented_image)
        
        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_10_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_10_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 11 : Zoom + Brightness
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_zoom(augmented_image, augmented_labels)
        augmented_image = augment_brightness(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_11_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_11_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 12 : Zoom + Blur
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_zoom(augmented_image, augmented_labels)
        augmented_image = augment_blur(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_12_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_12_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 13 : Zoom + Noise
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image, augmented_labels = augment_zoom(augmented_image, augmented_labels)
        augmented_image = augment_noise(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_13_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_13_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 14 : Blur + Noise
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image = augment_blur(augmented_image)
        augmented_image = augment_noise(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_14_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_14_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 15 : Blur + Brightness
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image = augment_blur(augmented_image)
        augmented_image = augment_brightness(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_15_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_15_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 16 : Noise + Brightness
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image = augment_noise(augmented_image)
        augmented_image = augment_brightness(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_16_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_16_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")


        # Augmentation 17 : Blur + Noise + Brightness
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        augmented_image = augment_blur(augmented_image)
        augmented_image = augment_noise(augmented_image)
        augmented_image = augment_brightness(augmented_image)

        # Sauvegarde
        augmented_image_path = os.path.join(augmented_images_dir, "aug_17_" + filename)
        cv2.imwrite(augmented_image_path, augmented_image)

        augmented_label_path = os.path.join(augmented_labels_dir, "aug_17_" + label_filename)
        with open(augmented_label_path, 'w') as f:
            for label in augmented_labels:
                f.write(" ".join(map(str, label)) + "\n")
"""