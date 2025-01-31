import time
import torch
from ultralytics import YOLO
import torch.multiprocessing as mp

print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    
model_v1 = YOLO("runs/train/custom_yolov11_training_v2/weights/best.pt") 
model_v2 = YOLO("runs/train/custom_yolov11_training_v3/weights/best.pt") 

image_path = "runs/train/custom_yolov11_training/les-personnes-qui-traversent-la-route-a-un-passage-cloute-sur-london-wall-city-of-london-uk-mmx975.jpg"
model_v1(image_path)
model_v2(image_path)
# Fonction d'inférence pour YOLO
def yolo_inferencev1(queue, num_iterations=10):
    print(f"YOLO_v1 Chargement du modèle YOLO...")
    durations = []
    for i in range(num_iterations):
        time1 = time.time_ns()
        results = model_v1(image_path)
        time2 = time.time_ns()
        duration_ms = (time2 - time1) / 1e6  # Conversion en millisecondes
        durations.append(duration_ms)
        print(f"YOLO_v1 Temps d'inférence (mesure {i + 1}): {duration_ms:.2f} ms")
        
        # Envoyer le résultat au post-traitement spécifique
        queue.put((results, duration_ms))

    # Envoyer un signal de fin au post-traitement
    average_duration = sum(durations) / len(durations)
    print(f"YOLO_v1 Temps moyen d'inférence : {average_duration:.2f} ms")
    queue.put(("FINISHED", None))
    print(f"YOLO_v1 Toutes les inférences sont terminées.")
    
def yolo_inferencev2(queue, num_iterations=10):
    print(f"YOLO_v2 Chargement du modèle YOLO...")
    durations = []
    for i in range(num_iterations):
        time1 = time.time_ns()
        results = model_v2(image_path)
        time2 = time.time_ns()
        duration_ms = (time2 - time1) / 1e6  # Conversion en millisecondes
        durations.append(duration_ms)
        print(f"YOLO_v2 Temps d'inférence (mesure {i + 1}): {duration_ms:.2f} ms")
        
        # Envoyer le résultat au post-traitement spécifique
        queue.put((results, duration_ms))

    # Envoyer un signal de fin au post-traitement
    average_duration = sum(durations) / len(durations)
    print(f"YOLO_v2 Temps moyen d'inférence : {average_duration:.2f} ms")
    queue.put(("FINISHED", None))
    print(f"YOLO_v2 Toutes les inférences sont terminées.")

# Fonction de post-traitement spécifique à chaque modèle
def post_processing_task(model_name, queue):
    print(f"Processus de post-traitement ({model_name}) démarré...")
    
    while True:
        results, duration = queue.get()

        if results == "FINISHED":
            print(f"[{model_name} - Post-traitement] Fin du post-traitement.")
            break
        else:
            # Simuler un affichage ou un traitement des résultats
            print(f"[{model_name} - Post-traitement] Temps d'inférence : {duration:.2f} ms")
            print(f"[{model_name} - Post-traitement] Résultats enregistrés.")

    print(f"[{model_name} - Post-traitement] Terminé.")

if __name__ == "__main__":
    mp.set_start_method("spawn")


    # Création des queues de communication
    queue_v1 = mp.Queue()
    queue_v2 = mp.Queue()

    # Création des processus pour chaque modèle
    yolo_process_v1 = mp.Process(target=yolo_inferencev1, args=(queue_v1,))
    yolo_process_v2 = mp.Process(target=yolo_inferencev2, args=(queue_v2,))
    
    # Création des processus de post-traitement
    post_process_v1 = mp.Process(target=post_processing_task, args=("YOLO_v1", queue_v1))
    post_process_v2 = mp.Process(target=post_processing_task, args=("YOLO_v2", queue_v2))

    # Démarrage des processus
    yolo_process_v1.start()
    yolo_process_v2.start()
    post_process_v1.start()
    post_process_v2.start()

    # Attente de la fin des processus
    yolo_process_v1.join()
    yolo_process_v2.join()
    post_process_v1.join()
    post_process_v2.join()

    print("Tous les processus sont terminés.")
