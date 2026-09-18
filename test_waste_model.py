from ultralytics import YOLO


# Charge le modèle de détection
model = YOLO("models/best.pt")


# Affiche les classes connues par le modèle
print("Classes du modèle :")
print(model.names)


# Image à analyser
image_path = "/home/aissata-sow/Téléchargements/papiers-dechets.jpeg"


# Lance la détection
results = model(image_path)


# Parcourt les résultats
for result in results:

    print("\n--- DÉTECTIONS ---")

    # Parcourt chaque objet détecté
    for box in result.boxes:

        # Récupère l'identifiant de la classe
        classe_id = int(box.cls[0])

        # Récupère le nom de la classe
        classe = model.names[classe_id]

        # Récupère le score de confiance
        confiance = float(box.conf[0])

        print(f"{classe} : {confiance:.2f}")