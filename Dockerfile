# =============================================================
# DOCKERFILE — Microservice IA FastAPI (DechetScan)
# =============================================================
# Ce conteneur héberge le microservice d'analyse d'images.
# Il reçoit une photo de déchet, l'envoie à Google Gemini,
# et retourne le type de déchet identifié au backend Django.
# =============================================================


# -------------------------------------------------------------
# ÉTAPE 1 : Image de base
# -------------------------------------------------------------
# Python 3.12 slim : léger, suffisant pour FastAPI + Pillow.
# On n'a pas besoin de dépendances système lourdes ici.
# -------------------------------------------------------------
FROM python:3.12-slim


# -------------------------------------------------------------
# ÉTAPE 2 : Variables d'environnement système
# -------------------------------------------------------------
# Désactive le bytecode .pyc et active les logs temps réel.
# -------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# -------------------------------------------------------------
# ÉTAPE 3 : Dépendances système pour Pillow
# -------------------------------------------------------------
# Pillow (traitement d'images) nécessite quelques bibliothèques
# système pour gérer les formats JPEG, PNG, etc.
# -------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# -------------------------------------------------------------
# ÉTAPE 4 : Répertoire de travail dans le conteneur
# -------------------------------------------------------------
WORKDIR /app


# -------------------------------------------------------------
# ÉTAPE 5 : Installation des dépendances Python
# -------------------------------------------------------------
# On copie requirements.txt en premier pour profiter du cache
# Docker (si requirements.txt ne change pas, cette couche
# n'est pas reconstruite même si le code change).
# -------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# -------------------------------------------------------------
# ÉTAPE 6 : Copie du code source
# -------------------------------------------------------------
# On copie tout le projet FastAPI dans /app du conteneur.
# -------------------------------------------------------------
COPY . .


# -------------------------------------------------------------
# ÉTAPE 7 : Port exposé
# -------------------------------------------------------------
# Le microservice IA écoute sur le port 8001.
# Ce port est différent de Django (8000) pour éviter les conflits.
# -------------------------------------------------------------
EXPOSE 8001


# -------------------------------------------------------------
# ÉTAPE 8 : Commande de lancement
# -------------------------------------------------------------
# Uvicorn est le serveur ASGI utilisé pour lancer FastAPI.
#   "app.main:app"       → fichier app/main.py, objet FastAPI "app"
#   --host 0.0.0.0       → écoute sur toutes les interfaces réseau
#   --port 8001          → port d'écoute
#   --workers 2          → 2 processus parallèles pour les requêtes
# -------------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
