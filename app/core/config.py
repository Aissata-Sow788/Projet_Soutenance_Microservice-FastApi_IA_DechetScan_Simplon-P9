import os
from dotenv import load_dotenv


# Charge les variables du fichier .env
load_dotenv()


# Récupère la clé API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Vérifie que la clé est bien configurée
if not GEMINI_API_KEY:
    raise ValueError("La variable GEMINI_API_KEY n'est pas configurée.")