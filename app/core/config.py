import os
from dotenv import load_dotenv


# Charge les variables du fichier .env
load_dotenv()


# Récupère la clé API Gemini
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Clé de l'API Groq (définie dans le .env du docker-compose).
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("La variable GROQ_API_KEY n'est pas configurée.")