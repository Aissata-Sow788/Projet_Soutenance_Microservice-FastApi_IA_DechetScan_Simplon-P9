from fastapi import FastAPI

# Importe les routes de notre API
from app.api.routes import router
import logging


# Création de l'application FastAPI
app = FastAPI(
    title="DechetScan IA",
    description="Microservice IA de reconnaissance des déchets",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
# Ajoute les routes à l'application
app.include_router(router)


# Route d'accueil
@app.get("/")
def accueil():

    # Réponse envoyée par l'API
    return {
        "message": "Microservice DechetScan IA opérationnel"
    }