from fastapi import APIRouter, UploadFile, File

# Importe le service IA
from app.services.ia_service import analyser_image


# Création du routeur
router = APIRouter()


# Route de test
@router.get("/test")
def test_ia():

    # Réponse de test
    return {
        "message": "Route IA opérationnelle"
    }


# Route pour analyser une photo
@router.post("/api/ia/analyse")
async def analyser_dechet(photoUrl: UploadFile = File(...)):

    # Lance l'analyse de la photo
    resultat = await analyser_image(photoUrl)

    # Retourne le résultat IA
    return resultat