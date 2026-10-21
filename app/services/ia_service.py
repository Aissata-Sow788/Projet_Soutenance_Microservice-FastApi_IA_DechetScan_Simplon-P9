import logging
from io import BytesIO

from fastapi import UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps

# Fonction qui envoie l'image aux modèles OpenRouter pour l'analyse
from app.services.ia_service_groq import analyser_image_avec_ia

logger = logging.getLogger(__name__)

# Formats acceptés (détectés à partir du contenu réel de l'image,
# pas de ce que déclare le navigateur).
FORMATS_AUTORISES = {"JPEG", "PNG", "WEBP"}

# Limite maximale de 5 Mo
TAILLE_MAX = 5 * 1024 * 1024

# Taille maximale (en pixels) du plus grand côté envoyé à l'IA.
# Une image plus petite se transmet et s'analyse beaucoup plus vite,
# sans perdre en précision pour reconnaître un déchet.
COTE_MAX_IA = 1024


def _preparer_image_pour_ia(contenu: bytes) -> bytes:
    """
    Redimensionne l'image (1024 px max) et la convertit en JPEG.
    Corrige aussi l'orientation des photos de téléphone.
    En cas de problème, retourne l'image d'origine.
    """
    try:
        image = Image.open(BytesIO(contenu))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.thumbnail((COTE_MAX_IA, COTE_MAX_IA))

        sortie = BytesIO()
        image.save(sortie, format="JPEG", quality=85)
        return sortie.getvalue()

    except Exception:
        logger.warning("Réduction de l'image impossible, envoi tel quel.")
        return contenu


# Analyse une image de déchet
async def analyser_image(photoUrl: UploadFile):

    # Lit les données binaires
    contenu = await photoUrl.read()

    # Vérifie si le fichier est vide
    if not contenu:
        raise HTTPException(
            status_code=400,
            detail="Le fichier est vide"
        )

    # Vérifie la taille du fichier
    if len(contenu) > TAILLE_MAX:
        raise HTTPException(
            status_code=400,
            detail="L'image est trop volumineuse. Taille maximale : 5 Mo"
        )

    try:
        # Reconstruit l'image
        image = Image.open(BytesIO(contenu))

        # Récupère le format
        format_image = image.format

        # Vérifie le format
        if format_image not in FORMATS_AUTORISES:
            raise HTTPException(
                status_code=400,
                detail="Format d'image non autorisé"
            )

        # Vérifie que l'image est valide
        image.verify()

        # Rouvre l'image après verify()
        image = Image.open(BytesIO(contenu))

        # Vérifie les dimensions
        largeur, hauteur = image.size

        if largeur < 100 or hauteur < 100:
            raise HTTPException(
                status_code=400,
                detail="L'image est trop petite. Dimensions minimales : 100x100 pixels"
            )

    except HTTPException:
        # Relaye l'erreur de validation
        raise

    except Exception:
        # Rejette l'image invalide
        raise HTTPException(
            status_code=400,
            detail="Le fichier envoyé n'est pas une image valide"
        )

    # Réduit l'image pour accélérer l'analyse.
    image_pour_ia = _preparer_image_pour_ia(contenu)

    # Envoie l'image aux modèles d'IA.
    # La fonction d'analyse est bloquante (appels réseau) :
    # on l'exécute dans un thread pour ne pas figer le serveur FastAPI.
    try:
        resultat = await run_in_threadpool(
            analyser_image_avec_ia,
            image_bytes=image_pour_ia,
            mime_type="image/jpeg",
        )

    except Exception as erreur:
        logger.exception("Échec de l'analyse d'image par l'IA")

        # TEMPORAIRE : montre la vraie cause dans la réponse.
        raise HTTPException(
            status_code=503,
            detail=f"Analyse indisponible : {str(erreur)[:400]}"
        )

    # Retourne le résultat de l'IA
    return resultat.model_dump()