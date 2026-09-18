from fastapi import UploadFile, HTTPException
from PIL import Image
from io import BytesIO


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

        # Limite maximale de 5 Mo
    TAILLE_MAX = 5 * 1024 * 1024

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
        if format_image not in ["JPEG", "PNG", "WEBP"]:
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

    # Résultat temporaire
    return {
        "resultat": "plastique",
        "scoreConfiance": 0.95
    }