from fastapi import UploadFile, HTTPException
from PIL import Image
from io import BytesIO
# Fonction qui envoie l'image à Gemini pour l'analyse
from app.services.gemini_service import analyser_image_avec_gemini


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
    
    # Envoie l'image validée à Gemini pour analyser les déchets
    try:
        resultat_gemini = analyser_image_avec_gemini(
            image_bytes=contenu,
            mime_type=photoUrl.content_type
        )

    except Exception as erreur:
        # Affiche l'erreur réelle envoyée par Gemini dans le terminal.
        print("========================================")
        print("ERREUR GEMINI :", repr(erreur))
        print("========================================")

        # Retourne temporairement l'erreur pour faciliter le diagnostic.
        raise HTTPException(
            status_code=503,
            detail=f"Erreur Gemini : {str(erreur)}"
        )

    # Retourne le résultat de Gemini
    return resultat_gemini.model_dump()