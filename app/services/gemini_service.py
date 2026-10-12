import base64
import json
import time
import logging

from google import genai
from google.genai import types

from app.core.config import GEMINI_API_KEY
from app.schemas.gemini_schema import GeminiAnalysisResponse

logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Client Gemini
# -------------------------------------------------------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Modèle testé et fonctionnel avec cette clé API
GEMINI_MODEL = "gemini-3.5-flash"

# -------------------------------------------------------------
# Catégories de déchets autorisées dans DechetScan
# -------------------------------------------------------------
CATEGORIES_DECHETS = [
    "Plastique", "Papier", "Carton", "Verre", "Métal",
    "Textile", "Bois", "Déchet organique", "Déchet électronique",
    "Pile et batterie", "Déchet médical", "Déchet dangereux",
    "Huile usagée", "Déchet de construction", "Déchet vert",
    "Caoutchouc", "Cuir", "Déchet composite",
    "Déchet non recyclable", "Inconnu",
]

# -------------------------------------------------------------
# Prompt envoyé à Gemini
# -------------------------------------------------------------
GEMINI_PROMPT = """
Tu es le système d'analyse d'images de l'application DechetScan.

Ta mission est d'identifier les déchets réellement visibles dans l'image.

RÈGLES OBLIGATOIRES :
1. Analyse uniquement ce qui est clairement visible dans l'image.
2. Ne jamais inventer un objet ou un déchet qui n'est pas visible.
3. Utilise uniquement les catégories autorisées listées ci-dessous.
4. Si un déchet est difficile à identifier, utilise "Inconnu".
5. Si aucun déchet n'est présent, retourne un item avec "Inconnu".
6. Le champ "confiance" doit être entre 0 et 1.

CATÉGORIES AUTORISÉES :
Plastique, Papier, Carton, Verre, Métal, Textile, Bois,
Déchet organique, Déchet électronique, Pile et batterie,
Déchet médical, Déchet dangereux, Huile usagée,
Déchet de construction, Déchet vert, Caoutchouc, Cuir,
Déchet composite, Déchet non recyclable, Inconnu

FORMAT DE RÉPONSE (JSON strict) :
{
  "dechets": [
    {
      "objet": "nom de l'objet détecté",
      "categorie": "une des catégories autorisées",
      "confiance": 0.95
    }
  ]
}

Retourne UNIQUEMENT le JSON, sans texte autour.
"""


def analyser_image_avec_gemini(
    image_bytes: bytes,
    mime_type: str
) -> GeminiAnalysisResponse:
    """
    Envoie l'image à Gemini et retourne les déchets détectés.
    Utilise generate_content (API standard) avec envoi direct
    des bytes — compatible avec tous les comptes Gemini.
    """

    nombre_tentatives = 4

    for tentative in range(nombre_tentatives):
        try:
            logger.debug(
                f"Analyse Gemini - tentative {tentative + 1}/{nombre_tentatives}"
            )

            # ---------------------------------------------------------
            # Envoi de l'image directement en bytes via types.Part
            # C'est la méthode standard recommandée par Google.
            # ---------------------------------------------------------
            reponse = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type or "image/jpeg",
                    ),
                    types.Part.from_text(text=GEMINI_PROMPT),
                ],
                config=types.GenerateContentConfig(
                    # Force Gemini à répondre en JSON
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            texte_reponse = reponse.text

            if not texte_reponse:
                raise ValueError("Gemini a retourné une réponse vide.")

            logger.debug(f"Réponse Gemini : {texte_reponse}")

            # ---------------------------------------------------------
            # Nettoyage : retire les balises markdown ```json si présentes
            # ---------------------------------------------------------
            texte_propre = texte_reponse.strip()
            if texte_propre.startswith("```"):
                lignes = texte_propre.split("\n")
                texte_propre = "\n".join(lignes[1:-1])

            # Validation du JSON avec Pydantic
            resultat = GeminiAnalysisResponse.model_validate_json(texte_propre)

            print(" Analyse Gemini réussie.")
            return resultat

        except Exception as erreur:
            message_erreur = str(erreur)
            print(f" Erreur Gemini : {message_erreur}")

            # Erreurs temporaires → on réessaie
            erreur_temporaire = any(x in message_erreur for x in [
                "503", "UNAVAILABLE", "429",
                "RESOURCE_EXHAUSTED", "service_unavailable",
                "high demand", "overloaded",
            ])

            # Erreur définitive → on arrête immédiatement
            if not erreur_temporaire:
                raise

            # Toutes les tentatives épuisées
            if tentative == nombre_tentatives - 1:
                print(" Gemini reste indisponible après plusieurs tentatives.")
                raise

            # Attente progressive : 2s → 4s → 8s
            temps_attente = 2 ** (tentative + 1)
            print(f" Nouvelle tentative dans {temps_attente}s...")
            time.sleep(temps_attente)

    raise RuntimeError("Impossible d'obtenir une réponse de Gemini.")
