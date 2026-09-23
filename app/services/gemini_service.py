import base64
import time

from google import genai

from app.core.config import GEMINI_API_KEY
from app.schemas.gemini_schema import GeminiAnalysisResponse


# Création du client Gemini avec la clé API configurée dans le projet.
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Modèle Gemini utilisé pour l'analyse des images.
GEMINI_MODEL = "gemini-3.6-flash"


# Catégories de déchets autorisées dans DechetScan.
CATEGORIES_DECHETS = [
    "Plastique",
    "Papier",
    "Carton",
    "Verre",
    "Métal",
    "Textile",
    "Bois",
    "Déchet organique",
    "Déchet électronique",
    "Pile et batterie",
    "Déchet médical",
    "Déchet dangereux",
    "Huile usagée",
    "Déchet de construction",
    "Déchet vert",
    "Caoutchouc",
    "Cuir",
    "Déchet composite",
    "Déchet non recyclable",
    "Inconnu",
]


# Instructions envoyées à Gemini pour analyser l'image.
GEMINI_PROMPT = """
Tu es le système d'analyse d'images de l'application DechetScan.

Ta mission est d'identifier les déchets réellement visibles dans l'image.

RÈGLES OBLIGATOIRES :

1. Analyse uniquement ce qui est clairement visible dans l'image.
2. Ne jamais inventer un objet ou un déchet qui n'est pas visible.
3. Utilise uniquement les catégories de déchets autorisées par DechetScan.
4. Ne crée jamais une nouvelle catégorie.
5. Si un déchet est difficile à identifier, ambigu, trop petit, flou
   ou impossible à classer correctement, utilise la catégorie "Inconnu".
6. Si aucun déchet identifiable n'est présent dans l'image,
   retourne un déchet avec la catégorie "Inconnu".
7. Pour plusieurs déchets clairement visibles, retourne chaque déchet séparément.
8. Le champ "confiance" doit être une estimation entre 0 et 1.
9. N'invente jamais une information qui n'est pas visible.
10. Retourne uniquement les informations demandées par le schéma de réponse.

CATÉGORIES AUTORISÉES :

- Plastique
- Papier
- Carton
- Verre
- Métal
- Textile
- Bois
- Déchet organique
- Déchet électronique
- Pile et batterie
- Déchet médical
- Déchet dangereux
- Huile usagée
- Déchet de construction
- Déchet vert
- Caoutchouc
- Cuir
- Déchet composite
- Déchet non recyclable
- Inconnu
"""


def analyser_image_avec_gemini(
    image_bytes: bytes,
    mime_type: str
) -> GeminiAnalysisResponse:

    # Nombre de tentatives en cas d'indisponibilité temporaire de Gemini.
    nombre_tentatives = 4

    for tentative in range(nombre_tentatives):

        try:
            print(
                f"🔎 Analyse Gemini - tentative "
                f"{tentative + 1}/{nombre_tentatives}"
            )

            # ---------------------------------------------------------
            # Conversion de l'image en Base64.
            # L'Interactions API attend les images sous cette forme.
            # ---------------------------------------------------------
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # ---------------------------------------------------------
            # Appel de l'Interactions API.
            #
            # Contrairement à generate_content(), on ne transmet pas
            # directement types.Part.from_bytes().
            #
            # L'Interactions API attend des contenus avec un champ
            # "type", notamment "text" et "image".
            # ---------------------------------------------------------
            interaction = gemini_client.interactions.create(
                model=GEMINI_MODEL,
                input=[
                    {
                        "type": "text",
                        "text": GEMINI_PROMPT,
                    },
                    {
                        "type": "image",
                        "data": image_base64,
                        "mime_type": mime_type,
                    },
                ],

                # -----------------------------------------------------
                # On demande à Gemini de retourner du JSON.
                # Le schéma Pydantic de notre application est fourni
                # comme schéma de réponse.
                # -----------------------------------------------------
               response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": GeminiAnalysisResponse.model_json_schema(),
                },

               
            )

            # ---------------------------------------------------------
            # Récupération du texte produit par Gemini.
            # ---------------------------------------------------------
            # L'Interactions API expose directement le texte final.
            texte_reponse = interaction.output_text

            if not texte_reponse:
                raise ValueError("Gemini a retourné une réponse vide.")

            print("📦 Réponse Gemini :", texte_reponse)

            # Validation du JSON reçu avec notre modèle Pydantic.
            resultat = GeminiAnalysisResponse.model_validate_json(
                texte_reponse
            )

            print("✅ Analyse Gemini réussie.")

            return resultat

        except Exception as erreur:

            message_erreur = str(erreur)

            print("❌ Erreur Gemini :", message_erreur)

            # ---------------------------------------------------------
            # Certaines erreurs sont temporaires :
            # - 503 : service momentanément indisponible
            # - 429 : trop de requêtes / quota
            # - UNAVAILABLE
            # - RESOURCE_EXHAUSTED
            # ---------------------------------------------------------
            erreur_temporaire = (
                "503" in message_erreur
                or "UNAVAILABLE" in message_erreur
                or "429" in message_erreur
                or "RESOURCE_EXHAUSTED" in message_erreur
            )

            # Les autres erreurs doivent être remontées immédiatement.
            if not erreur_temporaire:
                raise

            # Si toutes les tentatives ont échoué, on arrête.
            if tentative == nombre_tentatives - 1:

                print(
                    "❌ Gemini reste indisponible après "
                    "plusieurs tentatives."
                )

                raise

            # Attente progressive avant une nouvelle tentative :
            # 1 s → 2 s → 4 s...
            temps_attente = 2 ** tentative

            print(
                f"⏳ Gemini indisponible. "
                f"Nouvelle tentative dans "
                f"{temps_attente} seconde(s)..."
            )

            time.sleep(temps_attente)

    # Sécurité : cette ligne ne devrait normalement jamais être atteinte.
    raise RuntimeError(
        "Impossible d'obtenir une réponse de Gemini."
    )