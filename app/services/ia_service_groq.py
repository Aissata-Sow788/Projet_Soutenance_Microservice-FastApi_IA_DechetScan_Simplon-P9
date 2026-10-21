import base64
import logging
import os
import time

import httpx

from app.core.config import GROQ_API_KEY
from app.schemas.gemini_schema import GeminiAnalysisResponse

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Modèle vision de Groq (modifiable dans le .env via GROQ_MODEL).
MODELE = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()

# Budget de temps total et nombre d'essais.
DELAI_TOTAL = 40.0
NB_ESSAIS = 2

# true = renvoie "Inconnu" au lieu d'une erreur (à activer plus tard).
RESULTAT_PAR_DEFAUT = os.getenv(
    "IA_RESULTAT_PAR_DEFAUT", "false"
).strip().lower() in ("1", "true", "yes", "oui")

TEXTE_PAR_DEFAUT = "Déchet non identifié (analyse indisponible, réessayez)"

# Client partagé : IPv4 forcé (utile dans Docker) + reconnexion auto.
_client = httpx.Client(
    transport=httpx.HTTPTransport(retries=2, local_address="0.0.0.0"),
)

PROMPT_ANALYSE = """
Tu es le système d'analyse d'images de l'application DechetScan.

Analyse uniquement les déchets réellement visibles dans l'image.

RÈGLES :
1. N'invente jamais un objet absent de l'image.
2. Utilise uniquement les catégories autorisées.
3. Si l'objet est impossible à identifier, utilise "Inconnu".
4. Si aucun déchet n'est visible, retourne un seul item
   avec objet "Aucun déchet identifiable" et catégorie "Inconnu".
5. La confiance doit être un nombre entre 0 et 1.
6. Regroupe les objets identiques en un seul item
   (exemple : "canettes en aluminium").
7. Retourne au maximum 6 items, les plus visibles.
8. Réponds immédiatement, sans réfléchir à voix haute.

CATÉGORIES AUTORISÉES :
Plastique, Papier, Carton, Verre, Métal, Textile, Bois,
Déchet organique, Déchet électronique, Pile et batterie,
Déchet médical, Déchet dangereux, Huile usagée,
Déchet de construction, Déchet vert, Caoutchouc, Cuir,
Déchet composite, Déchet non recyclable, Inconnu.

FORMAT JSON STRICT :
{
  "dechets": [
    {
      "objet": "bouteille en plastique",
      "categorie": "Plastique",
      "confiance": 0.95
    }
  ]
}

Retourne uniquement le JSON, sans Markdown ni explication.
"""


class ErreurGroq(RuntimeError):
    """Erreur renvoyée par Groq, avec le code HTTP et le délai conseillé."""

    def __init__(self, message, status_code=None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _extraire_json(texte: str) -> str:
    """Extrait le JSON si le modèle ajoute du texte autour."""
    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("La réponse de l'IA est vide.")

    debut = texte.find("{")
    fin = texte.rfind("}")

    if debut == -1 or fin < debut:
        raise ValueError("Aucun JSON valide trouvé dans la réponse.")

    return texte[debut:fin + 1]


def _valider_resultat(texte: str) -> GeminiAnalysisResponse:
    """Valide le JSON retourné par le modèle."""
    resultat = GeminiAnalysisResponse.model_validate_json(
        _extraire_json(texte)
    )
    if not resultat.dechets:
        raise ValueError("L'IA n'a retourné aucun élément.")
    return resultat


def _resultat_par_defaut() -> GeminiAnalysisResponse:
    """Résultat neutre quand l'analyse échoue."""
    return GeminiAnalysisResponse.model_validate({
        "dechets": [{
            "objet": TEXTE_PAR_DEFAUT,
            "categorie": "Inconnu",
            "confiance": 0.0,
        }]
    })


def _appeler_groq(image_bytes: bytes, mime_type: str, delai: float) -> str:
    """Envoie l'image au modèle vision de Groq."""

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type or 'image/jpeg'};base64,{image_b64}"

    reponse = _client.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODELE,
            "temperature": 0.1,
            "max_completion_tokens": 800,
            # Désactive le raisonnement : réponse plus rapide.
            "reasoning_effort": "none",
            # Force une réponse JSON.
            "response_format": {"type": "json_object"},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_ANALYSE},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        },
        timeout=httpx.Timeout(delai, connect=6.0),
    )

    if reponse.status_code != 200:
        retry = reponse.headers.get("retry-after")
        raise ErreurGroq(
            f"Groq HTTP {reponse.status_code}: {reponse.text[:300]}",
            status_code=reponse.status_code,
            retry_after=float(retry) if retry and retry.replace(".", "", 1).isdigit() else None,
        )

    choix = reponse.json().get("choices", [])
    if not choix:
        raise ValueError("Groq n'a retourné aucun choix.")

    contenu = choix[0].get("message", {}).get("content")
    if not isinstance(contenu, str) or not contenu.strip():
        raise ValueError("Groq a retourné une réponse vide.")

    return contenu


def analyser_image_avec_ia(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> GeminiAnalysisResponse:
    """Analyse une image avec Groq (2 essais maximum)."""

    if not image_bytes:
        raise ValueError("L'image fournie est vide.")
    if not GROQ_API_KEY:
        raise ValueError("La clé GROQ_API_KEY est absente.")

    debut = time.monotonic()
    derniere_erreur = None

    for essai in range(NB_ESSAIS):

        restant = DELAI_TOTAL - (time.monotonic() - debut)
        if restant < 5:
            break

        try:
            logger.info("Analyse Groq %s (essai %s)", MODELE, essai + 1)
            texte = _appeler_groq(image_bytes, mime_type, restant)
            return _valider_resultat(texte)

        except ErreurGroq as erreur:
            derniere_erreur = erreur
            logger.warning("Échec Groq : %s", str(erreur)[:300])

            # Clé ou requête invalide : inutile de réessayer.
            if erreur.status_code in (400, 401, 403, 404):
                break

            # 429 ou panne passagère : courte attente puis nouvel essai.
            time.sleep(min(erreur.retry_after or 2.0, 8.0))

        except Exception as erreur:
            # Délai dépassé, réseau, JSON invalide...
            derniere_erreur = erreur
            logger.warning("Échec Groq : %s", str(erreur)[:300])

    logger.error("Analyse Groq impossible : %s", derniere_erreur)

    if RESULTAT_PAR_DEFAUT:
        return _resultat_par_defaut()

    raise RuntimeError(
        f"Impossible d'analyser l'image avec Groq. Dernière erreur : {derniere_erreur}"
    ) from derniere_erreur