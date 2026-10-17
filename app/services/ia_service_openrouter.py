import base64
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

import httpx

from app.core.config import OPENROUTER_API_KEY
from app.schemas.gemini_schema import GeminiAnalysisResponse

logger = logging.getLogger(__name__)

# =============================================================
# CONFIGURATION OPENROUTER
# =============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELES_URL = "https://openrouter.ai/api/v1/models"

# Routeur gratuit d'OpenRouter : choisit lui-même un modèle gratuit
# qui accepte les images. Il fait partie de chaque série d'essais.
MODELE_ROUTEUR = "openrouter/free"

# Modèle principal OPTIONNEL, à définir dans le .env (variable
# OPENROUTER_MODEL). Exemple : un modèle vision payant très bon
# marché pour une analyse fiable. Laisse vide pour rester gratuit.
MODELE_PRINCIPAL = os.getenv("OPENROUTER_MODEL", "").strip()

# Modèles gratuits recommandés. Les noms changent souvent sur
# OpenRouter : vérifie-les sur
# https://openrouter.ai/models?input_modalities=image (filtre "Free").
MODELES_SECOURS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-super-vl:free",
]

# Modèles à ne jamais utiliser :
# - modération de contenu (safety, guard, moderation)
# - modèles réservés aux agents (inkling) : erreur 403
# - génération de musique (lyria) : ne sait pas lire une image
MOTS_EXCLUS = (
    "safety",
    "guard",
    "moderation",
    "note-preview",
    "inkling",
    "lyria",
)

# -------------------------------------------------------------
# BUDGET DE TEMPS
# -------------------------------------------------------------
# Durée maximale de toute l'analyse. Django attend 60 s : on garde
# de la marge pour l'envoi de l'image et la validation.
DELAI_TOTAL = 40.0

# Temps minimum restant pour lancer une nouvelle série d'essais.
DELAI_MINIMUM = 8.0

# Nombre de modèles interrogés EN MÊME TEMPS (le plus rapide gagne).
NB_PARALLELE = 4

# Nombre maximal de séries d'essais par analyse.
NB_VAGUES = 2

# Pauses appliquées aux modèles qui ont échoué (secondes).
PAUSE_LIMITE = 3 * 60            # 429 ou délai dépassé : modèle saturé
PAUSE_INCOMPATIBLE = 10 * 60     # 400/402/403/404 : modèle non adapté
PAUSE_REPONSE_INVALIDE = 2 * 60  # réponse vide ou JSON cassé
PAUSE_COURTE = 60                # panne réseau ou fournisseur

# Rafraîchissement de la liste des modèles gratuits (secondes).
DUREE_CACHE_MODELES = 60 * 60

# -------------------------------------------------------------
# RÉSULTAT PAR DÉFAUT
# -------------------------------------------------------------
# Si aucun modèle ne répond à temps, on renvoie un résultat
# "Inconnu" clairement indiqué, au lieu d'une erreur 503 : l'utilisateur
# ne voit plus de message d'erreur et peut simplement refaire un scan.
# Mets IA_RESULTAT_PAR_DEFAUT=false dans le .env pour revenir
# à une vraie erreur.
RESULTAT_PAR_DEFAUT = os.getenv(
    "IA_RESULTAT_PAR_DEFAUT", "true"
).strip().lower() in ("1", "true", "yes", "oui")

TEXTE_PAR_DEFAUT = "Déchet non identifié (analyse indisponible, réessayez)"

# -------------------------------------------------------------
# ÉTAT INTERNE
# -------------------------------------------------------------
_cache_modeles: dict = {"liste": []}

# Modèle -> instant (timestamp) jusqu'auquel on évite de l'utiliser.
_pauses: dict[str, float] = {}
_verrou = threading.Lock()

# Client HTTP partagé :
# - force IPv4 (évite "Network is unreachable" dans Docker) ;
# - retente automatiquement les erreurs de connexion ;
# - réutilise les connexions (plus rapide).
_client = httpx.Client(
    transport=httpx.HTTPTransport(retries=2, local_address="0.0.0.0"),
)

# =============================================================
# CATÉGORIES AUTORISÉES
# =============================================================

CATEGORIES_DECHETS = {
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
}

# =============================================================
# PROMPT D'ANALYSE
# =============================================================

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


# =============================================================
# ERREUR OPENROUTER (avec son code HTTP)
# =============================================================

class ErreurOpenRouter(RuntimeError):
    """Erreur renvoyée par OpenRouter, avec le code HTTP associé."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# =============================================================
# PAUSE DES MODÈLES SATURÉS
# =============================================================

def _en_pause(modele: str) -> bool:
    """Indique si un modèle est temporairement mis de côté."""
    with _verrou:
        return _pauses.get(modele, 0.0) > time.time()


def _mettre_en_pause(modele: str, secondes: int) -> None:
    """Évite d'utiliser ce modèle pendant quelques minutes."""
    with _verrou:
        _pauses[modele] = time.time() + secondes
    logger.info("Modèle %s mis en pause %s s.", modele, secondes)


# =============================================================
# OUTILS
# =============================================================

def _extraire_json(texte: str) -> str:
    """Extrait le JSON si le modèle ajoute du texte autour."""

    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("La réponse de l'IA est vide.")

    texte = texte.strip()

    # Retire les balises Markdown éventuelles.
    if texte.startswith("```"):
        lignes = texte.splitlines()
        if lignes and lignes[0].startswith("```"):
            lignes = lignes[1:]
        if lignes and lignes[-1].strip() == "```":
            lignes = lignes[:-1]
        texte = "\n".join(lignes).strip()

    debut = texte.find("{")
    fin = texte.rfind("}")

    if debut == -1 or fin < debut:
        raise ValueError("Aucun JSON valide trouvé dans la réponse.")

    return texte[debut:fin + 1]


def _resultat_par_defaut() -> GeminiAnalysisResponse:
    """Résultat neutre renvoyé quand aucun modèle n'a répondu."""
    return GeminiAnalysisResponse.model_validate({
        "dechets": [
            {
                "objet": TEXTE_PAR_DEFAUT,
                "categorie": "Inconnu",
                "confiance": 0.0,
            }
        ]
    })


# =============================================================
# LISTE DES MODÈLES GRATUITS AVEC VISION (EN ARRIÈRE-PLAN)
# =============================================================
# La liste est récupérée par un thread séparé, jamais pendant
# l'analyse d'une image : cela économise plusieurs secondes.

def _rafraichir_liste_modeles() -> bool:
    """Récupère les modèles gratuits qui acceptent les images."""

    try:
        reponse = _client.get(
            OPENROUTER_MODELES_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=10.0,
        )
        reponse.raise_for_status()

        candidats = []

        for modele in reponse.json().get("data", []):
            identifiant = modele.get("id", "")

            # Uniquement les modèles explicitement gratuits.
            if not identifiant.endswith(":free"):
                continue

            # Ignore les modèles inadaptés (modération, musique...).
            if any(mot in identifiant for mot in MOTS_EXCLUS):
                continue

            architecture = modele.get("architecture", {})
            entrees = architecture.get("input_modalities") or []
            sorties = architecture.get("output_modalities") or []

            # Doit accepter les images et répondre par du texte seul.
            if "image" not in entrees:
                continue
            if sorties and set(sorties) != {"text"}:
                continue

            # Les modèles de raisonnement (lents) passent en dernier,
            # puis tri par taille de contexte décroissante.
            candidats.append((
                "reasoning" in identifiant,
                -(modele.get("context_length") or 0),
                identifiant,
            ))

        candidats.sort()
        liste = [identifiant for _, _, identifiant in candidats]

        if liste:
            _cache_modeles["liste"] = liste
            logger.info(
                "%s modèles gratuits avec vision trouvés.", len(liste)
            )
            return True

    except Exception as erreur:
        logger.warning(
            "Liste des modèles indisponible : %s", str(erreur)[:200]
        )

    return False


def _boucle_rafraichissement() -> None:
    """Met à jour la liste des modèles toutes les heures."""
    while True:
        reussi = _rafraichir_liste_modeles()
        time.sleep(DUREE_CACHE_MODELES if reussi else 300)


# Démarre le rafraîchissement dès le chargement du module.
threading.Thread(
    target=_boucle_rafraichissement,
    name="rafraichissement-modeles",
    daemon=True,
).start()


def _liste_a_essayer(autoriser_pauses: bool = False) -> list[str]:
    """
    Construit la liste des modèles à interroger en parallèle, en
    sautant ceux qui sont en pause. Le routeur gratuit est toujours
    inclus s'il n'est pas en pause.
    """

    ordre = []
    if MODELE_PRINCIPAL:
        ordre.append(MODELE_PRINCIPAL)
    ordre += MODELES_SECOURS
    ordre += _cache_modeles["liste"]

    # Supprime les doublons en gardant l'ordre.
    vus = set()
    candidats = []
    for identifiant in ordre:
        if identifiant in vus or identifiant == MODELE_ROUTEUR:
            continue
        vus.add(identifiant)
        candidats.append(identifiant)

    disponibles = [m for m in candidats if not _en_pause(m)]
    liste = disponibles[:NB_PARALLELE - 1]

    if not _en_pause(MODELE_ROUTEUR):
        liste.append(MODELE_ROUTEUR)

    # Première série seulement : si tout est en pause, on réessaie
    # quand même les premiers modèles plutôt que d'abandonner.
    if not liste and autoriser_pauses:
        liste = candidats[:NB_PARALLELE - 1] + [MODELE_ROUTEUR]

    return liste


# =============================================================
# APPEL À OPENROUTER
# =============================================================

def _appeler_openrouter(
    modele: str,
    image_bytes: bytes,
    mime_type: str,
    delai: float,
) -> str:
    """Envoie l'image à un modèle compatible vision sur OpenRouter."""

    if not OPENROUTER_API_KEY:
        raise ValueError("La clé OPENROUTER_API_KEY est absente.")

    type_image = mime_type or "image/jpeg"

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{type_image};base64,{image_b64}"

    reponse = _client.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": modele,
            "temperature": 0.1,
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT_ANALYSE,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            },
                        },
                    ],
                }
            ],
        },
        timeout=httpx.Timeout(delai, connect=6.0),
    )

    if reponse.status_code != 200:
        # Le texte renvoyé ne contient jamais la clé API.
        raise ErreurOpenRouter(
            f"OpenRouter HTTP {reponse.status_code}: "
            f"{reponse.text[:300]}",
            status_code=reponse.status_code,
        )

    donnees = reponse.json()

    # OpenRouter peut renvoyer une erreur avec un code HTTP 200.
    if "error" in donnees:
        erreur = donnees["error"]
        code = erreur.get("code") if isinstance(erreur, dict) else None
        raise ErreurOpenRouter(
            f"OpenRouter erreur: {str(erreur)[:300]}",
            status_code=code if isinstance(code, int) else None,
        )

    choix = donnees.get("choices", [])

    if not choix:
        raise ValueError("OpenRouter n'a retourné aucun choix.")

    contenu = choix[0].get("message", {}).get("content")

    # Certains modèles peuvent renvoyer le contenu sous forme de liste.
    if isinstance(contenu, list):
        morceaux = [
            element.get("text", "")
            for element in contenu
            if isinstance(element, dict)
            and element.get("type") == "text"
        ]
        contenu = "\n".join(morceaux)

    if not isinstance(contenu, str) or not contenu.strip():
        raise ValueError("OpenRouter a retourné une réponse vide.")

    return contenu


# =============================================================
# VALIDATION DU RÉSULTAT
# =============================================================

def _valider_resultat(texte: str) -> GeminiAnalysisResponse:
    """Valide le JSON retourné par le modèle."""

    resultat = GeminiAnalysisResponse.model_validate_json(
        _extraire_json(texte)
    )

    if not resultat.dechets:
        raise ValueError("L'IA n'a retourné aucun élément.")

    return resultat


# =============================================================
# ESSAI D'UN MODÈLE (exécuté dans un thread)
# =============================================================

def _tenter_modele(
    modele: str,
    image_bytes: bytes,
    mime_type: str,
    delai: float,
) -> tuple[str, GeminiAnalysisResponse]:
    """
    Interroge un modèle et valide sa réponse. En cas d'échec, met le
    modèle en pause selon la cause, puis relève l'erreur.
    """

    debut = time.monotonic()

    try:
        logger.info("Analyse avec %s (délai %.0f s)", modele, delai)

        texte = _appeler_openrouter(modele, image_bytes, mime_type, delai)
        resultat = _valider_resultat(texte)

        logger.info(
            "%s a répondu en %.1f s", modele, time.monotonic() - debut
        )
        return modele, resultat

    except ErreurOpenRouter as erreur:
        code = erreur.status_code
        logger.warning("Échec de %s : %s", modele, str(erreur)[:300])

        if code == 429:
            _mettre_en_pause(modele, PAUSE_LIMITE)
        elif code in (400, 402, 403, 404):
            _mettre_en_pause(modele, PAUSE_INCOMPATIBLE)
        elif code != 401:
            _mettre_en_pause(modele, PAUSE_COURTE)
        raise

    except httpx.TimeoutException:
        logger.warning("Délai dépassé pour %s.", modele)
        _mettre_en_pause(modele, PAUSE_LIMITE)
        raise

    except httpx.TransportError as erreur:
        logger.warning("Erreur réseau avec %s : %s", modele, erreur)
        _mettre_en_pause(modele, PAUSE_COURTE)
        raise

    except Exception as erreur:
        # Réponse vide, JSON invalide ou tronqué...
        logger.warning("Échec de %s : %s", modele, str(erreur)[:300])
        _mettre_en_pause(modele, PAUSE_REPONSE_INVALIDE)
        raise


# =============================================================
# FONCTION PRINCIPALE
# =============================================================

def analyser_image_avec_ia(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> GeminiAnalysisResponse:
    """
    Analyse une image avec plusieurs modèles OpenRouter EN PARALLÈLE :
    le premier modèle qui répond avec un JSON valide gagne.

    - Si tous échouent, une deuxième série est tentée avec d'autres
      modèles tant qu'il reste du temps.
    - Au bout de DELAI_TOTAL secondes, renvoie un résultat par défaut
      (ou lève une erreur si IA_RESULTAT_PAR_DEFAUT=false).
    """

    if not image_bytes:
        raise ValueError("L'image fournie est vide.")

    if not OPENROUTER_API_KEY:
        raise ValueError("La clé OPENROUTER_API_KEY est absente.")

    debut = time.monotonic()
    derniere_erreur = None

    for vague in range(NB_VAGUES):

        restant = DELAI_TOTAL - (time.monotonic() - debut)
        if restant < DELAI_MINIMUM:
            break

        modeles = _liste_a_essayer(autoriser_pauses=(vague == 0))
        if not modeles:
            break

        logger.info(
            "Série %s/%s : %s (%.0f s restantes)",
            vague + 1, NB_VAGUES, modeles, restant,
        )

        executeur = ThreadPoolExecutor(max_workers=len(modeles))
        futurs = [
            executeur.submit(
                _tenter_modele, modele, image_bytes, mime_type, restant
            )
            for modele in modeles
        ]

        try:
            for futur in as_completed(futurs, timeout=restant):
                try:
                    modele, resultat = futur.result()

                except ErreurOpenRouter as erreur:
                    derniere_erreur = erreur
                    # Clé invalide : inutile de continuer.
                    if erreur.status_code == 401:
                        raise
                    continue

                except Exception as erreur:
                    derniere_erreur = erreur
                    continue

                logger.info(
                    "Analyse réussie avec %s en %.1f s au total",
                    modele,
                    time.monotonic() - debut,
                )
                return resultat

        except FuturesTimeout:
            logger.warning("Délai de la série %s dépassé.", vague + 1)

        finally:
            # Ne pas attendre les modèles encore en cours.
            executeur.shutdown(wait=False, cancel_futures=True)

    logger.error(
        "Aucun modèle n'a répondu. Dernière erreur : %s", derniere_erreur
    )

    if RESULTAT_PAR_DEFAUT:
        return _resultat_par_defaut()

    raise RuntimeError(
        "Impossible d'analyser l'image avec les modèles disponibles. "
        f"Dernière erreur : {derniere_erreur}"
    ) from derniere_erreur