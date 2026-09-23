from enum import Enum

from pydantic import BaseModel, Field


# Liste officielle des catégories acceptées par DechetScan.
class WasteCategory(str, Enum):
    PLASTIQUE = "Plastique"
    PAPIER = "Papier"
    CARTON = "Carton"
    VERRE = "Verre"
    METAL = "Métal"
    TEXTILE = "Textile"
    BOIS = "Bois"
    ORGANIQUE = "Déchet organique"
    ELECTRONIQUE = "Déchet électronique"
    PILE_BATTERIE = "Pile et batterie"
    MEDICAL = "Déchet médical"
    DANGEREUX = "Déchet dangereux"
    HUILE = "Huile usagée"
    CONSTRUCTION = "Déchet de construction"
    VERT = "Déchet vert"
    CAOUTCHOUC = "Caoutchouc"
    CUIR = "Cuir"
    COMPOSITE = "Déchet composite"
    NON_RECYCLABLE = "Déchet non recyclable"
    INCONNU = "Inconnu"


# Représente un déchet détecté dans l'image.
class WasteDetection(BaseModel):
    categorie: WasteCategory = Field(
        description="Catégorie du déchet parmi les catégories autorisées."
    )

    objet: str = Field(
        description="Nom de l'objet identifié dans l'image."
    )

    confiance: float = Field(
        ge=0.0,
        le=1.0,
        description="Niveau de confiance estimé entre 0 et 1."
    )


# Réponse complète attendue de Gemini.
class GeminiAnalysisResponse(BaseModel):
    dechets: list[WasteDetection]