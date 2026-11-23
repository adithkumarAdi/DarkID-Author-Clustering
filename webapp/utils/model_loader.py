"""
DarkID Model Loader Module
==========================
Loads and caches saved StandardScaler, MiniBatchKMeans, and config artifacts
from stage4_svd/stylometric_only/final_model/ without modifying any disk files.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np

# Root path for final model artifacts
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FINAL_MODEL_DIR = PROJECT_DIR / "stage4_svd" / "stylometric_only" / "final_model"

EXPERIMENT_PACKAGES = {
    "random_holdout": "Random Holdout (Balanced Corpus Cross-Validation)",
    "unseen_source": "Unseen Source (Strict Cross-Domain Generalization)",
}

# Rich stylometric personas derived from empirical cluster centroids
CLUSTER_PERSONAS = {
    "random_holdout": {
        0: {
            "name": "Narrative / Experiential",
            "badge": "Cluster 0",
            "color": "#38BDF8",  # Cyan
            "description": "Marked by elevated personal pronoun usage and balanced prepositions. Common in first-person retrospectives, personal accounts, and narrative storytelling.",
            "dominant_traits": ["High Pronouns (16.6%)", "Moderate Prepositions (13.6%)", "Low Auxiliary Verbs (1.1%)"]
        },
        1: {
            "name": "Quantitative / Enumerative",
            "badge": "Cluster 1",
            "color": "#F59E0B",  # Amber
            "description": "Characterized by exceptionally high quantifier density (more than 2x the corpus mean). Typical of trade listings, itemized inventories, price listings, and volumetric descriptions.",
            "dominant_traits": ["Peak Quantifiers (4.56%)", "High Determiners (14.5%)", "Moderate Conjunctions (9.0%)"]
        },
        2: {
            "name": "Sparse / Telegraphic",
            "badge": "Cluster 2",
            "color": "#94A3B8",  # Muted Slate
            "description": "Depressed frequencies across pronouns, conjunctions, and determiners. Reflects minimalist, terse forum replies, code-like snippets, or clipped declarative statements.",
            "dominant_traits": ["Lowest Pronouns (7.5%)", "Low Conjunctions (6.3%)", "Low Determiners (10.1%)"]
        },
        3: {
            "name": "Expository / Structural",
            "badge": "Cluster 3",
            "color": "#A855F7",  # Purple
            "description": "Dense prepositional and conjunctional glue binding complex subordinate clauses. Characteristic of formal essays, technical instructions, OPSEC manuals, and detailed tutorials.",
            "dominant_traits": ["Peak Prepositions (16.1%)", "Peak Conjunctions (10.0%)", "High Determiners (14.7%)"]
        },
        4: {
            "name": "Dialogic / Opinionated",
            "badge": "Cluster 4",
            "color": "#EC4899",  # Rose Pink
            "description": "Prominent modal and auxiliary verb usage combined with heavy personal pronouns. Indicative of reactive forum debates, advisory comments, and subjective arguments.",
            "dominant_traits": ["Peak Auxiliary Verbs (2.30%)", "Peak Pronouns (18.4%)", "High Conjunctions (9.8%)"]
        }
    },
    "unseen_source": {
        0: {
            "name": "Informational / Referential",
            "badge": "Cluster 0",
            "color": "#38BDF8",
            "description": "High prepositional and determiner usage with restrained personal pronouns. Standard across objective informational posts and third-person commentary.",
            "dominant_traits": ["High Prepositions (14.7%)", "High Determiners (13.2%)", "Low Pronouns (11.9%)"]
        },
        1: {
            "name": "Narrative / Connective",
            "badge": "Cluster 1",
            "color": "#A855F7",
            "description": "Balanced high pronoun and conjunction rates. Characteristic of multi-clause forum discussions, blogs, and flowing communicative writing.",
            "dominant_traits": ["High Pronouns (16.4%)", "High Conjunctions (9.9%)", "High Prepositions (14.9%)"]
        },
        2: {
            "name": "Conversational Dialogue",
            "badge": "Cluster 2",
            "color": "#10B981",  # Emerald
            "description": "Prominent pronoun density with reduced determiner usage. Frequent in fast-paced peer-to-peer forum replies and direct messaging.",
            "dominant_traits": ["Peak Pronouns (18.4%)", "Low Determiners (11.2%)", "Moderate Prepositions (12.5%)"]
        },
        3: {
            "name": "Assertive / Evaluative",
            "badge": "Cluster 3",
            "color": "#F59E0B",
            "description": "Very high auxiliary verb usage and quantifier frequency. Indicates strong opinion pieces, advice-giving, or probabilistic claims.",
            "dominant_traits": ["Peak Auxiliary Verbs (2.90%)", "High Quantifiers (3.76%)", "High Pronouns (16.3%)"]
        },
        4: {
            "name": "Fragmentary / Minimalist",
            "badge": "Cluster 4",
            "color": "#94A3B8",
            "description": "Substantially lowered function-word frequency across all grammatical categories. Found in brief confirmations, isolated URLs, or abbreviated alerts.",
            "dominant_traits": ["Lowest Across All Categories (Sub-5% Function Words)"]
        }
    }
}

_MODEL_CACHE: Dict[str, Tuple[Any, Any, Dict[str, Any]]] = {}


def load_model_package(experiment: str = "random_holdout") -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load and return (model, scaler, config) for the specified experiment package.
    Converts cluster_centers_ to float64 in memory to satisfy scikit-learn Cython prediction.
    """
    if experiment not in EXPERIMENT_PACKAGES:
        raise ValueError(
            f"Invalid experiment '{experiment}'. Choose from: {list(EXPERIMENT_PACKAGES.keys())}"
        )

    if experiment in _MODEL_CACHE:
        return _MODEL_CACHE[experiment]

    pkg_dir = FINAL_MODEL_DIR / experiment
    if not pkg_dir.exists():
        raise FileNotFoundError(
            f"Model package directory missing: {pkg_dir}. "
            "Ensure stage4_svd/stylometric_only/final_model/ is intact."
        )

    model_path = pkg_dir / "model_k5.joblib"
    scaler_path = pkg_dir / "scaler.joblib"
    config_path = pkg_dir / "config.json"

    if not (model_path.exists() and scaler_path.exists() and config_path.exists()):
        raise FileNotFoundError(f"One or more artifact files missing in {pkg_dir}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # In-memory dtype fix: ensure cluster centers are float64 for Cython buffer alignment
    if hasattr(model, "cluster_centers_"):
        model.cluster_centers_ = model.cluster_centers_.astype(np.float64)

    _MODEL_CACHE[experiment] = (model, scaler, config)
    return model, scaler, config


def get_raw_cluster_centroids(experiment: str = "random_holdout") -> np.ndarray:
    """Return the unscaled (original percentage per 100 words) cluster centroids."""
    model, scaler, _ = load_model_package(experiment)
    return scaler.inverse_transform(model.cluster_centers_)
