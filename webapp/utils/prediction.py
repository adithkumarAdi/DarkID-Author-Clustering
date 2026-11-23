"""
DarkID Prediction & Inference Module
====================================
Orchestrates text validation, stylometric extraction, feature scaling,
and MiniBatchKMeans cluster prediction for single documents and batch inputs.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from .feature_extraction import (
    extract_stylometric_features,
    get_feature_vector,
    FEATURE_NAMES,
    FEATURE_DISPLAY_NAMES,
)
from .model_loader import (
    load_model_package,
    CLUSTER_PERSONAS,
    get_raw_cluster_centroids,
)


def predict_single_document(
    text: str,
    experiment: str = "random_holdout",
) -> Dict[str, Any]:
    """
    Run the end-to-end DarkID inference pipeline on a single text document.

    Returns:
        Structured prediction result dictionary including predicted cluster,
        stylometric persona, distance breakdown, and input statistics.
    """
    # 1. Extract 6 stylometric features
    extraction = extract_stylometric_features(text)
    if not extraction["valid"]:
        return {
            "success": False,
            "error": extraction["error"],
            "stats": {
                "word_count": extraction["word_count"],
                "sentence_count": extraction["sentence_count"],
                "avg_word_length": extraction["avg_word_length"],
            },
            "features_raw": extraction["features"],
            "cluster_id": None,
            "persona": None,
            "distances": {},
        }

    raw_vector = get_feature_vector(extraction["features"])
    X_raw = np.asarray([raw_vector], dtype=np.float64)

    # 2. Load model and scaler
    model, scaler, config = load_model_package(experiment)

    # 3. Scale features using training-fitted StandardScaler
    X_scaled = scaler.transform(X_raw).astype(np.float64)

    # 4. Predict cluster
    cluster_id = int(model.predict(X_scaled)[0])

    # 5. Compute Euclidean distances to all 5 cluster centroids in scaled space
    dists = euclidean_distances(X_scaled, model.cluster_centers_)[0]
    distance_map = {i: float(round(dists[i], 4)) for i in range(len(dists))}

    # Identify runner-up cluster
    sorted_clusters = np.argsort(dists)
    runner_up = int(sorted_clusters[1]) if len(sorted_clusters) > 1 else cluster_id

    # 6. Retrieve persona metadata
    personas = CLUSTER_PERSONAS.get(experiment, {})
    persona = personas.get(cluster_id, {
        "name": f"Cluster {cluster_id}",
        "badge": f"Cluster {cluster_id}",
        "color": "#A855F7",
        "description": "Stylometric density grouping.",
        "dominant_traits": [],
    })

    return {
        "success": True,
        "cluster_id": cluster_id,
        "persona": persona,
        "runner_up_cluster": runner_up,
        "distances": distance_map,
        "features_raw": extraction["features"],
        "features_scaled": {
            col: float(X_scaled[0][i]) for i, col in enumerate(FEATURE_NAMES)
        },
        "stats": {
            "word_count": extraction["word_count"],
            "sentence_count": extraction["sentence_count"],
            "avg_word_length": extraction["avg_word_length"],
        },
        "warning": extraction.get("error"),
        "experiment": experiment,
    }


def predict_batch_documents(
    texts: List[str],
    doc_ids: Optional[List[str]] = None,
    experiment: str = "random_holdout",
) -> pd.DataFrame:
    """
    Run predictions over a collection of text documents and return a formatted DataFrame.
    """
    model, scaler, config = load_model_package(experiment)
    personas = CLUSTER_PERSONAS.get(experiment, {})

    if doc_ids is None or len(doc_ids) != len(texts):
        doc_ids = [f"DOC_{i+1:04d}" for i in range(len(texts))]

    records = []
    raw_feature_matrix = []
    valid_indices = []

    for idx, (doc_id, text) in enumerate(zip(doc_ids, texts)):
        res = extract_stylometric_features(text)
        if not res["valid"]:
            records.append({
                "Document ID": doc_id,
                "Status": "Error (Too Short or Empty)",
                "Word Count": res["word_count"],
                "Predicted Cluster": -1,
                "Cluster Persona": "Unassigned",
                "Nearest Distance": np.nan,
                **{FEATURE_DISPLAY_NAMES[col]: 0.0 for col in FEATURE_NAMES},
            })
        else:
            raw_vec = get_feature_vector(res["features"])
            raw_feature_matrix.append(raw_vec)
            valid_indices.append((idx, doc_id, res))

    if raw_feature_matrix:
        X_raw = np.asarray(raw_feature_matrix, dtype=np.float64)
        X_scaled = scaler.transform(X_raw).astype(np.float64)
        preds = model.predict(X_scaled)
        all_dists = euclidean_distances(X_scaled, model.cluster_centers_)

        for row_idx, (orig_idx, doc_id, res) in enumerate(valid_indices):
            c_id = int(preds[row_idx])
            persona_name = personas.get(c_id, {}).get("name", f"Cluster {c_id}")
            nearest_dist = float(round(all_dists[row_idx][c_id], 3))

            records.append({
                "Document ID": doc_id,
                "Status": "OK",
                "Word Count": res["word_count"],
                "Predicted Cluster": c_id,
                "Cluster Persona": persona_name,
                "Nearest Distance": nearest_dist,
                **{
                    FEATURE_DISPLAY_NAMES[col]: round(res["features"][col], 2)
                    for col in FEATURE_NAMES
                },
            })

    # Sort back by original appearance order
    df = pd.DataFrame(records)
    return df
