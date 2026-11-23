# DarkID — Unsupervised Stylometric Author Clustering

## 1. Project Overview

DarkID is an unsupervised machine learning project that explores
stylometric patterns in text to group documents with similar
writing characteristics.

The objective is to investigate whether documents can be grouped
based on linguistic writing patterns without using author labels
during clustering.

## 2. Methodology

The final baseline uses six stylometric features:

1. Auxiliary verbs frequency per 100 words
2. Conjunctions frequency per 100 words
3. Determiners frequency per 100 words
4. Prepositions frequency per 100 words
5. Pronouns frequency per 100 words
6. Quantifiers frequency per 100 words

The features are standardized using StandardScaler.

Clustering algorithm:
- MiniBatchKMeans

Selected number of clusters:
- K = 5

Random state:
- 42

## 3. Evaluation Experiments

Two evaluation protocols were used:

### Random Holdout
Evaluates clustering across a random document split.

### Unseen Source
Evaluates the model on documents from sources excluded
from its training partition.

These experiments assess different generalization settings.

## 4. Final Test Results

| Experiment | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|
| Random Holdout | 0.140539 | 1.714448 | 5230.510896 |
| Unseen Source | 0.132707 | 1.722753 | 5051.631931 |

Both models produced five active clusters.

### Cluster size observations

Random Holdout:
- Test documents: 40,199
- Smallest cluster: 3,218
- Largest cluster: 15,307

Unseen Source:
- Test documents: 61,740
- Smallest cluster: 30
- Largest cluster: 30,127

The unseen-source test has a substantial cluster imbalance.

## 5. Model Selection

K = 5 was selected over K = 10 and K = 20.

The K = 5 configuration achieved the strongest test
silhouette score in both evaluation experiments.

Source-dominance analysis also indicated better source
mixing for K = 5 compared with the larger cluster counts.

## 6. Enhanced Feature Experiment

A separate 20-dimensional stylometric experiment was conducted.

It was retained as an experimental branch rather than
selected as the final model.

The baseline 6D configuration was retained for the final
package based on the comparative evaluation.

## 7. Important Limitations

- Intrinsic clustering metrics measure cluster geometry,
  not ground-truth authorship accuracy.
- Source mixing does not prove common authorship.
- Cluster sizes are imbalanced, especially in the
  unseen-source test.
- The results should be interpreted as exploratory
  unsupervised document grouping, not verified author identity.

## 8. Final Model Package

Location:

stage4_svd/stylometric_only/final_model/

Each experiment package contains:

- model_k5.joblib
- scaler.joblib
- config.json

The packages were tested against the original models.
Predictions were identical during package verification.

## 9. Reproducibility

The following documentation artifacts are available
under stage5_documentation/:

- final_artifact_inventory.csv
- final_model_selection_report.txt
- final_model_checksums.csv
- README.md

The checksum manifest records SHA-256 fingerprints for
the packaged models, scalers, and configurations.

## 10. Future Work

- Develop a web interface for document submission and
  clustering visualization.
- Explore additional stylometric features.
- Investigate alternative clustering algorithms.
- Improve cluster balance and stability.
- Conduct further validation of authorship-related patterns.
