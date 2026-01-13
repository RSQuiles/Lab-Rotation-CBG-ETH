import logging
import random
import warnings

import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import check_array
from sklearn.mixture import GaussianMixture #added

logger = logging.getLogger(__name__)


def _compute_clustering_gmm(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    #added
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    gmm.fit(X)
    return gmm.predict(X)


def nmi_ari_cluster_labels_gmm(X: np.ndarray, labels: np.ndarray, seed: int = 42) -> dict[str, float]:
    #added
    """Compute nmi and ari between GMM clusters and labels.

    Parameters
    ----------
    X
        Array of shape (n_cells, n_features).
    labels
        Array of shape (n_cells,) representing label values

    Returns
    -------
    nmi
        Normalized mutual information score
    ari
        Adjusted rand index score
    """
    X = check_array(X, accept_sparse=False, ensure_2d=True)
    n_clusters = len(np.unique(labels))
    labels_pred = _compute_clustering_gmm(X, n_clusters, seed)
    nmi = normalized_mutual_info_score(labels, labels_pred, average_method="arithmetic")
    ari = adjusted_rand_score(labels, labels_pred)

    return {"nmi": nmi, "ari": ari}


