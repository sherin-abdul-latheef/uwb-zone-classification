import numpy as np

def compute_weighted_MD(X_test, X_train, train_weights):
    diff_tensor   = np.abs(X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :])
    
    
    weighted_diff = diff_tensor * train_weights[np.newaxis, :, :]
    weighted_MD   = np.sum(weighted_diff, axis=2)
    return weighted_MD

def get_top_k(sorted_indices, weighted_MD, y_train, k):
    
    top_k_indices   = sorted_indices[:, :k]
    top_k_labels    = y_train[top_k_indices]
    top_k_distances = np.take_along_axis(weighted_MD, top_k_indices, axis=1)
    return top_k_labels, top_k_distances


def compute_probabilities(top_k_labels, top_k_distances):
    eps          = 1e-8
    w            = 1.0 / (top_k_distances + eps)
    w_tilde      = w / np.sum(w, axis=1, keepdims=True)
    unique_zones = np.unique(top_k_labels)
    n_tests      = w_tilde.shape[0]
    P            = np.zeros((n_tests, len(unique_zones)))
    for j, zone in enumerate(unique_zones):
        mask     = (top_k_labels == zone)
        P[:, j]  = np.sum(w_tilde * mask, axis=1)
    return P, unique_zones