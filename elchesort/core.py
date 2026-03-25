"""
ElcheSort: A simple spike sorting library for single-channel sorting 
with template matching and isosplit clustering.

Core module

2026 Aitor Morales-Gregorio
"""
import torch
from torch.nn.functional import conv1d, max_pool1d
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.decomposition import PCA
from isosplit6 import isosplit6
from pathlib import Path


def match_templates(signal, templates,
                    detection_threshold, 
                    exclusion_bins=None):
    """
    Single-pass template matching for spike detection.
    Based on Kilosort's matching, simplified for single channel.
    
    Parameters
    ----------
    signal : torch.Tensor
        Single channel data with shape (n_timepoints,).
    templates : torch.Tensor
        Template waveforms with shape (n_templates, nt).
    detection_threshold : float
        Detection threshold (squared internally).
    exclusion_bins : int or None
        Minimum spacing between detections in samples. Avoids multiple
        detection of the same waveform across templates.
        Default: None, uses the template length.
    
    Returns
    -------
    times : torch.Tensor
        Spike times as sample indices, shape (n_spikes,).
    template_ids : torch.Tensor
        Which template matched each spike, shape (n_spikes,).
    """
    nt = templates.shape[1]

    if exclusion_bins is None:
        exclusion_bins = nt
    
    # Template squared norms for normalization
    nm = (templates**2).sum(-1)  # (n_templates,)
    
    # Unfold signal into sliding windows (no copy, just view)
    # Matrix multiply: MUCH faster on GPU
    # Add padding to match conv1d output size
    signal_unfolded = signal.unfold(0, nt, 1)  # (n_timepoints - nt + 1, nt)
    B = torch.mm(templates, signal_unfolded.T)  # (n_templates, n_timepoints - nt + 1)

    # Compute Cf only on valid region, pad zeros
    nm_inv = (1.0 / nm).unsqueeze(-1)  # Pre-compute inverse
    Cf_valid = torch.relu(B)**2 * nm_inv
    Cf = torch.nn.functional.pad(Cf_valid, (nt//2, nt//2), value=0)
    
    # Best template per timepoint
    Cfmax, imax = torch.max(Cf, 0)  # (n_timepoints,), (n_timepoints,)
    
    # Non-maximum suppression
    Cmax = max_pool1d(
        Cfmax.unsqueeze(0).unsqueeze(0),  # (1, 1, n_timepoints)
        (2*exclusion_bins+1), stride=1, padding=exclusion_bins
    )  # (1, 1, n_timepoints)
    
    # Threshold + local max (threshold is squared)
    cnd1 = Cmax[0, 0] > detection_threshold**2
    cnd2 = torch.abs(Cmax[0, 0] - Cfmax) < 1e-9
    xs = torch.nonzero(cnd1 * cnd2)  # (n_spikes, 1)
    
    if len(xs) == 0:
        return torch.tensor([], device=signal.device, dtype=torch.float32), \
               torch.tensor([], device=signal.device, dtype=torch.float32)
    
    times = xs[:, 0]  # (n_spikes,) - spike times
    template_ids = imax[times]  # (n_spikes,) - template ids
    
    return times, template_ids


def isosplit_on_features(features):
    """
    Cluster spike features using ISO-SPLIT with hierarchical subdivision.
    
    Runs isosplit6 on the feature matrix, then recursively splits the
    resulting clusters into two groups via hierarchical clustering on
    centroids until convergence.
    
    Parameters
    ----------
    features : np.ndarray
        PCA features with shape (n_spikes, n_components).
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels starting from 1, shape (n_spikes,).
    """
    n_spikes = len(features)
    if n_spikes == 0:
        return np.zeros(0, dtype=np.int32)
    
    # Cluster with isosplit6
    labels = isosplit6(features.astype(np.float32))
    
    # Stop if 0 or 1 clusters found
    n_clusters = int(np.max(labels)) if len(labels) > 0 else 0
    if n_clusters <= 1:
        return labels
    
    # Compute cluster centroids
    centroids = np.zeros((n_clusters, features.shape[1]), dtype=np.float32)
    for cluster_id in range(1, n_clusters + 1):
        centroids[cluster_id - 1] = np.median(features[labels == cluster_id], axis=0)
    
    # Hierarchical clustering to split centroids into 2 groups
    centroid_distances = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    linkage_matrix = linkage(squareform(centroid_distances), method='single')
    group_assignments = cut_tree(linkage_matrix, n_clusters=2)
    
    # Which cluster IDs belong to each group
    group1_cluster_ids = np.where(group_assignments == 0)[0] + 1
    group2_cluster_ids = np.where(group_assignments == 1)[0] + 1
    
    # Split features into two groups
    group1_mask = np.isin(labels, group1_cluster_ids)
    group2_mask = np.isin(labels, group2_cluster_ids)
    
    # Recursively cluster each group
    labels_group1 = isosplit_on_features(features[group1_mask])
    labels_group2 = isosplit_on_features(features[group2_mask])
    
    # Combine labels (offset group2 to avoid overlap)
    max_label_group1 = int(np.max(labels_group1)) if len(labels_group1) > 0 else 0
    result_labels = np.zeros(n_spikes, dtype=np.int32)
    result_labels[group1_mask] = labels_group1
    result_labels[group2_mask] = labels_group2 + max_label_group1
    
    return result_labels


def spike_sort_channel(signal, templates='human', 
                       detection_threshold=12, exclusion_bins=41,
                       amplitude_threshold=100, n_pcs=12,
                       device=torch.device('cpu')):
    """
    Sort spikes from a single channel using template matching and ISO-SPLIT clustering.
    
    Detects spikes by matching against a set of waveform templates, extracts
    aligned waveforms, projects them into a low-dimensional PCA space, and
    clusters them with ISO-SPLIT to identify putative neural units.
    
    Parameters
    ----------
    signal : np.ndarray or torch.Tensor
        Preprocessed single-channel voltage trace with shape (n_samples,).
    templates : str or np.ndarray or torch.Tensor
        Spike templates for detection. Pass 'human' or 'macaque' to use
        built-in templates, or provide a custom array with shape (n_templates, 61).
    detection_threshold : float
        Minimum template-match score to flag a spike.
    exclusion_bins : int
        Refractory window in samples; detections closer than this are merged.
    amplitude_threshold : float
        Maximum allowed spike amplitude; larger values are discarded.
    n_pcs : int
        Number of principal components for waveform feature extraction.
    device : torch.device
        Device for computation (e.g., 'cpu' or 'cuda').
    
    Returns
    -------
    spike_time_idx : np.ndarray
        Sample indices of each detected spike, shape (n_spikes,).
    clusters : np.ndarray
        Cluster label for each spike, shape (n_spikes,).
    waveforms : np.ndarray
        Aligned waveform snippets, shape (n_spikes, 61).
    wv_features : np.ndarray
        PCA projection of each waveform, shape (n_spikes, n_pcs).
    
    If fewer than 2 * n_pcs spikes are detected, all four outputs are
    empty lists.
    """

    # Template IO
    if isinstance(templates, str):
            if templates == 'human':
                templates = np.load(Path(__file__).parent / "sample_templates" / "human_waveform_templates.npy")
            elif templates == 'macaque':
                templates = np.load(Path(__file__).parent / "sample_templates" / "macaque_waveform_templates.npy")
            else:
                raise ValueError(f"Unknown template set '{templates}'. Use 'human', 'macaque', or pass an array.")

    # Cast to pytorch tensors for fast computation
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, device=device, dtype=torch.float32)
    else:
        signal = signal.to(device=device, dtype=torch.float32)
    if not isinstance(templates, torch.Tensor):
        templates = torch.tensor(templates, device=device, dtype=torch.float32)  # (n_templates, 61)
    else:
        templates = templates.to(device=device, dtype=torch.float32) # (n_templates, 61)

    # Robust normalization to units of noise (MAD-based)
    mad = torch.median(torch.abs(signal - torch.median(signal)))
    signal = signal / (mad * 1.4826)  # 1.4826 scales MAD to std for normal dist

    # Get spike times from template matching
    spike_time_idx, _ = match_templates(signal, templates,
                                        detection_threshold=detection_threshold,
                                        exclusion_bins=exclusion_bins)

    # Exit if too few spikes present
    if len(spike_time_idx) < n_pcs*2:
        return [], [], [], []
    
    # Remove spikes within 21 samples of start or 40 samples of end
    valid_spikes = (spike_time_idx >= 21) & (spike_time_idx < len(signal) - 40)
    spike_time_idx = spike_time_idx[valid_spikes]

    # Vectorized waveform extraction
    indices = spike_time_idx[:, None] + torch.arange(-21, 40, device=spike_time_idx.device)
    waveforms = signal[indices]  # (n_spikes, 61)

    # Fast pytorch PCA to compute features
    # waveforms_centered = waveforms - waveforms.mean(dim=0, keepdim=True)
    U, S, _ = torch.pca_lowrank(waveforms, q=n_pcs)
    wv_features = U * S  # Scores (n_spikes, n_pcs)
    wv_features = wv_features.cpu().numpy()

    # Clustering of units
    clusters = isosplit_on_features(wv_features)

    # Convert all outputs to numpy
    spike_time_idx = spike_time_idx.cpu().numpy()
    waveforms = waveforms.cpu().numpy()

    return spike_time_idx, clusters, waveforms, wv_features
