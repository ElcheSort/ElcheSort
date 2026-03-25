"""
ElcheSort: A simple spike sorting library for single-channel sorting 
with template matching and isosplit clustering.

Postprocessing module

2026 Aitor Morales-Gregorio
"""
import numpy as np
import scipy
import neo
import quantities as pq
from statsmodels.graphics.tsaplots import acf
import elephant


def detect_waveform_outliers(features, n_mad=10):
    """
    Detect outlier waveforms based on Mahalanobis distance in PCA space.
    
    Uses median absolute deviation (MAD) for a robust threshold.
    
    Parameters
    ----------
    features : np.ndarray
        PCA features with shape (n_spikes, n_components).
    n_mad : float
        Number of MADs from median to consider outlier. Default is 10
        (very conservative).
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask, True for good spikes, shape (n_spikes,).
    distances : np.ndarray
        Mahalanobis distance for each spike, shape (n_spikes,).
    threshold : float
        Distance threshold used for outlier detection.
    """
    features = features.reshape(features.shape[0], -1)
    
    # Robust center and covariance
    center = np.median(features, axis=0)
    cov = np.cov(features, rowvar=False)
    
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    
    # Mahalanobis distances
    diff = features - center
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    
    # MAD-based threshold
    med_dist = np.median(distances)
    mad = np.median(np.abs(distances - med_dist))
    threshold = med_dist + n_mad * mad * 1.4826  # 1.4826 scales MAD to std for normal dist
    
    mask = distances < threshold
    return mask, distances, threshold


def estimate_line_noise_from_spikes(spike_train, binsize=2*pq.ms, n_lags=100):
    """
    Detect whether a unit's firing is contaminated by line noise.
    
    Computes the autocorrelation histogram (ACH) and correlates it with
    shifted versions of itself at 50 Hz and 60 Hz periods. High correlation
    indicates line noise contamination.
    
    Parameters
    ----------
    spike_train : neo.SpikeTrain
        Spike train object containing spike times.
    binsize : pq.Quantity
        Bin size for the autocorrelation calculation. Default: 2 ms.
    n_lags : int
        Number of bins for the autocorrelation histogram. Default: 100.
    
    Returns
    -------
    corr_50hz : float
        Correlation of the ACH with a 50 Hz shift of itself.
    corr_60hz : float
        Correlation of the ACH with a 60 Hz shift of itself.
    """
    # Binning
    bst = elephant.conversion.BinnedSpikeTrain(spike_train, bin_size=binsize)
    spike_vec = bst.to_array()[0]

    # Calculate ACG
    autocorr = acf(spike_vec, nlags=n_lags)
    vector = autocorr[1:]
    shifted_50_vector = np.roll(vector, int(20/binsize.rescale('ms').magnitude))
    shifted_60_vector = np.roll(vector, int(100/6/binsize.rescale('ms').magnitude))

    # Before computing correlation, check for valid data
    if np.any(~np.isfinite(vector)) or np.any(~np.isfinite(shifted_50_vector)):
        corr_50hz = 0.0  # Default value when data is invalid
    else:
        corr_50hz, _ = scipy.stats.pearsonr(vector, shifted_50_vector)

    # Same for 60Hz
    if np.any(~np.isfinite(vector)) or np.any(~np.isfinite(shifted_60_vector)):
        corr_60hz = 0.0
    else:
        corr_60hz, _ = scipy.stats.pearsonr(vector, shifted_60_vector)


    return corr_50hz, corr_60hz


def waveform_snr(waveforms, noise_level, peak_sign="neg", peak_mode="extremum"):
    """
    Compute signal-to-noise ratio for a set of waveforms.
    
    Parameters
    ----------
    waveforms : np.ndarray
        Waveform matrix with shape (n_spikes, n_samples).
    noise_level : float
        Standard deviation of background noise.
    peak_sign : str
        Which peak to use: 'neg', 'pos', or 'both'.
    peak_mode : str
        How to compute amplitude: 'extremum', 'at_index', or 'peak_to_peak'.
    
    Returns
    -------
    snr : float
        Signal-to-noise ratio of the mean waveform.
    """
    assert peak_sign in ("neg", "pos", "both")
    assert peak_mode in ("extremum", "at_index", "peak_to_peak")
    
    mean_waveform = np.mean(waveforms, axis=0)
    
    if peak_mode == "extremum":
        if peak_sign == "neg":
            amplitude = np.abs(np.min(mean_waveform))
        elif peak_sign == "pos":
            amplitude = np.abs(np.max(mean_waveform))
        else:  # both
            amplitude = np.max(np.abs(mean_waveform))
    elif peak_mode == "at_index":
        center = len(mean_waveform) // 2
        amplitude = np.abs(mean_waveform[center])
    else:  # peak_to_peak
        amplitude = np.max(mean_waveform) - np.min(mean_waveform)
    
    return amplitude / noise_level

def presence_ratio(st, bin_size=10*pq.s):
    """
    Calculate the fraction of time bins where the unit fires at least once.
    
    Parameters
    ----------
    st : neo.SpikeTrain
        Spike train object.
    bin_size : pq.Quantity
        Size of time bins. Default: 10 s.
    
    Returns
    -------
    ratio : float
        Fraction of bins containing at least one spike, between 0 and 1.
    """
    h = elephant.statistics.time_histogram([st], bin_size).magnitude
    return np.count_nonzero(h) / len(h)

def generate_spiketrain_object(spike_times, waveforms, metadata, t_start, t_stop, noise_level):
    """
    Create a neo.SpikeTrain with waveforms and quality metrics.
    
    Builds a SpikeTrain, attaches waveforms and metadata, and computes
    firing rate, SNR, line noise contamination, and presence ratio as
    annotations.
    
    Parameters
    ----------
    spike_times : np.ndarray or pq.Quantity
        Spike times relative to the start of the segment.
    waveforms : np.ndarray
        Waveform snippets with shape (n_spikes, n_samples).
    metadata : dict
        Channel metadata to include as annotations.
    t_start : pq.Quantity
        Start time of the recording segment.
    t_stop : pq.Quantity
        Stop time of the recording segment.
    noise_level : float
        Standard deviation of background noise for SNR calculation.
    
    Returns
    -------
    spiketrain : neo.SpikeTrain
        SpikeTrain object with waveforms and quality metric annotations.
    """
    # Create spiketrain with waveforms
    spiketrain = neo.SpikeTrain(
        times=spike_times + t_start,
        t_start=t_start,
        t_stop=t_stop)

    # Add waveforms
    spiketrain.waveforms = waveforms

    # Include channel metadata
    spiketrain.annotations.update(metadata)

    # Compute metrics (keep as numbers, don't convert to string)
    n_spikes = len(spiketrain.times)
    duration = (t_stop - t_start).rescale('s').magnitude
    firing_rate = n_spikes / duration
    snr = waveform_snr(waveforms, noise_level)
    spiketrain.annotations['n_spikes'] = int(n_spikes)
    spiketrain.annotations['firing_rate'] = float(firing_rate)
    spiketrain.annotations['waveform_SNR'] = float(snr)

    # Calculate line noise contamination
    corr50, corr60 = estimate_line_noise_from_spikes(spiketrain)
    spiketrain.annotations['line_noise_50Hz'] = float(corr50)
    spiketrain.annotations['line_noise_60Hz'] = float(corr60)
    
    # Presence ratio
    spiketrain.annotations['presence_ratio'] = presence_ratio(spiketrain)

    # Check that annotations are not illegal values (None, arrays, lists)
    for k in spiketrain.annotations.keys():
        if isinstance(spiketrain.annotations[k], (list, np.ndarray)):
            spiketrain.annotations[k] = str(spiketrain.annotations[k])
        elif spiketrain.annotations[k] != spiketrain.annotations[k]:
            spiketrain.annotations[k] = 'NaN'
        elif spiketrain.annotations[k] is None:
            spiketrain.annotations[k] = 'None'

    return spiketrain