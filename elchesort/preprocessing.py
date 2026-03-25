"""
ElcheSort: A simple spike sorting library for single-channel sorting 
with template matching and isosplit clustering.

Preprocessing module

2026 Aitor Morales-Gregorio
"""
import torch
import numpy as np
import quantities


def bandpass_filter(signals, sampling_rate,
                    highpass_freq=500, lowpass_freq=7500,
                    order=4, device=torch.device('cpu')):
    """
    Fast FFT-based Butterworth bandpass filtering using PyTorch.
    Equivalent to scipy.signal.filtfilt with Butterworth filter.
    
    The Butterworth filter magnitude response is:
        |H(f)|² = 1 / (1 + (f/fc)^(2n))
    
    filtfilt applies the filter forward and backward, squaring the magnitude
    response. The effective amplitude response becomes:
        |H_filtfilt(f)| = 1 / (1 + (f/fc)^(2n))
    
    For bandpass, we cascade highpass and lowpass responses.
    
    Parameters
    ----------
    signals : np.ndarray or torch.Tensor
        Input signals with shape (channels, timepoints) or (timepoints,) or (timepoints, channels).
    sampling_rate : float or quantities.Quantity
        Sampling rate in Hz.
    highpass_freq : float
        High-pass cutoff frequency in Hz.
    lowpass_freq : float
        Low-pass cutoff frequency in Hz.
    order : int
        Butterworth filter order.
    device : torch.device
        Device for computation (e.g., 'cpu' or 'cuda').
    
    Returns
    -------
    torch.Tensor
        Filtered signals with shape (channels, timepoints).
    """
    # IO
    if isinstance(sampling_rate, quantities.quantity.Quantity):
        sampling_rate = float(sampling_rate.magnitude)
    
    # Convert to tensor (use ascontiguousarray to avoid copy if possible)
    if isinstance(signals, np.ndarray):
        signals = torch.from_numpy(np.ascontiguousarray(signals))
    signals = signals.to(device=device, dtype=torch.float32)
    
    # Ensure shape: (channels, timepoints)
    if signals.ndim == 1:
        signals = signals.unsqueeze(0)
    elif signals.shape[0] > signals.shape[1]:
        signals = signals.T
    
    n_channels, n_samples = signals.shape
    
    # Pad to optimal FFT length (power of 2 is fastest)
    # This adds 0s to the signal to reach the closest 2**n, which then accelerates the calculation
    n_fft = 1 << (n_samples - 1).bit_length()  # next power of 2
    
    # Build frequency axis
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sampling_rate, device=device)
    
    # Butterworth bandpass amplitude response (equivalent to filtfilt)
    # |H(f)|² = 1/(1+(f/fc)^(2n)), filtfilt squares this giving |H|² as amplitude
    highpass = 1.0 / (1.0 + (highpass_freq / freqs.clamp(min=1e-10)) ** (2 * order))
    highpass[0] = 0
    lowpass = 1.0 / (1.0 + (freqs / lowpass_freq) ** (2 * order))
    mask = highpass * lowpass
    
    # Batch FFT with optimal padding
    signals_fft = torch.fft.rfft(signals, n=n_fft, dim=1)
    signals_fft *= mask
    signals = torch.fft.irfft(signals_fft, n=n_fft, dim=1)[:, :n_samples]
    del signals_fft
        
    return signals


def whiten_signals(signals, noise_threshold=3, device=torch.device('cpu')):
    """
    ZCA whitening based on background noise periods.
    
    Parameters
    ----------
    signals : np.ndarray or torch.Tensor
        Input signals with shape (channels, timepoints).
    noise_threshold : float
        Threshold in standard deviations for detecting non-background periods.
    device : torch.device
        Device for computation.
    
    Returns
    -------
    torch.Tensor
        Whitened signals with shape (channels, timepoints).
    """
    # Convert to tensor
    if isinstance(signals, np.ndarray):
        signals = torch.from_numpy(signals.copy())
    signals = signals.to(device=device, dtype=torch.float32)
    
    # Ensure shape: (channels, timepoints)
    if signals.shape[0] > signals.shape[1]:
        signals = signals.T
    
    # Determine background periods for better whitening
    thresholds = noise_threshold * signals.std(dim=1, keepdim=True)
    is_background_noise = (signals.abs() <= thresholds).all(dim=0)
    del thresholds

    # Fast covariance: (X @ X.T) / (n-1)
    bg_signals = signals[:, is_background_noise]
    bg_signals = bg_signals - bg_signals.mean(dim=1, keepdim=True)
    cov = (bg_signals @ bg_signals.T) / (bg_signals.shape[1] - 1)
    del bg_signals
    
    # Perform whitening
    U, S, _ = torch.linalg.svd(cov)
    whiten_matrix = U @ (U / torch.sqrt(S + 1e-5)).T
    signals = (signals.T @ whiten_matrix).T
        
    return signals

