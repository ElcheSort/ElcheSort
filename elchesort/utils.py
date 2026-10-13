
import numpy as np

def waveform_amp_and_width(wf, center_frac=0.4):
    """Measure amplitude and width from central trough to next peak.

    Parameters
    ----------
    wf : array_like, shape (T,) or (N, T)
        Single waveform or batch of waveforms.
    center_frac : float
        Fraction of the waveform (centered) in which to search for the trough.

    Returns
    -------
    amplitude : float or ndarray of shape (N,)
        wf[peak] - wf[trough].
    width : int or ndarray of shape (N,)
        peak_index - trough_index, in samples.
    """
    wf = np.asarray(wf, dtype=float)
    squeeze = wf.ndim == 1
    if squeeze:
        wf = wf[None, :]

    N, T = wf.shape
    lo = int(T * (0.5 - center_frac / 2))
    hi = int(T * (0.5 + center_frac / 2))
    trough = lo + np.argmin(wf[:, lo:hi], axis=1)

    diff = np.diff(wf, axis=1)
    idx = np.arange(T - 1)
    not_rising_after_trough = (diff <= 0) & (idx >= trough[:, None])
    peak = np.argmax(not_rising_after_trough, axis=1)

    width = peak - trough
    amplitude = wf[np.arange(N), peak] - wf[np.arange(N), trough]

    if squeeze:
        return amplitude[0], width[0]
    return amplitude, width