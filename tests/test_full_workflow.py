"""
End-to-end and unit tests for ElcheSort.

Loads a real recording from test_data.npyz for the pipeline test,
and uses small synthetic data for targeted unit tests.

Place test_data.npy in the same directory as this file.
It should contain:
    signal : float32 array, shape (n_channels, n_samples)
"""
import numpy as np
import torch
import pytest
import quantities as pq
import neo
from pathlib import Path

from elchesort.preprocessing import bandpass_filter, whiten_signals
from elchesort.core import spike_sort_channel, match_templates, isosplit_on_features
from elchesort.postprocessing import (
    detect_waveform_outliers, waveform_snr, estimate_line_noise_from_spikes,
    presence_ratio, generate_spiketrain_object,
)
from elchesort.spike_report import generate_pdf_report

DATA_PATH = Path(__file__).parent / "test_data.npy"
FS = 30000.0


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def raw_signal():
    assert DATA_PATH.exists(), f"Test data not found at {DATA_PATH}"
    signal = np.load(DATA_PATH)
    assert signal.ndim == 2, f"Expected (n_channels, n_samples), got {signal.shape}"
    return signal.astype(np.float32)


@pytest.fixture(scope="module")
def preprocessed(raw_signal):
    filtered = bandpass_filter(raw_signal, sampling_rate=FS, 
                               highpass_freq=500, lowpass_freq=7500,
                               order=4)
    whitened = whiten_signals(filtered, noise_threshold=3)
    return whitened


@pytest.fixture(scope="module")
def sorted_channel(preprocessed):
    """Sort channels until one produces spikes. Skip all dependent tests if none do."""
    for ch in range(preprocessed.shape[0]):
        times, clusters, waveforms, features = spike_sort_channel(
            preprocessed[ch], templates='human',
            detection_threshold=12, n_pcs=6,
        )
        if len(times) > 10:
            return times, clusters, waveforms, features, preprocessed[ch]
    pytest.skip("No channel produced enough spikes for postprocessing tests")


# =====================================================================
# Pipeline test
# =====================================================================

def test_full_pipeline(preprocessed):
    """Run sorting + postprocessing on every channel."""
    n_channels = preprocessed.shape[0]
    total_units = 0

    for ch in range(n_channels):
        times, clusters, waveforms, features = spike_sort_channel(
            preprocessed[ch], templates='human',
            detection_threshold=12, n_pcs=6,
        )
        if len(times) < 10:
            continue

        noise_level = preprocessed[ch].std().item()
        for clus in np.unique(clusters):
            mask = clusters == clus
            if mask.sum() < 10:
                continue
            clean_mask, _, _ = detect_waveform_outliers(features[mask], n_mad=5)
            snr = waveform_snr(waveforms[mask][clean_mask], noise_level)
            if snr > 1.0:
                total_units += 1

    assert total_units > 0, "Pipeline produced zero valid units across all channels"


# =====================================================================
# Preprocessing
# =====================================================================

def test_bandpass_quantities_input(raw_signal):
    """bandpass_filter accepts quantities.Quantity as sampling rate."""
    result = bandpass_filter(raw_signal[0], 30000.0 * pq.Hz)
    assert result.shape[1] == raw_signal.shape[1]


def test_bandpass_transposed_input(raw_signal):
    """bandpass_filter handles (timepoints, channels) input."""
    transposed = raw_signal[:2].T  # (n_samples, 2)
    result = bandpass_filter(transposed, sampling_rate=FS)
    assert result.shape[0] <= 2


def test_whiten_transposed_input(raw_signal):
    """whiten_signals handles (timepoints, channels) input."""
    transposed = raw_signal[:2].T
    result = whiten_signals(transposed, noise_threshold=3)
    assert result.shape[0] <= 2


# =====================================================================
# Core — template loading
# =====================================================================

def test_sort_with_human_templates(preprocessed):
    """spike_sort_channel loads human templates without error."""
    times, clusters, waveforms, features = spike_sort_channel(
        preprocessed[0], templates='human',
        detection_threshold=12, n_pcs=6,
    )
    assert isinstance(times, (list, np.ndarray))


def test_sort_with_macaque_templates(preprocessed):
    """spike_sort_channel loads macaque templates without error."""
    times, clusters, waveforms, features = spike_sort_channel(
        preprocessed[0], templates='macaque',
        detection_threshold=12, n_pcs=6,
    )
    assert isinstance(times, (list, np.ndarray))


def test_sort_unknown_template_raises():
    """Unknown template string raises ValueError."""
    signal = torch.randn(10000)
    with pytest.raises(ValueError, match="Unknown template set"):
        spike_sort_channel(signal, templates='rat')


# =====================================================================
# Core — edge cases
# =====================================================================

def test_sort_silent_signal_returns_empty():
    """Very quiet signal returns empty results."""
    signal = torch.randn(10000) * 0.001
    times, clusters, waveforms, features = spike_sort_channel(
        signal, templates='human', detection_threshold=120,
    )
    assert times == [] or len(times) == 0


def test_isosplit_empty_input():
    """isosplit_on_features handles empty array."""
    labels = isosplit_on_features(np.zeros((0, 6), dtype=np.float32))
    assert len(labels) == 0


def test_isosplit_single_cluster():
    """isosplit_on_features handles tight single-cluster data."""
    rng = np.random.default_rng(42)
    features = rng.normal(0, 0.01, (50, 6)).astype(np.float32)
    labels = isosplit_on_features(features)
    assert len(labels) == 50


def test_match_templates_no_spikes():
    """match_templates returns empty tensors when nothing crosses threshold."""
    signal = torch.randn(5000) * 0.001
    templates = torch.randn(2, 61)
    times, ids = match_templates(signal, templates, detection_threshold=1200)
    assert len(times) == 0
    assert len(ids) == 0


# =====================================================================
# Postprocessing — waveform_snr modes
# =====================================================================

def test_snr_neg(sorted_channel):
    _, clusters, waveforms, _, _ = sorted_channel
    clus = np.unique(clusters)[0]
    snr = waveform_snr(waveforms[clusters == clus], noise_level=1.0, peak_sign="neg")
    assert snr > 0


def test_snr_pos(sorted_channel):
    _, clusters, waveforms, _, _ = sorted_channel
    clus = np.unique(clusters)[0]
    snr = waveform_snr(waveforms[clusters == clus], noise_level=1.0, peak_sign="pos")
    assert snr > 0


def test_snr_both(sorted_channel):
    _, clusters, waveforms, _, _ = sorted_channel
    clus = np.unique(clusters)[0]
    snr = waveform_snr(waveforms[clusters == clus], noise_level=1.0, peak_sign="both")
    assert snr > 0


def test_snr_at_index(sorted_channel):
    _, clusters, waveforms, _, _ = sorted_channel
    clus = np.unique(clusters)[0]
    snr = waveform_snr(waveforms[clusters == clus], noise_level=1.0, peak_mode="at_index")
    assert snr >= 0


def test_snr_peak_to_peak(sorted_channel):
    _, clusters, waveforms, _, _ = sorted_channel
    clus = np.unique(clusters)[0]
    snr = waveform_snr(waveforms[clusters == clus], noise_level=1.0, peak_mode="peak_to_peak")
    assert snr > 0


# =====================================================================
# Postprocessing — line noise, presence ratio, spiketrain generation
# =====================================================================

def test_estimate_line_noise(sorted_channel):
    """estimate_line_noise_from_spikes returns two correlations."""
    times, clusters, _, _, _ = sorted_channel
    clus = np.unique(clusters)[0]
    spike_times = times[clusters == clus] / FS
    st = neo.SpikeTrain(spike_times * pq.s, t_stop=spike_times.max() * pq.s + 0.1 * pq.s)
    corr50, corr60 = estimate_line_noise_from_spikes(st)
    assert isinstance(corr50, float)
    assert isinstance(corr60, float)


def test_presence_ratio(sorted_channel):
    """presence_ratio returns a value between 0 and 1."""
    times, clusters, _, _, _ = sorted_channel
    clus = np.unique(clusters)[0]
    spike_times = times[clusters == clus] / FS
    st = neo.SpikeTrain(spike_times * pq.s, t_stop=spike_times.max() * pq.s + 0.1 * pq.s)
    ratio = presence_ratio(st, bin_size=1 * pq.s)
    assert 0 <= ratio <= 1


def test_generate_spiketrain_object(sorted_channel):
    """generate_spiketrain_object produces a valid neo.SpikeTrain with annotations."""
    times, clusters, waveforms, _, signal = sorted_channel
    clus = np.unique(clusters)[0]
    mask = clusters == clus
    spike_times = times[mask] / FS

    t_start = 0.0 * pq.s
    t_stop = (len(signal) / FS) * pq.s
    metadata = {"Array_ID": "test", "Electrode_ID": 1}

    st = generate_spiketrain_object(
        spike_times * pq.s, waveforms[mask] * pq.uV,
        metadata, t_start, t_stop,
        noise_level=signal.std().item(),
    )
    assert isinstance(st, neo.SpikeTrain)
    assert "firing_rate" in st.annotations
    assert "waveform_SNR" in st.annotations
    assert "line_noise_50Hz" in st.annotations
    assert "presence_ratio" in st.annotations
    assert st.annotations["Array_ID"] == "test"


# =====================================================================
# Spike report
# =====================================================================

def test_generate_pdf_report(sorted_channel, tmp_path):
    """generate_pdf_report creates a valid PDF file."""
    times, clusters, waveforms, features, signal = sorted_channel
    t_start = 0.0 * pq.s
    t_stop = (len(signal) / FS) * pq.s
    noise_level = signal.std().item()

    # Build one spiketrain per cluster with required annotations
    spiketrains = []
    for i, clus in enumerate(np.unique(clusters)):
        mask = clusters == clus
        if mask.sum() < 10:
            continue
        spike_times = times[mask] / FS
        st = generate_spiketrain_object(
            spike_times * pq.s, waveforms[mask] * pq.uV,
            {"Array_ID": "test", "Electrode_ID": 1},
            t_start, t_stop, noise_level,
        )
        st.name = f"Unit {i}"
        st.annotations["session"] = "test_session"
        st.annotations["Label"] = "Good"
        spiketrains.append(st)

    assert len(spiketrains) > 0, "Need at least one spiketrain to test report"

    report_path = str("test_report.pdf")
    generate_pdf_report(spiketrains, report_path, events=[5, 10], channel_key="Electrode_ID")

    assert Path(report_path).exists()
    assert Path(report_path).stat().st_size > 0