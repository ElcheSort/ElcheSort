"""
Example spike sorting pipeline using ElcheSort.

This script shows how to load a raw recording, preprocess it,
run spike sorting channel by channel, and generate a PDF report.

Adapt the data loading section to your own file format and metadata.
"""
import numpy as np
import quantities as pq
import neo
from tqdm import tqdm

from elchesort.preprocessing import bandpass_filter, whiten_signals
from elchesort.core import spike_sort_channel
from elchesort.postprocessing import generate_spiketrain_object, detect_waveform_outliers
from elchesort.spike_report import generate_pdf_report


# ---- Parameters ----

sorting_parameters = {
    'hipass_freq': 300,  # Hz
    'lowpass_freq': 6000,  # Hz
    'filter_order': 4,
    'detection_threshold': 16,
    'exclusion_bins': 41,
    'n_pcs': 8
}

quality_thresholds = {
    'firing_rate': 1,  # spikes/s
    'waveform_snr': 5,  # stds
    'line_noise_50Hz': 0.6,  # percentage
    'line_noise_60Hz': 0.6,  # percentage
    'presence_ratio': 0.7,  # percentage
}


# ---- Load your data ----
# Replace this section with your own data loading code.
# The result should be:
#   signals       : np.ndarray, shape (n_channels, n_timepoints)
#   sampling_rate : float, in Hz (e.g. 30000.0)
#   t_start       : quantities.Quantity (e.g. 0.0 * pq.s)
#   t_stop        : quantities.Quantity (e.g. 600.0 * pq.s)

# Example using Neo's BlackrockIO:
# io = neo.BlackrockIO('my_recording.ns5', load_nev=False)
# block = io.read_block(lazy=True)
# anasig_proxy = block.segments[0].analogsignals[0]
# signals = anasig_proxy.load().magnitude.T  # (channels, timepoints)
# sampling_rate = float(anasig_proxy.sampling_rate.magnitude)
# t_start = anasig_proxy.t_start
# t_stop = anasig_proxy.t_stop


# ---- Preprocessing ----

print(f"Bandpass filtering ({sorting_parameters['hipass_freq']}–{sorting_parameters['lowpass_freq']} Hz)...")
for i in tqdm(range(signals.shape[0])):
    signals[i] = bandpass_filter(
        signals[i], sampling_rate,
        highpass_freq=sorting_parameters['hipass_freq'], 
        lowpass_freq=sorting_parameters['lowpass_freq'], 
        order=sorting_parameters['filter_order']
    ).squeeze(0)

print('ZCA whitening...')
signals = whiten_signals(signals, noise_threshold=3)


# ---- Spike sorting ----

print('Sorting channels...')
results = []
for ch in tqdm(range(signals.shape[0])):
    result = spike_sort_channel(signals[ch], templates='human', **sorting_parameters)
    results.append(result)

spike_time_idx_list, clusters_list, waveforms_list, features_list = zip(*results)


# ---- Channel metadata ----
# If you have per-channel metadata (e.g. electrode positions, brain region),
# provide it as a dict mapping channel index -> dict of annotations.
# These get attached to each spiketrain produced from that channel.
# Leave as an empty dict per channel if you have no metadata.
#
# Example:
# channel_metadata = {
#     0: {'x': 0.0, 'y': 0.0, 'region': 'M1'},
#     1: {'x': 0.4, 'y': 0.0, 'region': 'M1'},
#     ...
# }
#
# Ideally the metadata should be loaded from an external file (e.g. csv, yml, json)
# Make sure the metadata indices and the data match each other!
#
n_channels = signals.shape[0]
channel_metadata = {ch: {} for ch in range(n_channels)}

# ---- Postprocessing ----

print('Building spike trains...')
spiketrains = []
unit_id = 0

for ch, (spike_idx, clusters, waveforms, features) in enumerate(
    zip(spike_time_idx_list, clusters_list, waveforms_list, features_list)
):
    if len(spike_idx) == 0:
        continue

    spike_times = spike_idx / sampling_rate

    for clus in np.unique(clusters):
        mask = clusters == clus
        if mask.sum() < 10:
            continue

        # Remove waveform outliers (5 MADs from cluster median in PCA space)
        clean, _, _ = detect_waveform_outliers(features[mask], n_mad=5)

        st = generate_spiketrain_object(
            spike_times[mask][clean] * pq.s,
            waveforms[mask][clean] * pq.uV,
            channel_metadata[ch],  # XXX You have to define channel_metadata yourself
            t_start=t_start,
            t_stop=t_stop,
            noise_level=signals[ch].std().item()
        )
        st.name = f"Ch {ch}, Unit {unit_id}"
        unit_id += 1

        # Label based on quality metrics (computed inside generate_spiketrain_object)
        ann = st.annotations
        if (ann['firing_rate'] > quality_thresholds['firing_rate'] and
            ann['waveform_SNR'] > quality_thresholds['waveform_snr'] and
            ann['line_noise_50Hz'] < quality_thresholds['line_noise_50Hz'] and
            ann['line_noise_60Hz'] < quality_thresholds['line_noise_60Hz'] and
            ann['presence_ratio'] > quality_thresholds['presence_ratio']):
            ann['Label'] = 'Good'
        else:
            ann['Label'] = 'Bad'

        spiketrains.append(st)

print(f'Found {len(spiketrains)} units '
      f'({sum(1 for s in spiketrains if s.annotations["Label"] == "Good")} good)')


# ---- Report ----

good = [st for st in spiketrains if st.annotations['Label'] == 'Good']
bad  = [st for st in spiketrains if st.annotations['Label'] != 'Good']

if good:
    generate_pdf_report(good, 'report_good.pdf')
if bad:
    generate_pdf_report(bad, 'report_bad.pdf')