# ElcheSort

A minimalistic spike sorting algorithm for sorting electrodes one at a time.

ElcheSort detects spikes via template matching, extracts aligned waveforms, projects them into PCA space, and clusters them with [ISO-SPLIT](https://github.com/flatironinstitute/isosplit6). The entire sorting pipeline for a single channel is about 250 lines of code.

## Installation

So far only installation from source is possible:

```bash
git clone https://github.com/ElcheSort/ElcheSort.git
cd ElcheSort
pip install -e .
```

Use the `-e` flag to install elchesort in development mode and adjust any parts as needed.

## Usage

```python
from elchesort import spike_sort_channel, bandpass_filter, whiten_signals

# Preprocess
filtered = bandpass_filter(raw_signal, sampling_rate=30000)
whitened = whiten_signals(filtered)

# Sort a single channel
spike_times, clusters, waveforms, features = spike_sort_channel(whitened[0])
```

`spike_sort_channel` accepts a 1-D signal and returns spike times, cluster labels, extracted waveforms, and PCA features. Built-in template sets are available for human and macaque data (`templates='human'` or `templates='macaque'`), or you can pass your own array of shape `(n_templates, 61)`. Different template shapes are also possible, typical sizes are 61 or 91 samples (roughly 2 or 3 ms).

## How it works

The pipeline has three steps:

1. **Template matching** — The signal is convolved against a set of waveform templates. Peaks above a threshold are detected with non-maximum suppression. This is a simplified single-channel version of the approach used in [Kilosort](https://github.com/MouseLand/Kilosort).

2. **Feature extraction** — Waveforms are extracted around each detected spike and projected into a low-dimensional space using PCA (`torch.pca_lowrank`).

3. **Clustering** — PCA features are clustered with ISO-SPLIT, followed by recursive hierarchical subdivision of the resulting clusters.

Preprocessing (bandpass filtering and ZCA whitening) and postprocessing (outlier detection, SNR, line noise estimation, quality labeling) are provided as separate functions.

## Report generation

ElcheSort can generate a PDF report with one page per electrode, showing waveforms, PCA projections, ISI distributions, amplitude stability over time, and quality labels for each sorted unit.

```python
from elchesort import generate_pdf_report

generate_pdf_report(spiketrains, 'my_report.pdf', channel_key='channel_ids', label_key=None, title_key=None)
```

The report assumes that spiketrains is a list of `neo.SpikeTrain` objects, with annotations for channels (`channel_key` kwarg), `label_key` and `title_key` are used to pass on which annotations (if any) should be used for the page title (e.g. the session name, or date) and the legend keys (e.g. labels for SUA, MUA etc).

Example report page:

![Example report page](report_example.png)

## Project structure

```
elchesort/
├── core.py              # Template matching, ISO-SPLIT clustering, main sorting function
├── preprocessing.py     # FFT bandpass filter, ZCA whitening
├── postprocessing.py    # Outlier detection, SNR, line noise, neo.SpikeTrain generation
├── spike_report.py      # PDF report generation with per-channel summary figures
└── sample_templates/    # Built-in waveform templates (human, macaque)
```

## Dependencies

PyTorch, NumPy, SciPy, scikit-learn, neo, elephant, quantities, isosplit6, statsmodels, matplotlib, pikepdf, tqdm.

## License

GPL-3.0. See [LICENSE](LICENSE) for details.