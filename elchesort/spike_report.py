"""
ElcheSort: A simple spike sorting library for single-channel sorting 
with template matching and isosplit clustering.

Spike report module

2026 Aitor Morales-Gregorio
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import elephant
from matplotlib.backends.backend_pdf import PdfPages
import pikepdf
from tqdm import tqdm


DEFAULT_COLORS = ['b', 'r', 'g', 'orange', 'darkviolet', 'gold', 'k']


def clean_axes(ax, spines_to_hide=('top', 'right'), hide_ticks=None):
    """Remove specified spines and optionally hide tick marks."""
    for spine in spines_to_hide:
        ax.spines[spine].set_visible(False)
    if hide_ticks:
        if 'x' in hide_ticks:
            ax.set_xticks([])
        if 'y' in hide_ticks:
            ax.set_yticks([])


def plot_waveforms_fast(ax, waveforms, color, alpha=0.05, lw=0.5):
    """Plot all waveforms using LineCollection (10-100x faster than individual plot calls)."""
    x = np.arange(waveforms.shape[1])
    segments = [np.column_stack([x, wv]) for wv in waveforms]
    ax.add_collection(LineCollection(segments, colors=color, alpha=alpha, linewidths=lw))


def generate_report_page(spiketrain_lst, colors, events=None, channel_key='channel_id'):
    """
    Creates summary plot for a list of given spiketrains (ideally from a single electrode).
    
    Parameters
    ----------
    spiketrain_lst : list
        List of neo.SpikeTrain objects
    colors : list
        List of colors for each unit
    events : array-like, optional
        Event times (e.g., stimulation times) in seconds to plot as vertical lines
    """
    
    # Pre-extract data for all units (ensure all are plain numpy arrays)
    unit_data = []
    for st in spiketrain_lst:
        wvf = np.asarray(st.waveforms[:, :])
        unit_data.append({
            'waveforms': wvf,
            'mean_wvf': np.mean(wvf, axis=0),
            'amplitudes': np.ptp(wvf, axis=1),
            'widths': np.abs(np.argmax(wvf, axis=1) - np.argmin(wvf, axis=1)) / 30,
            'times_s': np.asarray(st.times.rescale('s').magnitude),
            'isi': np.asarray(elephant.statistics.isi(st).rescale('ms').magnitude),
            't_start': float(st.t_start.rescale('s').magnitude),
            't_stop': float(st.t_stop.rescale('s').magnitude),
            'annotations': st.annotations,
            'name': st.name,
        })

    # Setup figure
    fig = plt.figure(constrained_layout=True, figsize=(12, 7.5))
    fig.set_rasterized(True)
    gs = GridSpec(5, 8, figure=fig)
    
    # Create axes
    ax1 = fig.add_subplot(gs[:2, :2])    # Waveforms
    ax2 = fig.add_subplot(gs[:2, 6:])    # Width distribution
    ax3 = fig.add_subplot(gs[-2:, :-2])  # Amplitude over time
    ax4 = fig.add_subplot(gs[-2:, -2:], sharey=ax3)  # Amplitude histogram
    ax5 = fig.add_subplot(gs[:2, 2:4])   # PCA
    ax6 = fig.add_subplot(gs[:2, 4:6])   # ISI
    ax7 = fig.add_subplot(gs[2:-2, :-2], sharex=ax3)  # Time histogram
    ax8 = fig.add_subplot(gs[2:-2, -2:]) # Legend

    # Configure axes aesthetics
    clean_axes(ax1)
    clean_axes(ax2, spines_to_hide=('top', 'right', 'left'), hide_ticks='y')
    clean_axes(ax3)
    clean_axes(ax4, spines_to_hide=('top', 'right', 'bottom'), hide_ticks='x')
    clean_axes(ax5, hide_ticks='xy')
    clean_axes(ax6, spines_to_hide=('top', 'right', 'left'), hide_ticks='y')
    clean_axes(ax7)
    clean_axes(ax8, spines_to_hide=('top', 'right', 'left', 'bottom'), hide_ticks='xy')

    # Labels and titles
    first_ann = unit_data[0]['annotations']
    fig.suptitle(f"{first_ann['session']}, Channel {first_ann[channel_key]}")
    
    ax1.set(ylabel=r'$\Delta$V from baseline (std)', xlabel='Relative time (ms)', title='Waveforms')
    xticks = np.arange(0, 91, step=15)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks / 30)
    ax1.grid(alpha=0.2)
    
    ax2.set_xlabel('Waveform width (ms)')
    ax3.set(xlabel='Time (s)', ylabel='Amplitude (std)')
    ax5.set(xlabel='PC1', ylabel='PC2', title='PCA')
    ax5.axis('equal')
    ax6.set(xlabel='ISI (ms)', xlim=[0, 20])
    ax7.set_ylabel('Time hist.')

    # Plot all waveforms and means
    for i, ud in enumerate(unit_data):
        c = colors[i % len(colors)]
        plot_waveforms_fast(ax1, ud['waveforms'], c)
    
    for i, ud in enumerate(unit_data):
        c = colors[i % len(colors)]
        ax1.plot(ud['mean_wvf'], color='w', lw=2)
        ax1.plot(ud['mean_wvf'], color=c, alpha=0.5)
    ax1.autoscale_view()

    # PCA
    all_waveforms = np.concatenate([ud['waveforms'] for ud in unit_data])
    pca_result = PCA(n_components=5).fit_transform(StandardScaler().fit_transform(all_waveforms))
    
    idx = 0
    for i, ud in enumerate(unit_data):
        n = ud['waveforms'].shape[0]
        c = colors[i % len(colors)]
        ax5.plot(pca_result[idx:idx+n, 0], pca_result[idx:idx+n, 1],
                 color=c, alpha=0.5, marker='.', markersize=5, lw=0)
        idx += n

    # Histograms and scatter plots
    for i, ud in enumerate(unit_data):
        c = colors[i % len(colors)]
        time_bins = np.arange(ud['t_start'], ud['t_stop'], step=1)
        
        ax2.hist(ud['widths'], color=c, alpha=0.5, bins=15)
        ax3.plot(ud['times_s'], ud['amplitudes'], color=c, alpha=0.5, marker='.', markersize=5, lw=0)
        ax4.hist(ud['amplitudes'], color=c, alpha=0.5, orientation='horizontal', bins=20)
        ax6.hist(ud['isi'], color=c, alpha=0.5, bins=np.arange(0, 20, 0.5))
        ax7.hist(ud['times_s'], color=c, alpha=0.5, bins=time_bins)

    # Plot events as shaded boxes (background) - one box per cluster of nearby events
    if events is not None:
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        
        events = np.asarray(events)
        if len(events) > 0:
            events_sorted = np.sort(events)
            gaps = np.diff(events_sorted) > 0.1  # 100ms threshold
            cluster_starts = np.concatenate([[0], np.where(gaps)[0] + 1])
            cluster_ends = np.concatenate([np.where(gaps)[0], [len(events_sorted) - 1]])
            
            # Compute all box widths and use the maximum to ensure consistency
            widths = events_sorted[cluster_ends] - events_sorted[cluster_starts]
            consistent_width = max(np.max(widths), 0.01)
            
            # Minimum width: 0.5% of time range (ensures visibility)
            t_range = unit_data[0]['t_stop'] - unit_data[0]['t_start']
            min_width = t_range * 0.005
            consistent_width = max(consistent_width, min_width)
            
            # Get y-range
            ymin, ymax = ax3.get_ylim()
            height = ymax - ymin
            
            # Build list of Rectangle patches with consistent width
            patches = []
            for start_idx in cluster_starts:
                t_start = events_sorted[start_idx]
                patches.append(Rectangle((t_start, ymin), consistent_width, height))
            
            pc = PatchCollection(patches, facecolor='orange', alpha=0.2, 
                                 edgecolor='none', zorder=-99, antialiased=False)
            ax3.add_collection(pc)

    # Legend
    patches = [
        mpatches.Patch(
            color=colors[i % len(colors)],
            label=f"{i} (ch{ud['annotations'][channel_key]} {ud['name']}) Label: '{ud['annotations']['Label']}'",
            alpha=0.5
        )
        for i, ud in enumerate(unit_data)
    ]
    ax8.legend(handles=patches, loc='center', bbox_to_anchor=(0.3, 0.5))

    # Force rasterization
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.set_rasterization_zorder(0.5)

    return fig


def generate_pdf_report(spiketrains, report_pdf, colors=None, channel_key='channel_id', events=None):
    """
    Generate a PDF report with one page per channel.
    
    Parameters
    ----------
    spiketrains : list
        List of neo.SpikeTrain objects
    report_pdf : str
        Output PDF file path
    colors : list, optional
        List of colors for each unit
    channel_key : str, optional
        Annotation key to group spiketrains by channel
    events : array-like, optional
        Event times (e.g., stimulation times) in seconds to plot as vertical lines
    """
    
    colors = colors or DEFAULT_COLORS
    
    # Group spiketrains by channel
    sts_by_ch = {}
    for st in spiketrains:
        ch = st.annotations[channel_key]
        sts_by_ch.setdefault(ch, []).append(st)

    uncompressed_report_pdf = report_pdf.replace('.pdf', '_uncompressed.pdf')
    
    with PdfPages(uncompressed_report_pdf) as pdf:
        for ch, sts in tqdm(sts_by_ch.items()):
            fig = generate_report_page(sts, colors, events=events, channel_key=channel_key)
            pdf.savefig(fig, dpi=150)
            plt.close('all')

    # Compress and save
    with pikepdf.open(uncompressed_report_pdf) as pdf:
        pdf.save(report_pdf, compress_streams=True, recompress_flate=True)
    
    os.remove(uncompressed_report_pdf)