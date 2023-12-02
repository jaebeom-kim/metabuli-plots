import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes


def cami_gtdb() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'


    # Load Data
    soil = pd.read_csv('cami-gtdb/cami-gtdb_soil.tsv', sep='\t')

    # Set figure size
    fig, axs = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(3.7, 3.5))
    markers = ['o', 's', 'H', 'D', 'v', 'P', 'X', 'd']
    colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    rank_order = ['Genus', 'Species']
    marker_size = 120

    # DATA
    data = soil
    labels = ['', '', '', '']
    x_pos = 0
    y_pos = 1.04

    # Set x and y limits
    axs.set_xlim(0, 1.04)
    axs.set_ylim(0, 1.04)
    axs.xaxis.set_ticks(np.arange(0, 1.02, 0.2))
    axs.yaxis.set_ticks(np.arange(0, 1.02, 0.2))

    # Subplot titles
    title = 'CAMI2 Plant-associated'

    sns.scatterplot(x='Sensitivity', y='Precision',
                    hue='Tool',  # different colors by group
                    style='Tool',  # different shapes by group
                    hue_order=order,
                    style_order=order,
                    edgecolor='black',
                    palette=colors,
                    markers=markers,
                    s=marker_size,  # marker size
                    data=data, ax=axs)

    # axs.set_title(title, fontsize=12, fontweight='bold', fontfamily='Arial')

    axs.set_xlabel('Recall', fontsize=14,  fontfamily='Arial')
    axs.set_ylabel('Precision', fontsize=14, fontfamily='Arial')

    axs.legend_.remove()
    axs.spines[['right', 'top']].set_visible(False)

    # Species ZOOM
    axin1 = inset_axes(axs, loc='lower left', width=0.7, height=0.7, borderpad=1)
    axin1.set_xlim(0.32, 0.37)
    axin1.set_ylim(0.46, 0.51)
    axin1 = sns.scatterplot(x='Sensitivity', y='Precision',
                            hue='Tool',  # different colors by group
                            style='Tool',  # different shapes by group
                            hue_order=order,
                            style_order=order,
                            edgecolor='black',
                            palette=colors,
                            markers=markers,
                            s=70,  # marker size
                            data=data, ax=axin1)
                            # different shapes by group

    mark_inset(axs, axin1, loc1=2, loc2=1, fc="none", ec="0", lw=1)
    axin1.legend_.remove()
    for spine in axin1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    axin1.set_xlabel('')
    axin1.set_ylabel('')
    axin1.set_yticklabels(['', ''])
    axin1.set_xticklabels(['', '', '', '', '', ''])
    axin1.yaxis.get_major_locator().set_params(nbins=1)
    axin1.xaxis.get_major_locator().set_params(nbins=1)

    # Genus ZOOM
    axin2 = inset_axes(axs, width=0.9, height=0.9, borderpad=0.5, bbox_transform=axs.transAxes,
                       bbox_to_anchor=(1, 0.7))
    axin2.set_xlim(0.6, 0.8)
    axin2.set_ylim(0.88, 1)
    axin2 = sns.scatterplot(x='Sensitivity', y='Precision',
                            hue='Tool',  # different colors by group
                            style='Tool',  # different shapes by group
                            hue_order=order,
                            style_order=order,
                            edgecolor='black',
                            palette=colors,
                            markers=markers,
                            s=70,  # marker size
                            data=data, ax=axin2)
    # different shapes by group

    mark_inset(axs, axin2, loc1=1, loc2=3, fc="none", ec="0", lw=1)
    axin2.legend_.remove()
    for spine in axin2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    axin2.set_xlabel('')
    axin2.set_ylabel('')
    axin2.set_yticklabels(['', ''])
    axin2.set_xticklabels(['', '', '', '', '', ''])
    axin2.yaxis.get_major_locator().set_params(nbins=1)
    axin2.xaxis.get_major_locator().set_params(nbins=1)
    # axin1.text(0.99, 0.9, '1.0', ha='right', va='bottom', fontsize=15, fontfamily='Arial')
    # # axins.text(0.51, 0.9, '0.5', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
    # # axins.text(0.5, 0.9, '0.9', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
    # axin1.text(0.505, 0.998, '1.0', ha='left', va='top', fontsize=15, fontfamily='Arial')
    # axin1.annotate('0.5', xy=(0.5, 0.9), xytext=(0.54, 0.9), fontsize=15, fontfamily='Arial', va='bottom',
    #                ha='left', arrowprops=dict(arrowstyle='-', color='black'))
    # axin1.annotate('0.9', xy=(0.5, 0.9), xytext=(0.505, 0.908), fontsize=15, fontfamily='Arial', va='bottom',
    #                ha='left', arrowprops=dict(arrowstyle='-', color='black'))


    # # Add a line for subspecies and species
    axs.axvline(x=0.45, color='gray', linestyle='--', linewidth=1.5)
    axs.text(0.1, 0.8, 'Species', fontsize=12, fontweight='bold', fontfamily='Arial')
    axs.text(0.65, 0.8, 'Genus', fontsize=12, fontweight='bold', fontfamily='Arial')

    # Add x tick labels to axs[0,1] and make visible
    axs.tick_params(axis='x', which='both', labelbottom=True)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9)
    plt.savefig('./plots/figure2e.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    cami_gtdb()