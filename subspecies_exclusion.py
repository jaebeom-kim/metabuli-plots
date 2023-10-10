import os
import os.path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib_venn import venn2, venn3
import pandas as pd
import seaborn as sns
import numpy as np

from matplotlib import patches
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes


def subspecies_exclusion():
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['figure.figsize'] = [16, 4.5]

    # Read data
    short_data = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_short.tsv', sep='\t')
    short_data = short_data[short_data['Rank'] == 'Species']
    ont_data = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_ont.tsv', sep='\t')
    ont_data = ont_data[ont_data['Rank'] == 'Species']
    sequel_data = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_sequel.tsv', sep='\t')
    sequel_data = sequel_data[sequel_data['Rank'] == 'Species']

    # Scatter plot parameters
    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2',
             'Centrifuge', 'Metamaps', 'Kraken2X', 'Kaiju', 'MMseqs2']
    markers = ['o', 's', 'H', 'D', 'v', "^", 'P', 'X', 'd']
    colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    marker_size = 120

    # Things for each panel
    titles = ["Illumina", "ONT", "PacBio-Sequel"]
    labels = ['a', 'b', 'c']
    data = [short_data, ont_data, sequel_data]
    zoom_xlims_min = [0.8, 0.8, 0.75]
    zoom_xlims_max = [0.9, 0.9, 0.85]
    zoom_ylims_min = [0.9, 0.9, 0.89]
    zoom_ylims_max = [1, 1, 0.99]
    zoom_width = [0.9, 1.8, 1.8]
    zoom_height = [1.8, 1.8, 1.8]

    # Make subplots
    panels = gridspec.GridSpec(1, 4)
    gs_short = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=panels[0])
    gs_ont = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=panels[1])
    gs_sequel = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=panels[2])
    short_panel = plt.subplot(gs_short[0])
    ont_panel = plt.subplot(gs_ont[0])
    sequel_panel = plt.subplot(gs_sequel[0])
    axs = [short_panel, ont_panel, sequel_panel]

    ont_panel.set_xlabel('Recall (TP / # of reads)', fontsize=14, fontweight='bold', fontfamily='Arial')
    # ont_panel.xaxis.set_label_coords(2.3, -0.15)
    short_panel.set_ylabel('Precision (TP / TP+FP)', fontsize=14, fontweight='bold', fontfamily='Arial')
    # gtdb_in_short.yaxis.set_label_coords(-0.15, 0)

    for i in range(3):
        # set title
        axs[i].set_title(titles[i], fontsize=15, fontweight='bold', fontfamily='Arial')
        # Axis range
        axs[i].set_xlim(0.0, 1.0)
        axs[i].set_ylim(0.0, 1.0)

        axs[i].spines[['right', 'top']].set_visible(False)

        # Ticks
        axs[i].xaxis.set_ticks(np.arange(0, 1.0001, 0.2))
        axs[i].yaxis.set_ticks(np.arange(0, 1.0001, 0.2))
        if i == 0:
            axs[i].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            axs[i].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        else:
            axs[i].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            axs[i].set_yticklabels([])

        for tick in axs[i].get_yticklabels() + axs[i].get_xticklabels():
            tick.set_fontname('Arial')

        axs[i] = sns.scatterplot(x='Sensitivity', y='Precision',
                                 hue='Tool',  # different colors by group
                                 style='Tool',  # different shapes by group
                                 hue_order=order,
                                 style_order=order,
                                 edgecolor='black',
                                 palette=colors,
                                 markers=markers,
                                 s=marker_size,  # marker size
                                 data=data[i], ax=axs[i])
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')

        # Zoom-in plot
        axins = inset_axes(axs[i], loc='lower left', borderpad=2.7,
                           width=1.8,
                           height=1.8)
        axins = sns.scatterplot(x='Sensitivity', y='Precision',
                                hue='Tool',  # different colors by group
                                style='Tool',  # different shapes by group
                                hue_order=order,
                                style_order=order,
                                edgecolor='black',
                                palette=colors,
                                markers=markers,
                                s=marker_size,  # marker size
                                data=data[i], ax=axins)

        # Adjust the size of the zoomed-in plot
        axins.set_xlim(zoom_xlims_min[i], zoom_xlims_max[i])
        axins.set_ylim(zoom_ylims_min[i], zoom_ylims_max[i])

        # Draw a box around the inset axes in the parent axes and
        mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="0", lw=1)
        # axins.tick_params(color='black')
        for spine in axins.spines.values():
            spine.set_edgecolor('black')
        # fix the number of ticks on the inset axes
        axins.yaxis.get_major_locator().set_params(nbins=2)
        axins.xaxis.get_major_locator().set_params(nbins=2)

        # Remove legend
        axins.legend_.remove()

        # Remove x and y labels
        axins.set_xlabel('')
        axins.set_ylabel('')

        # legend
        if i != 2:
            axs[i].legend_.remove()
        else:
            axs[i].legend(loc='lower right', markerscale=2, edgecolor='black', fontsize=10)
            axs[i].legend(bbox_to_anchor=(1.5, 1.05))
            handles, labels2 = axs[i].get_legend_handles_labels()
            print(handles)
            print(labels2)

            r = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
            labels2.insert(2, " ")
            handles.insert(2, r)

            # Column names
            labels2.insert(3, "DNA-based")
            labels2.insert(8, "AA-based")
            labels2.insert(0, "Both")
            handles.insert(0, r)
            handles.insert(4, r)
            handles.insert(9, r)

            handles = handles[4:9] + handles[9:13] + handles[0:4]
            labels2 = labels2[4:9] + labels2[9:13] + labels2[0:4]

            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[i].legend(handles, labels2, loc='lower left', markerscale=2, fontsize=11,
                                         handletextpad=0.5, handlelength=0.7, edgecolor='black', ncol=3,
                                         columnspacing=0.2, framealpha=0.3, bbox_to_anchor=(1, 0))
            axs[i].add_artist(first_legend)

        # axins.text(zoom_xlims_max[i],
        #            zoom_ylims_min[i] - 0.002,
        #            str(zoom_xlims_max[i]),
        #            ha='center', va='top', fontsize=14, fontfamily='Arial')
        # axins.annotate(str(zoom_xlims_min[i]), xy=(zoom_xlims_min[i], zoom_ylims_min[i]),
        #                xytext=(zoom_xlims_min[i], zoom_ylims_min[i]-0.018),
        #                fontsize=13, fontfamily='Arial', va='bottom',
        #                ha='center', arrowprops=dict(arrowstyle='-', color='black'))

    ont_panel.set_xlabel('Recall (TP / # of reads)', fontsize=14, fontweight='bold', fontfamily='Arial')
    short_panel.set_ylabel('Precision (TP / TP+FP)', fontsize=14, fontweight='bold', fontfamily='Arial')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # gtdb_longread()
    subspecies_exclusion()
