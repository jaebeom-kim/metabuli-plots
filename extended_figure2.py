import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes
from matplotlib import patches


def extended_figure2() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'


    # Load Data
    soil = pd.read_csv('./cami-gtdb/cami-gtdb_soil.tsv', sep='\t')
    marine = pd.read_csv('./cami-gtdb/cami-gtdb_marine.tsv', sep='\t')
    strain = pd.read_csv('./cami-gtdb/cami-gtdb_strain.tsv', sep='\t')

    # Set figure size
    fig, axs = plt.subplots(1, 4, sharex='all', sharey='all', figsize=(14, 3.5))

    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2', 'Hybrid']
    rank_order = ['Genus', 'Species']
    markers = ['o', 'X']
    colors = ['#D81B1B', '#E51EC3', 'darkorange', 'gold', 'navajowhite', 'darkgreen', 'turquoise', 'darkseagreen', 'dimgray']
    # colors = [ '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    marker_size = 120

    # DATA
    data = [strain, marine, soil]

    # Panel labels
    labels = ['a', 'b', 'c', '']
    x_pos = 0
    y_pos = 1.08

    # Set x and y limits
    axs[0].set_xlim(0, 1.04)
    axs[0].set_ylim(0, 1.04)
    axs[0].xaxis.set_ticks(np.arange(0, 1.02, 0.2))
    axs[0].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
    # axs[0, 1].set_xlim(0.4, 1.02)
    zoom_xmins = [0.55, 0.82, 0.32]
    zoom_xmaxs = [0.85, 0.97, 0.37]

    zoom_ymins = [0.85, 0.95, 0.46]
    zoom_ymaxs = [1, 1, 0.51]
    zoom_locs = [4, 4, 3]
    zoom_widths = [2, 2.4, 0.8]
    zoom_heights = [1, 0.8, 0.8]
    zoom_loc1s = [1, 4, 2]
    zoom_loc2s = [2, 2, 4]

    # Subplot titles
    titles = ['Strain-madness', 'Marine', 'Plant-associated']

    for i in range(3):
        axs[i] = sns.scatterplot(x='Sensitivity', y='Precision',
                                 hue='Tool',  # different colors by group
                                 style='Rank',  # different shapes by group
                                 hue_order=order,
                                 style_order=rank_order,
                                 edgecolor='black',
                                 palette=colors,
                                 markers=markers,
                                 s=marker_size,  # marker size
                                 data=data[i], ax=axs[i])
        # Set title
        axs[i].set_title(titles[i], fontsize=12, fontweight='bold', fontfamily='Arial')

        # Set panel label
        print(labels[i])
        axs[i].text(x_pos, y_pos, labels[i], fontsize=12, fontweight='bold', fontfamily='Arial')

        # Remove x and y labels
        if i != 0:
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')
        else:
            axs[i].set_xlabel('Recall', fontsize=14, fontweight='bold', fontfamily='Arial')
            axs[i].xaxis.set_label_coords(1.7, -0.15)
            axs[i].set_ylabel('Precision', fontsize=14, fontweight='bold', fontfamily='Arial')
        # Remove top and right spines
        axs[i].spines[['right', 'top']].set_visible(False)

        # # Add F1 score contour
        # x = np.linspace(0, 1, 100)
        # y = np.linspace(0, 1, 100)
        # X, Y = np.meshgrid(x, y)
        # Z = 2 * X * Y / (X + Y)
        # axs[i, j].contour(X, Y, Z, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], colors='grey', linestyles='dashed', linewidths=0.5)

        # Remove legend
        if i != 2:
            axs[i].legend_.remove()
        else:
            axs[i].legend(loc='lower right', markerscale=2, fontsize=12, edgecolor='black')
            handles, labels2 = axs[i].get_legend_handles_labels()
            handles_2 = handles.copy()
            labels_2 = labels2.copy()
            r = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
            print(handles)
            print(labels2)
            handles.remove(handles[0])
            handles.remove(handles[9])
            handles.remove(handles[9])
            handles.remove(handles[9])
            labels2.remove(labels2[0])
            labels2.remove(labels2[9])
            labels2.remove(labels2[9])
            labels2.remove(labels2[9])
            print(handles)
            print(labels2)

            labels2.insert(0, "Both")
            labels2.insert(3, "DNA-based")
            labels2.insert(7, "AA-based")

            handles.insert(0, r)
            handles.insert(3, r)
            handles.insert(7, r)

            print(handles)
            print(labels2)

            handles = handles[3:7] + handles[7:11] + handles[0:3] + handles[11:]
            labels2 = labels2[3:7] + labels2[7:11] + labels2[0:3] + labels2[11:]

            print(handles)
            print(labels2)



            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[i].legend(handles, labels2, markerscale=2, loc='lower left',
                                         fontsize=11, edgecolor='black', ncol=1, handletextpad=0.5, handlelength=0.7,
                                         bbox_to_anchor=(1.05, 0.0))
            axs[i].add_artist(first_legend)
            axs[i].legend(handles_2[-2:], labels_2[-2:], loc='lower left', markerscale=2, fontsize=11,
                          edgecolor='black', ncol=1, handletextpad=0.5, handlelength=0.7, bbox_to_anchor=(1.05, -0.25))

        if i == 2:
            axin2 = inset_axes(axs[i], width=1.2, height=0.6, borderpad=0, bbox_transform=axs[i].transAxes,
                               bbox_to_anchor=(1, 0.7))
            axin2.set_xlim(0.6, 0.8)
            axin2.set_ylim(0.88, 0.98)
            axin2 = sns.scatterplot(x='Sensitivity', y='Precision',
                                    hue='Tool',  # different colors by group
                                    style='Rank',  # different shapes by group
                                    hue_order=order,
                                    style_order=rank_order,
                                    edgecolor='black',
                                    palette=colors,
                                    markers=markers,
                                    s=marker_size,  # marker size
                                    data=data[i], ax=axin2)
            # different shapes by group
            mark_inset(axs[i], axin2, loc1=1, loc2=2, fc="none", ec="0", lw=1)
            axin2.legend_.remove()
            for spine in axin2.spines.values():
                spine.set_edgecolor('black')
            axin2.set_xlabel('')
            axin2.set_ylabel('')

            axin2.xaxis.set_ticks(np.arange(0.6, 0.8 + 0.01, 0.05))
            axin2.yaxis.set_ticks(np.arange(0.88, 0.98 + 0.01, 0.05))
            xtick_labels = axin2.get_xticklabels()
            ytick_labels = axin2.get_yticklabels()
            for label in range(len(xtick_labels) - 1):
                xtick_labels[label] = ''
            for label in range(len(ytick_labels) - 1):
                ytick_labels[label] = ''
            xtick_labels[0] = str(0.6)
            ytick_labels[0] = str(0.88)
            axin2.set_xticklabels(xtick_labels)
            axin2.set_yticklabels(ytick_labels)

        if i < 3:
            axins = inset_axes(axs[i], loc=zoom_locs[i], width=zoom_widths[i], height=zoom_heights[i], borderpad=1)
            axins = sns.scatterplot(x='Sensitivity', y='Precision',
                                    hue='Tool',  # different colors by group
                                    style='Rank',  # different shapes by group
                                    hue_order=order,
                                    style_order=rank_order,
                                    edgecolor='black',
                                    palette=colors,
                                    markers=markers,
                                    s=marker_size,  # marker size
                                    data=data[i], ax=axins)

            # Adjust the size of the zoomed-in plot
            axins.set_xlim(zoom_xmins[i], zoom_xmaxs[i])
            axins.set_ylim(zoom_ymins[i], zoom_ymaxs[i])

            # Draw a box around the inset axes in the parent axes and
            mark_inset(axs[i], axins, loc1=zoom_loc1s[i], loc2=zoom_loc2s[i], fc="none", ec="0", lw=1)
            axins.tick_params(color='black')
            for spine in axins.spines.values():
                spine.set_edgecolor('black')
            # fix the number of ticks on the inset axes
            # axins.yaxis.get_major_locator().set_params(nbins=1)
            # axins.xaxis.get_major_locator().set_params(nbins=5)

            # Remove legend
            axins.legend_.remove()

            # Remove x and y labels
            axins.set_xlabel('')
            axins.set_ylabel('')

            axins.xaxis.set_ticks(np.arange(zoom_xmins[i], zoom_xmaxs[i] + 0.01, 0.05))
            axins.yaxis.set_ticks(np.arange(zoom_ymins[i], zoom_ymaxs[i] + 0.01, 0.05))
            xtick_labels = axins.get_xticklabels()
            ytick_labels = axins.get_yticklabels()
            for label in range(len(xtick_labels) - 1):
                xtick_labels[label] = ''
            for label in range(len(ytick_labels) - 1):
                ytick_labels[label] = ''
            xtick_labels[0] = str(zoom_xmins[i])
            # xtick_labels[-1] = str(zoom_xlims_max[i][j])
            ytick_labels[0] = str(zoom_ymins[i])
            # ytick_labels[-1] = str(1)
            axins.set_xticklabels(xtick_labels)
            axins.set_yticklabels(ytick_labels)

            # axins.set_yticklabels(['', ''])
            # axins.set_xticklabels(['', '', '', '', '', ''])
            # axins.text(0.99, 0.9, '1.0', ha='right', va='bottom', fontsize=15, fontfamily='Arial')
            # # axins.text(0.51, 0.9, '0.5', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
            # # axins.text(0.5, 0.9, '0.9', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
            # axins.text(0.505, 0.998, '1.0', ha='left', va='top', fontsize=15, fontfamily='Arial')
            # axins.annotate('0.5', xy=(0.5, 0.9), xytext=(0.54, 0.9), fontsize=15, fontfamily='Arial', va='bottom',
            #                ha='left', arrowprops=dict(arrowstyle='-', color='black'))
            # axins.annotate('0.9', xy=(0.5, 0.9), xytext=(0.505, 0.908), fontsize=15, fontfamily='Arial', va='bottom',
            #                ha='left', arrowprops=dict(arrowstyle='-', color='black'))

    # Remove plot in axs[1,1]
    axs[3].remove()

    # Add x tick labels to axs[0,1] and make visible
    axs[1].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    # axs[1].xaxis.set_ticklabels([0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0])
    axs[1].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    # axs[1].yaxis.set_ticklabels([0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0])
    # axs[0, 1].set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    # axs[0, 1].set_xticklabels(['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    axs[0].tick_params(axis='x', which='both', labelbottom=True)

    # Set common x and y labels
    # fig.text(0.28, 0.04, 'Sensitivity', ha='center', va='center', fontsize=15, fontweight='bold', fontfamily='Arial')
    # fig.text(0.03, 0.5, 'Precision', ha='center', va='center', rotation='vertical', fontsize=15, fontweight='bold',
    #          fontfamily='Arial')

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9)
    plt.savefig('./revision/extended_figure2.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    extended_figure2()
