import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes


def prokaryote_all():
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['figure.figsize'] = [18, 12]
    fig, axs = plt.subplots(3, 4, sharex='all', sharey='all', figsize=(12, 10), gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    # Read data
    # Subspecies-level
    ssp_short_data = pd.read_csv('gtdb/revision/inclusion/inclusion_short.tsv', sep='\t')
    ssp_hifi_data = pd.read_csv('gtdb/revision/inclusion/inclusion_sequel_ccs.tsv', sep='\t')
    ssp_ont_data = pd.read_csv('gtdb/revision/inclusion/inclusion_ont.tsv', sep='\t')
    ssp_sequel_data = pd.read_csv('gtdb/revision/inclusion/inclusion_sequel.tsv', sep='\t')
    ssp_short_data = ssp_short_data[ssp_short_data['Rank'] == 'Subspecies']
    ssp_hifi_data = ssp_hifi_data[ssp_hifi_data['Rank'] == 'Subspecies']
    ssp_ont_data = ssp_ont_data[ssp_ont_data['Rank'] == 'Subspecies']
    ssp_sequel_data = ssp_sequel_data[ssp_sequel_data['Rank'] == 'Subspecies']

    # Species-level
    sp_short_data = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_short.tsv', sep='\t')
    sp_hifi_data = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_sequel_ccs.tsv', sep='\t')
    sp_ont_data = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_ont.tsv', sep='\t')
    sp_sequel_data = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_sequel.tsv', sep='\t')
    sp_short_data = sp_short_data[sp_short_data['Rank'] == 'Species']
    sp_hifi_data = sp_hifi_data[sp_hifi_data['Rank'] == 'Species']
    sp_ont_data = sp_ont_data[sp_ont_data['Rank'] == 'Species']
    sp_sequel_data = sp_sequel_data[sp_sequel_data['Rank'] == 'Species']

    # Genus-level
    genus_short_data = pd.read_csv('gtdb/revision/species_exclusion/sp-ex_short.tsv', sep='\t')
    genus_hifi_data = pd.read_csv('gtdb/revision/species_exclusion/sp-ex_sequel_ccs.tsv', sep='\t')
    genus_ont_data = pd.read_csv('gtdb/revision/species_exclusion/sp-ex_ont.tsv', sep='\t')
    genus_sequel_data = pd.read_csv('gtdb/revision/species_exclusion/sp-ex_sequel.tsv', sep='\t')
    genus_short_data = genus_short_data[genus_short_data['Rank'] == 'Genus']
    genus_hifi_data = genus_hifi_data[genus_hifi_data['Rank'] == 'Genus']
    genus_ont_data = genus_ont_data[genus_ont_data['Rank'] == 'Genus']
    genus_sequel_data = genus_sequel_data[genus_sequel_data['Rank'] == 'Genus']

    # Scatter plot parameters
    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2',
             'Centrifuge', 'Metamaps', 'Kraken2X',
             'Kaiju', 'MMseqs2', 'Hybrid']
    markers = ['o', 's', 'H', 'D',
               'v', ">", 'P',
               'X', 'd']
    colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208',
              '#FFC208', '#FFC208', '#38BF66',
              '#38BF66', '#38BF66', 'dimgray']

    marker_size = 100

    # Things for each panel
    titles = [['Illumina', 'PacBio HiFi', 'ONT', 'PacBio Sequel II'],
              ['Illumina', 'PacBio HiFi', 'ONT', 'PacBio Sequel II'],
              ['Illumina', 'PacBio HiFi', 'ONT', 'PacBio Sequel II']]
    data = [[ssp_short_data, ssp_hifi_data, ssp_ont_data, ssp_sequel_data],
            [sp_short_data, sp_hifi_data, sp_ont_data, sp_sequel_data],
            [genus_short_data, genus_hifi_data, genus_ont_data, genus_sequel_data]]
    panels = [['a', 'b', 'c', 'd'],
              ['e', 'f', 'g', 'h'],
              ['i', 'j', 'k', 'l']]
    x_pos = 0
    y_pos = 1.1
    zoom_xlims_min = [[0.3, 0.5, 0.4, 0.4],
                      [0.8, 0.8, 0.8, 0.7],
                      [0.4, 0.5, 0.25, 0.1]]

    zoom_xlims_max = [[0.5, 0.8, 0.7, 0.6],
                      [0.9, 1, 0.9, 0.9],
                      [0.51, 0.75, 0.47, 0.32]]

    zoom_ylims_min = [[0.9, 0.8, 0.8, 0.8],
                      [0.9, 0.9, 0.9, 0.8],
                      [0.6, 0.6, 0.55, 0.45]]

    zoom_ylims_max = [[1.01, 1.01, 1, 1],
                      [1, 1, 1, 1],
                      [0.85, 0.75, 0.65, 0.75]]

    zoom_locs = [['lower right', 4, 4, 4],
                 [4, 4, 4, 4],
                 [1, 3, 1, 1]]

    zoom_loc1s = [[2, 1, 1, 1],
                  [1, 1, 1, 1],
                  [3, 4, 2, 2]]

    zoom_loc2s = [[1, 2, 2, 2],
                  [2, 2, 2, 2],
                  [2, 2, 3, 3]]

    zoom_widths = [[1.2, 1.2, 1.2, 1.2],
                   [1.2, 1.2, 1.2, 1.2],
                   [0.8, 0.8, 0.6, 0.8]]

    zoom_heights = [[1.2, 1.2, 1.2, 1.2],
                    [1.2, 1.2, 1.2, 1.2],
                    [1.2, 1.2, 1.2, 1.2]]

    rank = ['Subspecies', 'Species', 'Genus']
    for i in range(3):
        for j in range(4):
            data[i][j] = data[i][j][data[i][j]['Tool'] != 'Metamaps_EM']
            axs[i, j].set_title(titles[i][j], fontsize=13, fontweight='bold', fontfamily='Arial')
            axs[i, j].set_xlim(0.0, 1.03)
            axs[i, j].set_ylim(0.0, 1.03)
            axs[i, j].spines[['right', 'top']].set_visible(False)
            axs[i, j].xaxis.set_ticks(np.arange(0, 1.0001, 0.2))
            axs[i, j].yaxis.set_ticks(np.arange(0, 1.0001, 0.2))
            axs[i, j].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=12, fontfamily='Arial')
            axs[i, j].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=12, fontfamily='Arial')
            if i == 2:
                axs[i, j].set_xlabel('Recall')

            axs[i, j] = sns.scatterplot(x='Sensitivity', y='Precision',
                                        hue='Tool',  # different colors by group
                                        style='Tool',  # different shapes by group
                                        hue_order=order,
                                        style_order=order,
                                        edgecolor='black',
                                        palette=colors,
                                        markers=markers,
                                        s=marker_size,  # marker size
                                        data=data[i][j], ax=axs[i, j])

            # axs[i, j].text(0.5, 0.2, rank[i], fontsize=12, fontweight='bold', fontfamily='Arial', color='gray',
            #             horizontalalignment='center', verticalalignment='top', transform=axs[i, j].transAxes)
            axs[i, j].text(x_pos, y_pos, panels[i][j], fontsize=13, fontweight='bold', fontfamily='Arial')

            # ZOOM IN
            if i < 3:
                axins = inset_axes(axs[i][j], loc=zoom_locs[i][j], width=zoom_widths[i][j],
                                   height=zoom_heights[i][j], borderpad=0.5)

                axins.tick_params(axis='both', which='major', pad=0, labelsize=12)
                axins = sns.scatterplot(x='Sensitivity', y='Precision',
                                        hue='Tool',  # different colors by group
                                        style='Tool',  # different shapes by group
                                        hue_order=order,
                                        style_order=order,
                                        edgecolor='black',
                                        palette=colors,
                                        markers=markers,
                                        s=marker_size,  # marker size
                                        data=data[i][j], ax=axins)

                # Adjust the size of the zoomed-in plot
                axins.set_xlim(zoom_xlims_min[i][j], zoom_xlims_max[i][j])
                axins.set_ylim(zoom_ylims_min[i][j], zoom_ylims_max[i][j])

                # Draw a box around the inset axes in the parent axes and
                mark_inset(axs[i][j], axins, loc1=zoom_loc1s[i][j], loc2=zoom_loc2s[i][j], fc="none", ec="0", lw=1)
                axins.tick_params(color='black')
                for spine in axins.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)

                # Remove legend
                axins.legend_.remove()

                # Remove x and y labels
                axins.set_xlabel('')
                axins.set_ylabel('')
                axins.xaxis.set_ticks(np.arange(zoom_xlims_min[i][j], zoom_xlims_max[i][j] + 0.01, 0.1))
                axins.yaxis.set_ticks(np.arange(zoom_ylims_min[i][j], zoom_ylims_max[i][j] + 0.01, 0.1))
                xtick_labels = axins.get_xticklabels()
                ytick_labels = axins.get_yticklabels()
                for label in range(len(xtick_labels)-1):
                    xtick_labels[label] = ''
                for label in range(len(ytick_labels)-1):
                    ytick_labels[label] = ''
                xtick_labels[0] = str(zoom_xlims_min[i][j])
                # xtick_labels[-1] = str(zoom_xlims_max[i][j])
                ytick_labels[0] = str(zoom_ylims_min[i][j])
                # ytick_labels[-1] = str(1)
                axins.set_xticklabels(xtick_labels)
                axins.set_yticklabels(ytick_labels)


            # Legend
            if i == 2 and j == 3:
                axs[i, j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=14, frameon=False,
                                 markerscale=1.8)
                handles, labels2 = axs[i, j].get_legend_handles_labels()
                r = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)

                labels2.insert(0, "Both")
                labels2.insert(3, "DNA-based")
                labels2.insert(8, "AA-based")

                handles.insert(0, r)
                handles.insert(3, r)
                handles.insert(8, r)

                handles = handles[3:8] + handles[8:12] + handles[0:3] + handles[12:]
                labels2 = labels2[3:8] + labels2[8:12] + labels2[0:3] + labels2[12:]

                axs[i, j].legend(handles, labels2, loc=2, markerscale=1.8, fontsize=13,
                                                handletextpad=0.5, handlelength=0.7, edgecolor='black', ncol=1,
                                                columnspacing=0.2, framealpha=0.3, bbox_to_anchor=(1.05, 1))
            else:
                axs[i, j].get_legend().remove()

    plt.tight_layout()
    plt.savefig('./revision/prokaryote_all.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    prokaryote_all()
