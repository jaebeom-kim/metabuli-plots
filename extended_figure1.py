import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import patches


def prokaryote_all():
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['figure.figsize'] = [16, 12]
    fig, axs = plt.subplots(3, 4, sharex='all', sharey='all', figsize=(12, 9))
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
             'Kaiju', 'MMseqs2', 'Ensemble']
    markers = ['o', 's', 'H', 'D',
               'v', ">", 'P',
               'X', 'd']
    colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208',
              '#FFC208', '#FFC208', '#38BF66',
              '#38BF66', '#38BF66', 'dimgray']

    # order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2',
    #          'Centrifuge', 'Metamaps', 'Kraken2X', 'Kaiju', 'MMseqs2', 'Ensemble']
    # markers = ['o', 's', 'H', 'D',
    #            'v', "<", ">", 'P',
    #            'X', 'd']
    # colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208',
    #           '#FFC208', '#FFC208', '#FFC208', '#38BF66',
    #           '#38BF66', '#38BF66', 'dimgray']
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
    zoom_xlims_min = [0.8, 0.8, 0.75]
    zoom_xlims_max = [0.9, 0.9, 0.85]
    zoom_ylims_min = [0.9, 0.9, 0.89]
    zoom_ylims_max = [1, 1, 0.99]
    zoom_width = [0.9, 1.8, 1.8]
    zoom_height = [1.8, 1.8, 1.8]

    # # Make subplots
    # panels = gridspec.GridSpec(3, 4)
    # gs_short = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=panels[0])
    # gs_ont = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=panels[1])
    # gs_sequel = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=panels[2])
    # short_panel = plt.subplot(gs_short[0])
    # ont_panel = plt.subplot(gs_ont[0])
    # sequel_panel = plt.subplot(gs_sequel[0])
    # axs = [short_panel, ont_panel, sequel_panel]
    #
    # ont_panel.set_xlabel('Recall (TP / # of reads)', fontsize=14, fontweight='bold', fontfamily='Arial')
    # # ont_panel.xaxis.set_label_coords(2.3, -0.15)
    # short_panel.set_ylabel('Precision (TP / TP+FP)', fontsize=14, fontweight='bold', fontfamily='Arial')
    # # gtdb_in_short.yaxis.set_label_coords(-0.15, 0)
    rank = ['Subspecies', 'Species', 'Genus']
    for i in range(3):
        for j in range(4):
            data[i][j] = data[i][j][data[i][j]['Tool'] != 'Metamaps_EM']
            axs[i, j].set_title(titles[i][j], fontsize=13, fontweight='bold', fontfamily='Arial')
            axs[i, j].set_xlim(0.0, 1.05)
            axs[i, j].set_ylim(0.0, 1.05)
            axs[i, j].spines[['right', 'top']].set_visible(False)
            axs[i, j].xaxis.set_ticks(np.arange(0, 1.0001, 0.2))
            axs[i, j].yaxis.set_ticks(np.arange(0, 1.0001, 0.2))

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

            axs[i, j].text(0.5, 0.2, rank[i], fontsize=12, fontweight='bold', fontfamily='Arial', color='gray',
                        horizontalalignment='center', verticalalignment='top', transform=axs[i, j].transAxes)
            axs[i, j].text(x_pos, y_pos, panels[i][j], fontsize=13, fontweight='bold', fontfamily='Arial')

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
