import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def distribution() -> None:
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})

    # # Load Data
    gtdb_genus_tp = pd.read_csv('./prokaryote/genus_sequel_ccs_1110_classifications.tsv.genus.tp', sep='\t', header=None)
    gtdb_genus_fp = pd.read_csv('./prokaryote/genus_sequel_ccs_1110_classifications.tsv.genus.fp', sep='\t', header=None)
    gtdb_species_tp = pd.read_csv('./prokaryote/strain_sequel_ccs_1110_classifications.tsv.species.tp', sep='\t', header=None)
    gtdb_species_fp = pd.read_csv('./prokaryote/strain_sequel_ccs_1110_classifications.tsv.species.fp', sep='\t', header=None)
    #
    # # Remove the second column
    # # gtdb_genus_tp = gtdb_genus_tp.drop(columns=0)
    # # gtdb_genus_fp = gtdb_genus_fp.drop(columns=0)
    # # gtdb_species_tp = gtdb_species_tp.drop(columns=0)
    # # gtdb_species_fp = gtdb_species_fp.drop(columns=0)
    #
    virus_genus_tp = pd.read_csv('./virus/genus_sequel_ccs_1120_classifications.tsv.genus.tp', sep='\t', header=None)
    virus_genus_fp = pd.read_csv('./virus/genus_sequel_ccs_1120_classifications.tsv.genus.fp', sep='\t', header=None)
    virus_species_tp = pd.read_csv('./virus/strain_sequel_ccs_1120_classifications.tsv.species.tp', sep='\t', header=None)
    virus_species_fp = pd.read_csv('./virus/strain_sequel_ccs_1120_classifications.tsv.species.fp', sep='\t', header=None)
    #
    # pd.to_pickle(gtdb_genus_tp, './pkls/illumina_gtdb_genus_tp.pkl')
    # pd.to_pickle(gtdb_genus_fp, './pkls/illumina_gtdb_genus_fp.pkl')
    # pd.to_pickle(gtdb_species_tp, './pkls/illumnia_gtdb_species_tp.pkl')
    # pd.to_pickle(gtdb_species_fp, './pkls/illumina_gtdb_species_fp.pkl')
    #
    # pd.to_pickle(virus_genus_tp, './pkls/illumina_virus_genus_tp.pkl')
    # pd.to_pickle(virus_genus_fp, './pkls/illumina_virus_genus_fp.pkl')
    # pd.to_pickle(virus_species_tp, './pkls/illumina_virus_species_tp.pkl')
    # pd.to_pickle(virus_species_fp, './pkls/illumina_virus_species_fp.pkl')


    # Set figure size
    fig, axs = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(14, 7))

    # Subtitle
    titles = [['Prokaryote', 'Virus', 'Prokaryote', 'Virus'],
              ['Prokaryote', 'Virus', 'Prokaryote', 'Virus']]
    subtitle_size = 14

    # Panel label
    panel_label = [['a', 'b', 'c', 'd'],
                   ['e', 'f', 'g', 'h']]
    x_pos = 0
    y_pos = 1.01

    # Data
    data = [[gtdb_genus_tp[0], virus_genus_tp[0], gtdb_species_tp[0], virus_species_tp[0]],
            [gtdb_genus_fp[0], virus_genus_fp[0], gtdb_species_fp[0], virus_species_fp[0]]]

    # Color
    colors = [['darkgreen', 'darkgreen', 'limegreen', 'limegreen'],
              ['coral', 'coral', 'orange', 'orange']]

    # Set y limit
    max_y = 0.3
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].set_ylim(0, max_y)
    # axs[0, 0].xaxis.set_ticks([0, 0.15, 0.5, 1])
    # axs[0, 0].xaxis.set_ticklabels([0, 0.15, 0.5, 1])

    # Second y-axis ranges
    y_ranges = [[5000, 200, 20000, 200],
                [5000, 200, 20000, 200]]

    bins = [[100, 100, 100, 100],
            [100, 100, 100, 100]]
    min_score = 0.07
    min_sp_score = 0.3
    secondary_ax = []
    # Plot histogram
    for i in range(2):
        for j in range(4):

            # Second y axis
            secondary_ax.append(axs[i, j].twinx())
            secondary_ax[-1].hist(data[i][j], bins=bins[i][j], color='darkgrey', weights=np.ones_like(data[i][j]), alpha=0.5)
            secondary_ax[-1].set_ylim(0, y_ranges[i][j])
            # Scientific notation
            secondary_ax[-1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            axs[i, j].hist(data[i][j], bins=bins[i][j], color=colors[i][j], weights=np.ones_like(data[i][j]) / len(data[i][j]),
                           alpha=1)

            axs[i, j].set_title(titles[i][j], fontfamily='Arial', fontsize=subtitle_size, weight='bold')
            axs[i, j].margins(0)

            # Add a vertical line at 0.15
            axs[i, j].axvline(x=min_score, color='black', linestyle='--', linewidth=1)

            # Add a text explaining the vertical line
            if i == 0 and j == 0:
                axs[i, j].text(min_score + 0.02, 0.12, 'Min. score\nto be classified\n(0.07)', transform=axs[i, j].transAxes, fontsize=11,
                               fontweight='bold', fontfamily='Arial', va='bottom', ha='left')
                # axs[i, j].text(0.15, 0.07, '0.15', transform=axs[i, j].transAxes, fontsize=11,
                #                fontweight='bold', fontfamily='Arial', va='bottom', ha='left')

            # Add a horizontal two-headed arrow spanning from 0.15 to 1.0
            axs[i, j].annotate('', xy=(min_score, max_y * 0.8), xytext=(1, max_y * 0.8), arrowprops=dict(arrowstyle='<->', color='black'))

            # Panel label
            axs[i, j].text(x_pos, y_pos, panel_label[i][j], transform=axs[i, j].transAxes, fontsize=14,
                           fontweight='bold', fontfamily='Arial', va='bottom', ha='left')

            # Set the font style of x and y-axis tick labels
            for tick in axs[i, j].get_xticklabels():
                tick.set_fontname('Arial')
            for tick in axs[i, j].get_yticklabels():
                tick.set_fontname('Arial')

            axs[i, j].text((min_score + 1)/2, 0.8, f'{100 * (data[i][j] > min_score).sum() / len(data[i][j]):.1f}%',
                           transform=axs[i, j].transAxes, color='black', weight='bold', fontsize=14,
                           fontfamily='Arial', verticalalignment='bottom',
                           horizontalalignment='center')

            if j == 2 or j == 3:
                # Add a vertical line at 0.5
                axs[i, j].axvline(x=min_sp_score, color='red', linestyle='--', linewidth=1)

                # Add a text explaining the vertical line
                if j == 2 and i == 0:
                    axs[i, j].text(min_sp_score+0.02, 0.012, 'Min. score \nto be classified\nat species rank\n(0.3)',
                                   transform=axs[i, j].transAxes, color='red', weight='bold', fontsize=11,
                                   fontfamily='Arial', verticalalignment='bottom',
                                   horizontalalignment='left')

                # Write the percentage of the number of samples that have a score higher than 0.5
                axs[i, j].text((1 + min_sp_score)/2, 0.6, f'{100 * (data[i][j] > min_sp_score).sum() / len(data[i][j]):.1f}%',
                               transform=axs[i, j].transAxes, color='red', weight='bold', fontsize=14,
                               fontfamily='Arial', verticalalignment='bottom',
                               horizontalalignment='center')
                axs[i, j].annotate('', xy=(min_sp_score, max_y * 0.6), xytext=(1, max_y * 0.6),
                                   arrowprops=dict(arrowstyle='<->', color='red'))

    # Set x,y label
    fig.text(0.5, 0.03, 'Sequence Similarity Score', ha='center', fontfamily='Arial', fontsize=18, weight='bold')
    fig.text(0.07, 0.5, 'Relative Frequency', va='center', rotation='vertical', fontfamily='Arial', fontsize=18,
             weight='bold')

    # Set secondary y-axis label
    fig.text(0.95, 0.5, 'Frequency (gray)', va='center', rotation='vertical', fontfamily='Arial', fontsize=18,
             weight='bold')

    # Set figure background transparent
    fig.patch.set_alpha(0)

    # Save figure
    plt.savefig('sequel_ccs_score.png', dpi=300, bbox_inches='tight', transparent=True)

    plt.show()


if __name__ == '__main__':
    distribution()