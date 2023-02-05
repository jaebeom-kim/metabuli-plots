import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def hiv_and_gtdb() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})

    # Read hiv exclusion and inclusion data
    hiv_incl = pd.read_csv('hiv-inclusion.tsv', sep='\t')
    hiv_excl = pd.read_csv('hiv-exclusion.tsv', sep='\t')

    # Read gtdb exclusion and inclusion data
    gtdb_incl = pd.read_csv('gtdb_inclusion.tsv', sep='\t')
    # gtdb_incl = gtdb_incl[gtdb_incl['Rank'] == 'Subspecies']
    gtdb_excl = pd.read_csv('gtdb_exclusion.tsv', sep='\t')
    gtdb_excl = gtdb_excl[gtdb_excl['Rank'] == 'Genus']

    order = ["Metabuli", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    markers = ['o', 's', 'D', 'v', 'P', 'X', 'd']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2']


    # Plot hiv and gtdb exclusion
    # Set figure size

    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(10, 10))
    # fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('Sensitivity', fontsize=15, fontweight='bold', fontfamily='Arial')
    # plt.ylabel('Precision', fontsize=15, fontweight='bold', fontfamily='Arial')

    marker_size = 100

    # DATA
    data = [[hiv_incl, hiv_excl],
            [gtdb_incl, gtdb_excl]]
    # Subplot titles
    titles = [['HIV-1 Inclusion Test', 'HIV-1 Exclusion Test'],
              ['GTDB Inclusion Test', 'GTDB Exclusion Test']]

    # Set x and y limits
    axs[0, 0].set_xlim(0, 1.02)
    axs[0, 0].set_ylim(0.4, 1.02)
    axs[0, 0].xaxis.set_ticks(np.arange(0, 1.02, 0.1))
    axs[0, 0].yaxis.set_ticks(np.arange(0.4, 1.02, 0.1))

    # Scatter Plot hiv inclusion
    for i in range(2):
        for j in range(2):
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
            # Set title
            axs[i, j].set_title(titles[i][j], fontsize=12, fontweight='bold', fontfamily='Arial')

            # Remove x and y labels
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')

            # Remove top and right spines
            axs[i, j].spines[['right', 'top']].set_visible(False)

            # Remove legend
            if i != 0 or j != 0:
                axs[i, j].legend_.remove()
            else:
                axs[i, j].legend(loc='lower right', markerscale=2)

    # Add a line in axs[1,0] to separate species and subspecies
    axs[1, 0].axvline(x=0.45, color='black', linestyle='--', linewidth=1.5)
    axs[1, 0].text(0.15, 0.6, 'Species', fontsize=12, fontweight='bold', fontfamily='Arial')
    axs[1, 0].text(0.62, 0.6, 'Subspecies', fontsize=12, fontweight='bold', fontfamily='Arial')


    # Set common x and y labels
    fig.text(0.5, 0.04, 'Sensitivity', ha='center', va='center', fontsize=15, fontweight='bold', fontfamily='Arial')
    fig.text(0.04, 0.5, 'Precision', ha='center', va='center', rotation='vertical', fontsize=15, fontweight='bold',
             fontfamily='Arial')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.show()


if __name__ == '__main__':
    hiv_and_gtdb()
