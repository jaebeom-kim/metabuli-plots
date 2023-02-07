import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch


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
    # markers = ['o', 's', 's', 's', 'X', 'X', 'X']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2']
    colors = ['#d62728', '#ff7f0e', '#ff7f0e', '#ff7f0e', '#2ca02c', '#2ca02c', '#2ca02c']

    # Set figure size
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(10, 10))
    # fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('Sensitivity', fontsize=15, fontweight='bold', fontfamily='Arial')
    # plt.ylabel('Precision', fontsize=15, fontweight='bold', fontfamily='Arial')

    marker_size = 150

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
                axs[i, j].legend(loc='lower right', markerscale=2, edgecolor='black')

    # Add a line in axs[1,0] to separate species and subspecies
    axs[1, 0].axvline(x=0.45, color='black', linestyle='--', linewidth=1.5)
    axs[1, 0].text(0.08, 0.6, 'Subspecies', fontsize=15, fontweight='bold', fontfamily='Arial')
    axs[1, 0].text(0.63, 0.6, 'Species', fontsize=15, fontweight='bold', fontfamily='Arial')

    # Set common x and y labels
    fig.text(0.5, 0.04, 'Sensitivity', ha='center', va='center', fontsize=15, fontweight='bold', fontfamily='Arial')
    fig.text(0.04, 0.5, 'Precision', ha='center', va='center', rotation='vertical', fontsize=15, fontweight='bold',
             fontfamily='Arial')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()


def cami_gtdb() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})

    # Load Data
    soil = pd.read_csv('cami-gtdb_soil.tsv', sep='\t')
    marine = pd.read_csv('cami-gtdb_marine.tsv', sep='\t')
    strain = pd.read_csv('cami-gtdb_strain.tsv', sep='\t')

    # Set figure size
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(10, 10))

    order = ["Metabuli", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    rank_order = ['Genus', 'Species']
    markers = ['o', 'X']
    colors = ['#d62728', 'darkorange', 'gold', 'navajowhite', 'darkgreen', 'turquoise', 'darkseagreen']
    marker_size = 150

    # DATA
    data = [[strain, marine],
            [soil]]

    # Set x and y limits
    axs[0, 0].set_xlim(0, 1.02)
    axs[0, 0].set_ylim(0.4, 1.02)
    axs[0, 0].xaxis.set_ticks(np.arange(0, 1.02, 0.1))
    axs[0, 0].yaxis.set_ticks(np.arange(0.4, 1.02, 0.1))

    # Subplot titles
    titles = [['Strain-madness', 'Marine'],
              ['Plant-associated']]

    # Scatter Plot hiv inclusion
    for i in range(2):
        for j in range(2):
            if i == 1 and j == 1:
                break
            axs[i, j] = sns.scatterplot(x='Sensitivity', y='Precision',
                                        hue='Tool',  # different colors by group
                                        style='Rank',  # different shapes by group
                                        hue_order=order,
                                        style_order=rank_order,
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
                axs[i, j].legend(loc='lower right', markerscale=2, fontsize=12, edgecolor='black')

    # Remove plot in axs[1,1]
    axs[1, 1].remove()

    # Add x tick labels to axs[0,1] and make visible
    axs[0, 1].set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    axs[0, 1].tick_params(axis='x', which='both', labelbottom=True)

    # Set common x and y labels
    fig.text(0.5, 0.04, 'Sensitivity', ha='center', va='center', fontsize=15, fontweight='bold', fontfamily='Arial')
    fig.text(0.04, 0.5, 'Precision', ha='center', va='center', rotation='vertical', fontsize=15, fontweight='bold',
             fontfamily='Arial')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()

def covid19_in() -> None:
    # sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})

    # Load Data
    df = pd.read_csv('covid19_in.tsv', sep='\t')
    df['Correct%'] = df['Correct'] / df['Reads']
    df['Wrong/Correct%'] = df['Wrong'] / df['Correct']
    order = ['KrakenUniq', 'Kraken2', 'Centrifuge', "Metabuli", 'Kraken2X', 'Kaiju', 'MMseqs2']

    # Set figure size
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(8, 5))
    plt.subplots_adjust(wspace=0, hspace=0)

    pa1 = Patch(facecolor='orangered', edgecolor='white')
    pa2 = Patch(facecolor='lightsalmon', edgecolor='white')
    #
    pb1 = Patch(facecolor='darkgreen', edgecolor='white')
    pb2 = Patch(facecolor='limegreen', edgecolor='white')

    # Set y limits
    axs[0].set_ylim(0, 0.15)
    axs[1].set_ylim(0.22, 0)

    # Set marker
    marker_size = 8
    correct_colors = ['orangered', 'darkgreen']
    wrong_colors = ['lightsalmon', 'limegreen']

    axs[0] = sns.stripplot(x='Tool', y='Correct%', hue='Variant',
                           data=df, order=order, ax=axs[0], palette=correct_colors, s=marker_size)
    axs[0].set_title('SARS-CoV-2 Inclusion Test', fontsize=15, fontweight='bold', fontfamily='Arial')

    # Add 'Correct variant / Total reads' label in the top right corner
    axs[0].text(0.98, 0.95, r'$\frac{Correct\;variant}{Total\;reads}$', transform=axs[0].transAxes,
                fontsize=18, fontfamily='Arial', verticalalignment='top', horizontalalignment='right')

    axs[1] = sns.stripplot(x='Tool', y='Wrong/Correct%', hue='Variant',
                           data=df, ax=axs[1], order=order, palette=wrong_colors, s=marker_size)

    # Add 'Incorrect variant / Correct variant' label in the top right corner
    axs[1].text(0.98, 0.05, r'$\frac{Incorrect\;variant}{Correct\;variant}$', transform=axs[1].transAxes,
                fontsize=18, fontfamily='Arial', verticalalignment='bottom', horizontalalignment='right')

    # Set the font style for axis tick labels
    for tick in axs[0].get_xticklabels():
        tick.set_fontname('Arial')
    for tick in axs[0].get_yticklabels():
        tick.set_fontname('Arial')

    for tick in axs[1].get_xticklabels():
        tick.set_fontname('Arial')
    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Arial')

    # Remove legend
    axs[0].legend_.remove()
    axs[1].legend_.remove()

    # Shade the area of the middle tool
    axs[0].axvspan(2.5, 3.5, facecolor='lightgrey', alpha=0.5)
    axs[1].axvspan(2.5, 3.5, facecolor='lightgrey', alpha=0.5)

    # Remove x and y labels
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')

    axs[1].legend(handles=[pa1, pb1, pa2, pb2],
                  labels=['', '', 'Omicron', 'Beta'],
                  ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                  loc='lower left', fontsize=12)

    plt.show()


def covid_ex() -> None:
    # sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})

    # Load Data
    patient = pd.read_csv('covid19_ex_patient.tsv', sep='\t')
    control = pd.read_csv('covid19_ex_control.tsv', sep='\t')
    control['logValue'] = np.log2(control['Value'] + 1)
    order = ['KrakenUniq', 'Kraken2', 'Centrifuge', "Metabuli", 'Kraken2X', 'Kaiju', 'MMseqs2']

    # Set figure size
    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(8, 5))
    plt.subplots_adjust(wspace=0, hspace=0)

    # Set y limits
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(8, 0)

    # Set marker
    marker_size = 8

    axs[0].set_title('SARS-CoV-2 Exclusion Test', fontsize=15, fontweight='bold', fontfamily='Arial')
    # Patient
    axs[0] = sns.stripplot(x='Tool', y='Value',
                           data=patient, order=order, ax=axs[0], s=marker_size, c='orangered')
    # Add 'Patient' label in the top left corner
    axs[0].text(0.02, 0.95, 'Patient', transform=axs[0].transAxes, fontsize=15, fontfamily='Arial',
                verticalalignment='top')
    # Add 'Sarbecovirus reads /Estimated SARS-CoV-2 reads' label in the top right corner
    axs[0].text(0.98, 0.95, r'$\frac{Sarbecovirus\;reads}{Estimated\;SARS-CoV-2\;reads}$', transform=axs[0].transAxes,
                fontsize=18, fontfamily='Arial', verticalalignment='top', horizontalalignment='right')

    # Control
    axs[1] = sns.stripplot(x='Tool', y='logValue',
                           data=control, ax=axs[1], order=order, s=marker_size, c='darkgreen')
    # Add 'Control' label in the bottom left corner
    axs[1].text(0.02, 0.05, 'Control', transform=axs[1].transAxes, fontsize=15, fontfamily='Arial',
                verticalalignment='bottom')
    # Add 'log2(Sarbecovirus reads + 1)' label in the bottom right corner
    axs[1].text(0.98, 0.05, r'$log_2(Sarbecovirus\;reads + 1)$', transform=axs[1].transAxes,
                fontsize=15, fontfamily='Arial', verticalalignment='bottom', horizontalalignment='right')
    # # Add 'DNA-based tools' label in the middle of first three tools
    # axs[1].text(0.2, 0.5, 'DNA-based\nsoftware', transform=axs[1].transAxes, color='orange', weight='bold',
    #             fontsize=15, fontfamily='Arial', verticalalignment='bottom', horizontalalignment='center')

    # Remove x and y labels
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')

    # Set the font style for axis tick labels
    for tick in axs[0].get_xticklabels():
        tick.set_fontname('Arial')
    for tick in axs[0].get_yticklabels():
        tick.set_fontname('Arial')

    for tick in axs[1].get_xticklabels():
        tick.set_fontname('Arial')
    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Arial')

    # Shade the area of the middle tool
    axs[0].axvspan(2.5, 3.5, facecolor='lightgrey', alpha=0.5)
    axs[1].axvspan(2.5, 3.5, facecolor='lightgrey', alpha=0.5)

    plt.show()


if __name__ == '__main__':
    # covid_ex()
    #covid19_in()
   # hiv_and_gtdb()
    cami_gtdb()