import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes


def hiv_and_gtdb() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    # change the width of grid lines
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1.2

    # Read hiv exclusion and inclusion data
    hiv_incl = pd.read_csv('hiv-inclusion.tsv', sep='\t')
    hiv_excl = pd.read_csv('hiv-exclusion.tsv', sep='\t')

    # Read gtdb exclusion and inclusion data
    gtdb_incl = pd.read_csv('gtdb_inclusion.tsv', sep='\t')
    # gtdb_incl = gtdb_incl[gtdb_incl['Rank'] == 'Subspecies']
    gtdb_excl = pd.read_csv('gtdb_exclusion.tsv', sep='\t')
    gtdb_excl = gtdb_excl[gtdb_excl['Rank'] == 'Genus']

    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    markers = ['o', 's', 'H', 'D', 'v', 'P', 'X', 'd']

    colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']

    # Set figure size
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(10, 10))

    # fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('Sensitivity', fontsize=15, fontweight='bold', fontfamily='Arial')
    # plt.ylabel('Precision', fontsize=15, fontweight='bold', fontfamily='Arial')

    marker_size = 180

    # DATA
    data = [[hiv_incl, hiv_excl],
            [gtdb_incl, gtdb_excl]]
    # Subplot titles
    titles = [['HIV-1 Inclusion Test', 'HIV-1 Exclusion Test'],
              ['GTDB Inclusion Test', 'GTDB Exclusion Test']]
    # Panel labels
    labels = [['a', 'b'],
              ['c', 'd']]
    x_pos = 0
    y_pos = 1.04

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
            axs[i, j].set_title(titles[i][j], fontsize=18, fontweight='bold', fontfamily='Arial')

            # Add panel labels
            axs[i, j].text(x_pos, y_pos, labels[i][j], fontsize=18, fontweight='bold', fontfamily='Arial')

            # Remove x and y labels
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')

            # Remove top and right spines
            axs[i, j].spines[['right', 'top']].set_visible(False)

            # Remove legend
            if i != 0 or j != 0:
                axs[i, j].legend_.remove()
            else:
                axs[i, j].legend(loc='lower right', markerscale=2, edgecolor='black', fontsize=14)
                handles, labels2 = axs[i, j].get_legend_handles_labels()
                for h in handles:
                    h.set_edgecolor('black')
                first_legend = axs[i, j].legend(handles, labels2, loc='lower right', markerscale=2, fontsize=14,
                                                edgecolor='black')
                axs[i, j].add_artist(first_legend)

    pa1 = Patch(facecolor=colors[3], edgecolor='white')
    pb1 = Patch(facecolor=colors[5], edgecolor='white')
    axs[0, 0].legend(handles=[pa1, pb1],
                     labels=['DNA-based', 'Protein-based'],
                     ncol=1, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                     loc='lower left', fontsize=14, edgecolor='black')
    # Add a line in axs[1,0] to separate species and subspecies
    axs[1, 0].axvline(x=0.45, color='black', linestyle='--', linewidth=1.5)
    axs[1, 0].text(0.08, 0.6, 'Subspecies', fontsize=15, fontweight='bold', fontfamily='Arial')
    axs[1, 0].text(0.63, 0.6, 'Species', fontsize=15, fontweight='bold', fontfamily='Arial')

    # Set common x and y labels
    fig.text(0.5, 0.04, 'Sensitivity', ha='center', va='center', fontsize=20, fontweight='bold', fontfamily='Arial')
    fig.text(0.04, 0.5, 'Precision', ha='center', va='center', rotation='vertical', fontsize=20, fontweight='bold',
             fontfamily='Arial')

    axs[0, 0].set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()


def cami_gtdb() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1.2

    # Load Data
    soil = pd.read_csv('cami-gtdb_soil.tsv', sep='\t')
    marine = pd.read_csv('cami-gtdb_marine.tsv', sep='\t')
    strain = pd.read_csv('cami-gtdb_strain.tsv', sep='\t')

    # Set figure size
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(10, 10))

    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    rank_order = ['Genus', 'Species']
    markers = ['o', 'X']
    colors = ['#D81B1B', '#E51EC3', 'darkorange', 'gold', 'navajowhite', 'darkgreen', 'turquoise', 'darkseagreen']
    # colors = [ '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    marker_size = 180

    # DATA
    data = [[strain, marine],
            [soil]]

    # Panel labels
    labels = [['a', 'b'],
              ['c', 'd']]
    x_pos = 0
    y_pos = 1.04

    # Set x and y limits
    axs[0, 0].set_xlim(0, 1.02)
    axs[0, 0].set_ylim(0, 1.02)
    axs[0, 0].xaxis.set_ticks(np.arange(0, 1.01, 0.1))
    axs[0, 0].yaxis.set_ticks(np.arange(0, 1.01, 0.1))
    # axs[0, 1].set_xlim(0.4, 1.02)

    # Subplot titles
    titles = [['Strain-madness', 'Marine'],
              ['Plant-associated']]

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
            axs[i, j].set_title(titles[i][j], fontsize=18, fontweight='bold', fontfamily='Arial')

            # Set panel label
            print(labels[i][j])
            axs[i, j].text(x_pos, y_pos, labels[i][j], fontsize=18, fontweight='bold', fontfamily='Arial')

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
                handles, labels2 = axs[i, j].get_legend_handles_labels()
                print(handles)
                print(labels2)
                for h in handles:
                    h.set_edgecolor('black')
                first_legend = axs[i, j].legend(handles[1:-3], labels2[1:-3], loc='lower right', markerscale=2,
                                                fontsize=14, edgecolor='black', ncol=1)
                axs[i, j].add_artist(first_legend)
                axs[i, j].legend(handles[-2:], labels2[-2:], loc='lower left', markerscale=2, fontsize=14,
                                 edgecolor='black', ncol=1)

            # Zoom in [0, 1]
            if i == 0 and j == 1:
                axins = inset_axes(axs[i, j], loc='lower left', width=2.5, height=2.5, borderpad=1.5)
                axins = sns.scatterplot(x='Sensitivity', y='Precision',
                                        hue='Tool',  # different colors by group
                                        style='Rank',  # different shapes by group
                                        hue_order=order,
                                        style_order=rank_order,
                                        edgecolor='black',
                                        palette=colors,
                                        markers=markers,
                                        s=marker_size,  # marker size
                                        data=data[i][j], ax=axins)

                # Adjust the size of the zoomed-in plot
                axins.set_xlim(0.5, 1)
                axins.set_ylim(0.9, 1)

                # Draw a box around the inset axes in the parent axes and
                mark_inset(axs[i, j], axins, loc1=2, loc2=4, fc="none", ec="0", lw=1)
                axins.tick_params(color='black')
                for spine in axins.spines.values():
                    spine.set_edgecolor('black')
                # fix the number of ticks on the inset axes
                axins.yaxis.get_major_locator().set_params(nbins=1)
                axins.xaxis.get_major_locator().set_params(nbins=5)

                # Remove legend
                axins.legend_.remove()

                # Remove x and y labels
                axins.set_xlabel('')
                axins.set_ylabel('')
                axins.set_xticklabels(['', '', '', '', '', ''])
                axins.set_yticklabels(['', ''])
                axins.text(0.94, 0.9, '1.0', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
                # axins.text(0.51, 0.9, '0.5', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
                # axins.text(0.5, 0.9, '0.9', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
                axins.text(0.505, 0.998, '1.0', ha='left', va='top', fontsize=15, fontfamily='Arial')
                axins.annotate('0.5', xy=(0.5, 0.9), xytext=(0.54, 0.9), fontsize=15, fontfamily='Arial', va='bottom',
                               ha='left', arrowprops=dict(arrowstyle='-', color='black'))
                axins.annotate('0.9', xy=(0.5, 0.9), xytext=(0.505, 0.908), fontsize=15, fontfamily='Arial', va='bottom',
                               ha='left', arrowprops=dict(arrowstyle='-', color='black'))

    # Remove plot in axs[1,1]
    axs[1, 1].remove()

    # Add x tick labels to axs[0,1] and make visible
    axs[0, 1].set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    # axs[0, 1].set_xticklabels(['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
    axs[0, 1].tick_params(axis='x', which='both', labelbottom=True)

    # Set common x and y labels
    fig.text(0.28, 0.04, 'Sensitivity', ha='center', va='center', fontsize=20, fontweight='bold', fontfamily='Arial')
    fig.text(0.04, 0.5, 'Precision', ha='center', va='center', rotation='vertical', fontsize=20, fontweight='bold',
             fontfamily='Arial')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()


def hiv_gtdb_cami2() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1.2
    plt.rcParams['xtick.major.pad'] = '0'
    plt.rcParams['ytick.major.pad'] = '0'

    # Load Data
    soil = pd.read_csv('cami-gtdb_soil.tsv', sep='\t')
    marine = pd.read_csv('cami-gtdb_marine.tsv', sep='\t')
    strain = pd.read_csv('cami-gtdb_strain.tsv', sep='\t')

    # Read hiv exclusion and inclusion data
    hiv_incl = pd.read_csv('hiv-inclusion.tsv', sep='\t')
    hiv_excl = pd.read_csv('hiv-exclusion.tsv', sep='\t')

    # Read gtdb exclusion and inclusion data
    gtdb_incl = pd.read_csv('gtdb_inclusion.tsv', sep='\t')
    gtdb_excl = pd.read_csv('gtdb_exclusion.tsv', sep='\t')
    gtdb_excl = gtdb_excl[gtdb_excl['Rank'] == 'Genus']

    # Set figure size
    fig, axs = plt.subplots(2, 4, sharex='row', sharey='row', figsize=(14, 7))

    # Set horizontal and vertical spacing between subplots
    fig.subplots_adjust(hspace=0.25, wspace=0.1)

    # Params for cami2 plots
    order_cami2 = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    rank_order_cami2 = ['Genus', 'Species']
    markers_cami2 = ['o', 'X']
    colors_cami2 = ['#D81B1B', '#E51EC3', 'darkorange', 'gold', 'navajowhite', 'darkgreen', 'turquoise', 'darkseagreen']
    # colors = [ '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    marker_size = 120

    # Params for hiv & gtdb plots
    order_hiv_gtdb = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    markers_hiv_gtdb = ['o', 's', 'H', 'D', 'v', 'P', 'X', 'd']
    colors_hiv_gtdb = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']

    data = [[hiv_incl, hiv_excl, gtdb_incl, gtdb_excl],
            [strain, marine, soil]]

    titles = [['HIV-1 Inclusion Test', 'HIV-1 Exclusion Test', 'GTDB Inclusion Test', 'GTDB Exclusion Test'],
              ['Strain-madness', 'Marine', 'Plant-associated']]
    fontsize_title = 12

    # Panel labels
    labels = [['a', 'b', 'c', 'd'],
              ['e', 'f', 'g']]
    x_pos = 0
    y_pos = 1.04

    # Plot hiv & gtdb plots
    axs[0, 0].set_xlim(0, 1.03)
    axs[0, 0].set_ylim(0.4, 1.03)
    axs[0, 0].xaxis.set_ticks(np.arange(0, 1.02, 0.1))
    axs[0, 0].yaxis.set_ticks(np.arange(0.4, 1.02, 0.1))
    axs[0, 0].xaxis.set_ticklabels([0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0])
    for j in range(4):
        axs[0, j] = sns.scatterplot(x='Sensitivity', y='Precision',
                                    hue='Tool',  # different colors by group
                                    style='Tool',  # different shapes by group
                                    hue_order=order_hiv_gtdb,
                                    style_order=order_hiv_gtdb,
                                    edgecolor='black',
                                    palette=colors_hiv_gtdb,
                                    markers=markers_hiv_gtdb,
                                    s=marker_size,  # marker size
                                    data=data[0][j], ax=axs[0, j])
        # Set title
        axs[0, j].set_title(titles[0][j], fontsize=fontsize_title, fontweight='bold', fontfamily='Arial')

        # Add panel labels
        axs[0, j].text(x_pos, y_pos, labels[0][j], fontsize=fontsize_title, fontweight='bold', fontfamily='Arial')

        # Remove x and y labels
        axs[0, j].set_xlabel('')
        axs[0, j].set_ylabel('')

        # Remove top and right spines
        axs[0, j].spines[['right', 'top']].set_visible(False)

        # Remove legend
        if j != 0:
            axs[0, j].legend_.remove()
        else:
            axs[0, j].legend(loc='lower right', markerscale=2, edgecolor='black', fontsize=10)
            handles, labels2 = axs[0, j].get_legend_handles_labels()
            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[0, j].legend(handles, labels2, loc='lower right', markerscale=2, fontsize=10,
                                            edgecolor='black')
            axs[0, j].add_artist(first_legend)

    pa1 = Patch(facecolor=colors_hiv_gtdb[3], edgecolor=colors_hiv_gtdb[3])
    pb1 = Patch(facecolor=colors_hiv_gtdb[5], edgecolor=colors_hiv_gtdb[5])
    both1 = Patch(facecolor='#D81B1B', edgecolor='#D81B1B')
    both2 = Patch(facecolor='#E51EC3', edgecolor='#E51EC3')

    axs[0, 0].legend(handles=[pa1, pb1, both1, pa1, pb1, both2],
                     labels=['', '', '', 'DNA', 'Protein', 'Both'],
                     ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                     loc='lower left', fontsize=10, edgecolor='black')

    axs[0, 2].axvline(x=0.45, color='black', linestyle='--', linewidth=1.5)
    axs[0, 2].text(0.225, 0.6, 'Subspecies', fontsize=12, fontweight='bold', fontfamily='Arial',
                   verticalalignment='bottom', horizontalalignment='center')
    axs[0, 2].text(0.725, 0.6, 'Species', fontsize=12, fontweight='bold', fontfamily='Arial',
                   verticalalignment='bottom', horizontalalignment='center')



    # ---------------  Plot cami2 plots -------------------- #
    # Set x and y limits
    axs[1, 0].set_xlim(0, 1.02)
    axs[1, 0].set_ylim(0, 1.02)
    axs[1, 0].xaxis.set_ticks(np.arange(0, 1.01, 0.1))
    axs[1, 0].yaxis.set_ticks(np.arange(0, 1.01, 0.1))
    axs[1, 0].xaxis.set_ticklabels([0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0])
    axs[1, 0].yaxis.set_ticklabels([0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0])
    for j in range(4):
        if j == 3:
            break
        axs[1, j] = sns.scatterplot(x='Sensitivity', y='Precision',
                                    hue='Tool',  # different colors by group
                                    style='Rank',  # different shapes by group
                                    hue_order=order_cami2,
                                    style_order=rank_order_cami2,
                                    edgecolor='black',
                                    palette=colors_cami2,
                                    markers=markers_cami2,
                                    s=marker_size,  # marker size
                                    data=data[1][j], ax=axs[1, j])
        # Set title
        axs[1, j].set_title(titles[1][j], fontsize=12, fontweight='bold', fontfamily='Arial')

        # Set panel label
        print(labels[1][j])
        axs[1, j].text(x_pos, y_pos, labels[1][j], fontsize=12, fontweight='bold', fontfamily='Arial')

        # Remove x and y labels
        axs[1, j].set_xlabel('')
        axs[1, j].set_ylabel('')

        # Remove top and right spines
        axs[1, j].spines[['right', 'top']].set_visible(False)

        # Remove legend
        if j != 0:
            axs[1, j].legend_.remove()
        else:
            axs[1, j].legend(loc='lower right', markerscale=2, fontsize=10, edgecolor='black')
            handles, labels2 = axs[1, j].get_legend_handles_labels()
            print(handles)
            print(labels2)
            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[1, j].legend(handles[1:-3], labels2[1:-3], loc='lower right', markerscale=2,
                                            fontsize=10, edgecolor='black', ncol=1)
            axs[1, j].add_artist(first_legend)
            axs[1, j].legend(handles[-2:], labels2[-2:], loc='lower left', markerscale=2, fontsize=10,
                             edgecolor='black', ncol=1)

        # Zoom in [0, 1]
        if j == 1:
            axins = inset_axes(axs[1, j], loc='lower left', width=1.8, height=1.8, borderpad=0.5)
            axins = sns.scatterplot(x='Sensitivity', y='Precision',
                                    hue='Tool',  # different colors by group
                                    style='Rank',  # different shapes by group
                                    hue_order=order_cami2,
                                    style_order=rank_order_cami2,
                                    edgecolor='black',
                                    palette=colors_cami2,
                                    markers=markers_cami2,
                                    s=marker_size,  # marker size
                                    data=data[1][j], ax=axins)

            # Adjust the size of the zoomed-in plot
            axins.set_xlim(0.5, 1)
            axins.set_ylim(0.9, 1)

            # Draw a box around the inset axes in the parent axes and
            mark_inset(axs[1, j], axins, loc1=2, loc2=4, fc="none", ec="0", lw=1)
            axins.tick_params(color='black')
            for spine in axins.spines.values():
                spine.set_edgecolor('black')
            # fix the number of ticks on the inset axes
            axins.yaxis.get_major_locator().set_params(nbins=1)
            axins.xaxis.get_major_locator().set_params(nbins=5)

            # Remove legend
            axins.legend_.remove()

            # Remove x and y labels
            axins.set_xlabel('')
            axins.set_ylabel('')
            axins.set_xticklabels(['', '', '', '', '', ''])
            axins.set_yticklabels(['', ''])
            axins.text(0.99, 0.9, '1.0', ha='right', va='bottom', fontsize=12, fontfamily='Arial')
            # axins.text(0.51, 0.9, '0.5', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
            # axins.text(0.5, 0.9, '0.9', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
            axins.text(0.505, 0.998, '1.0', ha='left', va='top', fontsize=12, fontfamily='Arial')
            axins.annotate('0.5', xy=(0.5, 0.9), xytext=(0.54, 0.9), fontsize=12, fontfamily='Arial', va='bottom',
                           ha='left', arrowprops=dict(arrowstyle='-', color='black'))
            axins.annotate('0.9', xy=(0.5, 0.9), xytext=(0.505, 0.908), fontsize=12, fontfamily='Arial', va='bottom',
                           ha='left', arrowprops=dict(arrowstyle='-', color='black'))

    axs[1, 3].remove()

    # Set the font size of the tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Set common x and y labels
    fig.text(0.5, 0.05, 'Sensitivity', ha='center', va='center', fontsize=15, fontweight='bold', fontfamily='Arial')
    fig.text(0.09, 0.5, 'Precision', ha='center', va='center', rotation='vertical', fontsize=15, fontweight='bold',
             fontfamily='Arial')



    plt.show()

def covid19_in() -> None:
    # sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})

    # Load Data
    df = pd.read_csv('covid19_in.tsv', sep='\t')
    df['Correct%'] = df['Correct'] / df['Reads']
    df['Wrong/Correct%'] = df['Wrong'] / df['Correct']
    order = ['KrakenUniq', 'Kraken2', 'Centrifuge', "Metabuli-P", "Metabuli-S", 'Kraken2X', 'Kaiju', 'MMseqs2']

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
        # tilt
        tick.set_rotation(15)
        # size
        tick.set_fontsize(12)
    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Arial')

    # Remove legend
    axs[0].legend_.remove()
    axs[1].legend_.remove()

    # Shade the area of the middle tool
    axs[0].axvspan(2.5, 4.5, facecolor='lightgrey', alpha=0.5)
    axs[1].axvspan(2.5, 4.5, facecolor='lightgrey', alpha=0.5)

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
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})

    # Load Data
    patient = pd.read_csv('covid19_ex_patient.tsv', sep='\t')
    control = pd.read_csv('covid19_ex_control.tsv', sep='\t')
    control['logValue'] = np.log2(control['Value'] + 1)
    order = ['KrakenUniq', 'Kraken2', 'Centrifuge', "Metabuli-P", "Metabuli-S", 'Kraken2X', 'Kaiju', 'MMseqs2']

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

    # Add text in the top of third tool
    axs[0].text(0.3, 0.95, '[1.0, 2.8, 3.6,\n 6.0, 8.4]', transform=axs[0].transAxes, fontsize=10, fontfamily='Arial',
                verticalalignment='top', horizontalalignment='center', weight='bold')

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
    # axs[1].text(0.2, 0.5, 'DNA', transform=axs[1].transAxes, color='orange', weight='bold',
    #             fontsize=15, fontfamily='Arial', verticalalignment='bottom', horizontalalignment='center')
    #
    # # Add 'DNA-based tools' label in the middle of first three tools
    # axs[1].text(0.8, 0.5, 'Protein', transform=axs[1].transAxes, color='limegreen', weight='bold',
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
        # tilt x labels
        tick.set_rotation(15)
        # font size
        tick.set_fontsize(12)

    for tick in axs[1].get_yticklabels():
        tick.set_fontname('Arial')

    # Shade the area of the middle two tools
    axs[0].axvspan(2.5, 4.5, facecolor='lightgrey', alpha=0.5)
    axs[1].axvspan(2.5, 4.5, facecolor='lightgrey', alpha=0.5)

    # Shade the area of left half of the plot without white border

    plt.show()


def distribution() -> None:
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})

    # Load Data
    # gtdb_genus_tp = pd.read_csv('/Users/jaebeom/metabuli-dist/genus20_141_classifications.tsv.genus.tp', sep='\t',
    #                             header=None)
    # gtdb_genus_fp = pd.read_csv('/Users/jaebeom/metabuli-dist/genus20_141_classifications.tsv.genus.fp', sep='\t',
    #                             header=None)
    # gtdb_species_tp = pd.read_csv('/Users/jaebeom/metabuli-dist/strain20_141_classifications.tsv.species.tp', sep='\t',
    #                               header=None)
    # gtdb_species_fp = pd.read_csv('/Users/jaebeom/metabuli-dist/strain20_141_classifications.tsv.species.fp', sep='\t',
    #                               header=None)
    #
    # virus_genus_tp = pd.read_csv('/Users/jaebeom/metabuli-dist/genus141_classifications.tsv.genus.tp', sep='\t',
    #                              header=None)
    # virus_genus_fp = pd.read_csv('/Users/jaebeom/metabuli-dist/genus141_classifications.tsv.genus.fp', sep='\t',
    #                              header=None)
    # virus_species_tp = pd.read_csv('/Users/jaebeom/metabuli-dist/species141_classifications.tsv.species.tp', sep='\t',
    #                                header=None)
    # virus_species_fp = pd.read_csv('/Users/jaebeom/metabuli-dist/species141_classifications.tsv.species.fp', sep='\t',
    #                                header=None)
    #
    # pd.to_pickle(gtdb_genus_tp, '/Users/jaebeom/metabuli-dist/gtdb_genus_tp.pkl')
    # pd.to_pickle(gtdb_genus_fp, '/Users/jaebeom/metabuli-dist/gtdb_genus_fp.pkl')
    # pd.to_pickle(gtdb_species_tp, '/Users/jaebeom/metabuli-dist/gtdb_species_tp.pkl')
    # pd.to_pickle(gtdb_species_fp, '/Users/jaebeom/metabuli-dist/gtdb_species_fp.pkl')
    #
    # pd.to_pickle(virus_genus_tp, '/Users/jaebeom/metabuli-dist/virus_genus_tp.pkl')
    # pd.to_pickle(virus_genus_fp, '/Users/jaebeom/metabuli-dist/virus_genus_fp.pkl')
    # pd.to_pickle(virus_species_tp, '/Users/jaebeom/metabuli-dist/virus_species_tp.pkl')
    # pd.to_pickle(virus_species_fp, '/Users/jaebeom/metabuli-dist/virus_species_fp.pkl')

    gtdb_genus_tp = pd.read_pickle('/Users/jaebeom/metabuli-dist/gtdb_genus_tp.pkl')
    gtdb_genus_fp = pd.read_pickle('/Users/jaebeom/metabuli-dist/gtdb_genus_fp.pkl')
    gtdb_species_tp = pd.read_pickle('/Users/jaebeom/metabuli-dist/gtdb_species_tp.pkl')
    gtdb_species_fp = pd.read_pickle('/Users/jaebeom/metabuli-dist/gtdb_species_fp.pkl')

    virus_genus_tp = pd.read_pickle('/Users/jaebeom/metabuli-dist/virus_genus_tp.pkl')
    virus_genus_fp = pd.read_pickle('/Users/jaebeom/metabuli-dist/virus_genus_fp.pkl')
    virus_species_tp = pd.read_pickle('/Users/jaebeom/metabuli-dist/virus_species_tp.pkl')
    virus_species_fp = pd.read_pickle('/Users/jaebeom/metabuli-dist/virus_species_fp.pkl')

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
    # colors = [['gr', 'darkgreen', 'limegreen', 'limegreen'],
    #           ['coral', 'coral', 'orange', 'orange']]

    # Set y limit
    axs[0, 0].set_ylim(0, 0.5)
    axs[0, 0].xaxis.set_ticks([0, 0.15, 0.5, 1])
    axs[0, 0].xaxis.set_ticklabels([0, 0.15, 0.5, 1])

    # Second y-axis ranges
    y_ranges = [[520000, 650000, 8500000, 800000],
                [520000, 650000, 8500000, 800000]]

    secondary_ax = []
    # Plot histogram
    for i in range(2):
        for j in range(4):

            # Second y axis
            secondary_ax.append(axs[i, j].twinx())
            secondary_ax[-1].hist(data[i][j], bins=42, color='darkgrey', weights=np.ones_like(data[i][j]), alpha=0.5)
            secondary_ax[-1].set_ylim(0, y_ranges[i][j])
            # Scientific notation
            secondary_ax[-1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            axs[i, j].hist(data[i][j], bins=42, color=colors[i][j], weights=np.ones_like(data[i][j]) / len(data[i][j]),
                           alpha=1)

            axs[i, j].set_title(titles[i][j], fontfamily='Arial', fontsize=subtitle_size, weight='bold')
            axs[i, j].margins(0)
            # Add a vertical line at 0.15
            axs[i, j].axvline(x=0.15, color='black', linestyle='--', linewidth=1)

            # Add a text explaining the vertical line
            if i == 0 and j == 0:
                axs[i, j].text(0.15, 0.15, 'Min. score\nto be classified\n(0.15)', transform=axs[i, j].transAxes, fontsize=11,
                               fontweight='bold', fontfamily='Arial', va='bottom', ha='left')
                # axs[i, j].text(0.15, 0.07, '0.15', transform=axs[i, j].transAxes, fontsize=11,
                #                fontweight='bold', fontfamily='Arial', va='bottom', ha='left')


            # Add a horizontal two-headed arrow spanning from 0.15 to 1.0
            axs[i, j].annotate('', xy=(0.15, 0.4), xytext=(1, 0.4), arrowprops=dict(arrowstyle='<->', color='black'))

            # Panel label
            axs[i, j].text(x_pos, y_pos, panel_label[i][j], transform=axs[i, j].transAxes, fontsize=14,
                           fontweight='bold', fontfamily='Arial', va='bottom', ha='left')

            # Set the font style of x and y-axis tick labels
            for tick in axs[i, j].get_xticklabels():
                tick.set_fontname('Arial')
            for tick in axs[i, j].get_yticklabels():
                tick.set_fontname('Arial')

            if j == 2 or j == 3:
                # Add a vertical line at 0.5
                axs[i, j].axvline(x=0.5, color='red', linestyle='--', linewidth=1)

                # Add a text explaining the vertical line
                if j == 2 and i == 0:
                    axs[i, j].text(0.5, 0.18, 'Min. score \nto be classified\nat species rank\n(0.5)',
                                   transform=axs[i, j].transAxes, color='red', weight='bold', fontsize=11,
                                   fontfamily='Arial', verticalalignment='bottom',
                                   horizontalalignment='left')
                    # axs[i, j].text(0.5, 0.0, '0.5',
                    #                transform=axs[i, j].transAxes, color='red', weight='bold', fontsize=11,
                    #                fontfamily='Arial', verticalalignment='bottom',
                    #                horizontalalignment='left')

                # Write the percentage of the number of samples that have a score higher than 0.5
                axs[i, j].text(0.75, 0.6, f'{100 * (data[i][j] > 0.5).sum() / len(data[i][j]):.1f}%',
                               transform=axs[i, j].transAxes, color='red', weight='bold', fontsize=14,
                               fontfamily='Arial', verticalalignment='bottom',
                               horizontalalignment='center')
                axs[i, j].annotate('', xy=(0.5, 0.3), xytext=(1, 0.3),
                                   arrowprops=dict(arrowstyle='<->', color='red'))
                axs[i, j].text(0.65, 0.8, f'{100 * (data[i][j] > 0.15).sum() / len(data[i][j]):.1f}%',
                               transform=axs[i, j].transAxes, color='black', weight='bold', fontsize=14,
                               fontfamily='Arial', verticalalignment='bottom',
                               horizontalalignment='center')
            else:
                axs[i, j].text(0.6, 0.8, f'{100 * (data[i][j] > 0.15).sum() / len(data[i][j]):.1f}%',
                               transform=axs[i, j].transAxes, color='black', weight='bold', fontsize=14,
                               fontfamily='Arial', verticalalignment='bottom',
                               horizontalalignment='center')

    # Set x,y label
    fig.text(0.5, 0.03, 'Classification Score', ha='center', fontfamily='Arial', fontsize=18, weight='bold')
    fig.text(0.07, 0.5, 'Relative Frequency', va='center', rotation='vertical', fontfamily='Arial', fontsize=18,
             weight='bold')

    # Set secondary y-axis label
    fig.text(0.95, 0.5, 'Frequency (gray)', va='center', rotation='vertical', fontfamily='Arial', fontsize=18,
             weight='bold')

    # Set figure background transparent
    fig.patch.set_alpha(0)

    plt.show()


def venn():
    # Load data
    metabuli = pd.read_csv('metabuli-soil-4-tp.txt.2', sep='\t', header=None)[0].tolist()
    metabuli_15_50 = pd.read_csv('metabuli-soil-4-tp_15-50.txt', sep='\t', header=None)[0].tolist()
    kraken2 = pd.read_csv('kraken2-soil-4-tp.txt', sep='\t', header=None)[0].tolist()
    kaiju = pd.read_csv('kaiju-soil-4-tp.txt', sep='\t', header=None)[0].tolist()

    # Draw venn diagram
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    venn2([set(kaiju), set(kraken2)], set_labels=('Kaiju', 'Kraken2'), ax=axs[0])
    venn3([set(kaiju), set(kraken2), set(metabuli_15_50)], set_labels=('Kaiju', 'Kraken2', 'Metabuli'), ax=axs[1])
    venn3([set(kaiju), set(kraken2), set(metabuli)], set_labels=('Kaiju', 'Kraken2', 'Metabuli'), ax=axs[2])
    plt.show()


if __name__ == '__main__':
    # covid_ex()
    # covid19_in()
    # hiv_and_gtdb()
    # cami_gtdb()
    # distribution()
    # venn()
    hiv_gtdb_cami2()
