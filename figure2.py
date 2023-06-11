import os
import os.path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_venn import venn2, venn3
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes


def figure2():
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'

    # set figure size
    plt.rcParams['figure.figsize'] = [14, 3.5]

    # Read gtdb exclusion and inclusion data
    gtdb_incl = pd.read_csv('gtdb/gtdb_inclusion.tsv', sep='\t')
    gtdb_excl = pd.read_csv('gtdb/gtdb_exclusion.tsv', sep='\t')
    gtdb_excl = gtdb_excl[gtdb_excl['Rank'] == 'Genus']

    # Load COVID-19 data
    covid_in = pd.read_csv('covid19_in.tsv', sep='\t')
    covid_ex_p = pd.read_csv('covid19_ex_patient.tsv', sep='\t')
    covid_ex_c = pd.read_csv('covid19_ex_control.tsv', sep='\t')

    # Scatter plot parameters
    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    markers = ['o', 's', 'H', 'D', 'v', 'P', 'X', 'd']
    colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    marker_size = 120

    # fig, axs = plt.subplots(2, 4, figsize=(14, 3.5))
    # DATA
    data = [gtdb_incl, gtdb_excl]
    titles = ['Prokaryote Inclusion Test', 'Prokaryote Exclusion Test']
    # Panel labels
    # labels = ['a', 'b', 'c', 'd']
    labels = ['', ' ', ' ', ' ']
    x_pos = 0
    y_pos = 1.06

    # Create subplots
    # 1. GTDB inclusion & exclusion test
    # axs1 = plt.subplot(1, 4, 1)
    # axs2 = plt.subplot(1, 4, 2)
    gtdb_in_short = plt.subplot(2, 4, 1)
    gtdb_in_long = plt.subplot(2, 4, 5)
    gtdb_ex_short = plt.subplot(1, 8, 3)
    gtdb_ex_long = plt.subplot(1, 8, 4)
    # 2. COVID-19 inclusion & exclusion test
    covid_in_t_axs = plt.subplot(2, 4, 3)
    covid_in_t_axs.spines[['right', 'top']].set_visible(False)
    covid_in_b_axs = plt.subplot(2, 4, 7)
    covid_in_b_axs.spines[['right']].set_visible(False)
    covid_ex_p_axs = plt.subplot(2, 4, 4)  # , sharex=covid_in_t_axs)
    covid_ex_p_axs.spines[['right', 'top']].set_visible(False)
    covid_ex_c_axs = plt.subplot(2, 4, 8)
    covid_ex_c_axs.spines[['right']].set_visible(False)
    axs = [gtdb_in_short, gtdb_in_long, gtdb_ex_short, gtdb_ex_long]

    for j in range(4):
        if j == 0 or j == 1:
            axs[j].set_xlim(0, 1.04)
            axs[j].set_ylim(0.5, 1.04)
        else:
            axs[j].set_xlim(0, 0.6)
            axs[j].set_ylim(0.0, 1.04)

        axs[j].set_ylim(0.0, 1.04)
        axs[j].xaxis.set_ticks(np.arange(0, 1.02, 0.2))
        axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
        axs[j].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        axs[j].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        axs[j] = sns.scatterplot(x='Sensitivity', y='Precision',
                                 hue='Tool',  # different colors by group
                                 style='Tool',  # different shapes by group
                                 hue_order=order,
                                 style_order=order,
                                 edgecolor='black',
                                 palette=colors,
                                 markers=markers,
                                 s=marker_size,  # marker size
                                 data=data[j], ax=axs[j])
        # Set title
        axs[j].set_title(titles[j], fontsize=12, fontweight='bold', fontfamily='Arial')

        # Add panel labels
        axs[j].text(x_pos, y_pos, labels[j], fontsize=12, fontweight='bold', fontfamily='Arial')

        # Remove x and y labels
        # axs[j].set_xlabel('')
        # axs[j].set_ylabel('')

        # Remove top and right spines
        axs[j].spines[['right', 'top']].set_visible(False)

        # # Add F1 score contour
        # x = np.linspace(0, 1, 100)
        # y = np.linspace(0, 1, 100)
        # X, Y = np.meshgrid(x, y)
        # Z = 2 * X * Y / (X + Y)
        # axs[j].contour(X, Y, Z, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], colors='grey', linestyles='dashed', linewidths=0.5)

        # Remove legend
        if j != 1:
            axs[j].legend_.remove()
        else:
            axs[j].legend(loc='lower right', markerscale=2, edgecolor='black', fontsize=10)
            handles, labels2 = axs[j].get_legend_handles_labels()
            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[j].legend(handles, labels2, loc='lower right', markerscale=2, fontsize=10,
                                         handletextpad=0.5, handlelength=0.7, edgecolor='black', ncol=2,
                                         columnspacing=0.2)
            axs[j].add_artist(first_legend)

    # Axis labels for GTDB
    axs1.set_xlabel('Recall (TP / # of reads)', fontsize=14, fontweight='bold', fontfamily='Arial')
    axs1.set_ylabel('Precision (TP / TP+FP)', fontsize=14, fontweight='bold', fontfamily='Arial')
    axs1.xaxis.set_label_coords(1.1, -0.15)
    axs2.set_ylabel('')
    axs2.set_xlabel('')
    axs2.set_yticklabels(['', '', '', '', '', ''])
    for tick in axs1.get_xticklabels() + axs2.get_xticklabels():
        tick.set_fontname('Arial')
        # tick.set_fontsize(12)
    for tick in axs1.get_yticklabels() + axs2.get_yticklabels():
        tick.set_fontname('Arial')
        # tick.set_fontsize(12)

    # Legend for DNA, AA, and both
    pa1 = Patch(facecolor=colors[3], edgecolor=colors[3])
    pb1 = Patch(facecolor=colors[5], edgecolor=colors[5])
    both1 = Patch(facecolor='#D81B1B', edgecolor='#D81B1B')
    both2 = Patch(facecolor='#E51EC3', edgecolor='#E51EC3')

    axs[1].legend(handles=[pa1, pb1, both1, pa1, pb1, both2],
                  labels=['', '', '', 'DNA', 'AA', 'Both'],
                  ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                  loc='lower left', fontsize=10, edgecolor='black')

    # Add a line for subspecies and species
    axs[0].axvline(x=0.45, color='black', linestyle='-', linewidth=1.5)
    axs[0].text(0.04, 0.3, 'Subspecies', fontsize=12, fontweight='bold', fontfamily='Arial')
    axs[0].text(0.63, 0.3, 'Species', fontsize=12, fontweight='bold', fontfamily='Arial')

    axs[1].text(0.01, 0.93, 'Genus', fontsize=12, fontweight='bold', fontfamily='Arial')
    plt.subplots_adjust(hspace=0)


    # ----------------------------------------------- COVID-19 --------------------------------------------------------#
    # Use rows with Kraken2, Kaiju, and Metabuli
    # covid_in = covid_in[covid_in['Tool'].isin(['Kraken2', 'Metabuli', 'Kaiju'])]
    covid_order = ['Kraken2', 'Metabuli', 'Kaiju']
    covid_order = ['KrakenUniq', 'Kraken2', 'Centrifuge', 'Metabuli', 'Kraken2X', 'Kaiju', 'MMseqs2']
    covid_order = covid_order
    covid_in['Recall'] = covid_in['Correct'] / covid_in['Reads']
    covid_in['Predicted'] = covid_in['Correct'] + covid_in['Wrong']
    covid_in['Precision'] = covid_in['Correct'] / covid_in['Predicted']
    covid_in['1-Precision'] = 1 - covid_in['Precision']

    covid_in['FP/TP'] = covid_in['Wrong'] / covid_in['Correct']
    covid_in['FP'] = covid_in['Wrong']
    covid_in_omicron = covid_in[covid_in['Variant'] == 'Omicron']
    covid_in_beta = covid_in[covid_in['Variant'] == 'Beta']

    marker_size = 8
    # colors = ['#FFC208', '#38BF66', '#D81B1B']
    colors = ['#FFC208', '#FFC208', '#FFC208', '#D81B1B', '#38BF66', '#38BF66', '#38BF66']
    correct_colors = ['orangered', 'darkgreen']
    wrong_colors = ['sandybrown', 'sandybrown', 'sandybrown', 'salmon', 'darkseagreen', 'darkseagreen', 'darkseagreen']
    covid_in_t_axs.set_ylim(0, 0.15)
    covid_in_b_axs.set_ylim(0.25, 0)

    # ------------ COVID-19 inclusion test TOP ------------------#
    covid_in_t_axs = sns.stripplot(x='Tool', y='Recall', hue='Tool', palette=colors, edgecolor='black',
                                   data=covid_in_omicron, order=covid_order, ax=covid_in_t_axs, linewidth=0.8,
                                   s=marker_size, jitter=0.1, hue_order=covid_order)
    covid_in_t_axs = sns.stripplot(x='Tool', y='Recall', hue='Tool', marker='X', palette=colors, linewidth=0.8,
                                   data=covid_in_beta, order=covid_order, ax=covid_in_t_axs, edgecolor='black',
                                   s=marker_size, jitter=0.1, hue_order=covid_order)
    covid_in_t_axs.set_title('SARS-CoV-2 Inclusion Test', fontsize=12, fontweight='bold', fontfamily='Arial')
    covid_in_t_axs.set_xlabel('')
    covid_in_t_axs.set_xticklabels(['', '', '', '', '', '', ''])
    covid_in_t_axs.set_ylabel('')
    covid_in_t_axs.legend_.remove()
    covid_in_t_axs.set_yticklabels(['', '0.05', '0.10', '0.15'])
    # Add 'Sensitivity' text in the top right corner
    covid_in_t_axs.text(1, 0.95, 'Recall', fontsize=14, fontfamily='Arial',
                        horizontalalignment='right', verticalalignment='top', transform=covid_in_t_axs.transAxes)
    # covid_in_t_axs.text(-0.5, 0.16, 'c', fontsize=12, fontweight='bold', fontfamily='Arial')

    # ------------ COVID-19 inclusion test BOTTOM ------------------#
    covid_in_b_axs = sns.stripplot(x='Tool', y='1-Precision', hue='Tool', alpha=1,
                                   data=covid_in_omicron, ax=covid_in_b_axs, order=covid_order, palette=wrong_colors,
                                   s=marker_size, edgecolor='black', linewidth=0.8, jitter=0.1, hue_order=covid_order)
    covid_in_b_axs = sns.stripplot(x='Tool', y='1-Precision', hue='Tool', marker='X', alpha=1,
                                   data=covid_in_beta, ax=covid_in_b_axs, order=covid_order, palette=wrong_colors,
                                   s=marker_size, edgecolor='black', linewidth=0.8, jitter=0.1, hue_order=covid_order)
    covid_in_b_axs.set_xlabel('')
    covid_in_b_axs.set_ylabel('')
    covid_in_b_axs.legend_.remove()
    covid_in_b_axs.set_xticklabels(['KrakenUniq', 'Kraken2', 'Centrifuge', 'Metabuli', 'Kraken2X', 'Kaiju', 'MMseqs2'])
    covid_in_b_axs.text(1, 0.0, '1-Precision', fontsize=14, fontfamily='Arial',
                        horizontalalignment='right', verticalalignment='bottom', transform=covid_in_b_axs.transAxes)
    plt.setp(covid_in_b_axs.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontfamily='Arial',
             fontsize=11)

    # Add a custom legend in the bottom plot : o = Omicron, x = Beta
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Omicron', markerfacecolor='dimgrey',
                              markersize=10, markeredgewidth=0.8, markeredgecolor='black'),
                       Line2D([0], [0], marker='X', color='w', label='Beta', markerfacecolor='dimgrey', markersize=10,
                              markeredgewidth=0.8, markeredgecolor='black')]
    covid_in_b_axs.legend(handles=legend_elements, loc='lower left', handletextpad=0.5, handlelength=0.7,
                          ncol=2, columnspacing=0.5, prop={'family': 'Arial', 'size': 12}, edgecolor='black')
    # , bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12,
    # frameon=False, facecolor='white', edgecolor='white', prop={'family': 'Arial'})

    # -------------------- COVID-19 exclusion test ---------------------- #
    # covid_ex_p = covid_ex_p[covid_ex_p['Tool'].isin(['Kraken2', 'Metabuli', 'Kaiju'])]
    # covid_ex_c = covid_ex_c[covid_ex_c['Tool'].isin(['Kraken2', 'Metabuli', 'Kaiju'])]
    covid_ex_c['log(FP)'] = np.log2(covid_ex_c['Value'] + 1)
    covid_ex_p['Sentitivity'] = covid_ex_p['Value']
    covid_ex_c['FP'] = covid_ex_c['Value']
    covid_ex_p_axs.set_ylim(0, 0.75)
    covid_ex_c_axs.set_ylim(8, 0)

    # COVID-19 exclusion test TOP
    covid_ex_p_axs = sns.stripplot(x='Tool', y='Sentitivity', hue='Tool', palette=colors, edgecolor='black',
                                   data=covid_ex_p, order=covid_order, ax=covid_ex_p_axs, linewidth=0.8,
                                   s=marker_size, hue_order=covid_order)
    covid_ex_p_axs.set_title('SARS-CoV-2 Exclusion Test', fontsize=12, fontweight='bold', fontfamily='Arial')
    # Add a dot at the top of 3rd column
    covid_ex_p_axs.plot(2, 0.75, marker='o', markersize=8, color='#FFC208', markeredgecolor='black',
                        markeredgewidth=0.8)
    covid_ex_p_axs.text(0.35, 0.93, 'y=[1.0, 2.8,\n 3.6, 6.0, 8.4]', transform=covid_ex_p_axs.transAxes, fontsize=8,
                        fontfamily='Arial',
                        verticalalignment='top', horizontalalignment='center', weight='bold', color='black')

    covid_ex_p_axs.text(0.01, 0.97, 'Patient', transform=covid_ex_p_axs.transAxes, fontsize=13, fontfamily='Arial',
                        verticalalignment='top', horizontalalignment='left')
    covid_ex_p_axs.set_xlabel('')
    covid_ex_p_axs.set_xticklabels(['', '', '', '', '', '', ''])
    covid_ex_p_axs.set_ylabel('')
    covid_ex_p_axs.legend_.remove()
    covid_ex_p_axs.text(1, 0.95, 'Recall', fontsize=14, fontfamily='Arial',
                        horizontalalignment='right', verticalalignment='top', transform=covid_ex_p_axs.transAxes)
    # covid_ex_p_axs.text(-0.5, 0.8, 'd', fontsize=12, fontweight='bold', fontfamily='Arial')

    covid_ex_p_axs.set_yticklabels(['', '0.25', '0.5', '0.75'])

    # COVID-19 exclusion test BOTTOM
    covid_ex_c_axs = sns.stripplot(x='Tool', y='log(FP)', hue='Tool', alpha=1,
                                   data=covid_ex_c, ax=covid_ex_c_axs, order=covid_order, palette=wrong_colors,
                                   s=marker_size, edgecolor='black', linewidth=0.8, hue_order=covid_order)

    covid_ex_c_axs.set_xlabel('')
    covid_ex_c_axs.set_ylabel('')
    covid_ex_c_axs.text(1, 0.0, r'$log_2(FP)$', fontsize=14, fontfamily='Arial', fontstyle='normal',
                        horizontalalignment='right', verticalalignment='bottom', transform=covid_ex_c_axs.transAxes)
    covid_ex_c_axs.legend_.remove()
    covid_ex_c_axs.text(0.01, 0.03, 'Control', transform=covid_ex_c_axs.transAxes, fontsize=13, fontfamily='Arial',
                        verticalalignment='bottom', horizontalalignment='left')

    plt.setp(covid_ex_c_axs.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontfamily='Arial',
             fontsize=11)

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9)
    plt.savefig('./plots/gtdb_covid.png', dpi=300, bbox_inches='tight')
    plt.show()


def cami_gtdb() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'


    # Load Data
    soil = pd.read_csv('cami-gtdb_soil.tsv', sep='\t')
    marine = pd.read_csv('cami-gtdb_marine.tsv', sep='\t')
    strain = pd.read_csv('cami-gtdb_strain.tsv', sep='\t')

    # Set figure size
    fig, axs = plt.subplots(1, 4, sharex='all', sharey='all', figsize=(14, 3.5))
    # order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    # markers = ['o', 's', 'H', 'D', 'v', 'P', 'X', 'd']
    # colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    # marker_size = 120

    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    rank_order = ['Genus', 'Species']
    markers = ['o', 'X']
    colors = ['#D81B1B', '#E51EC3', 'darkorange', 'gold', 'navajowhite', 'darkgreen', 'turquoise', 'darkseagreen']
    # colors = [ '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    marker_size = 120

    # DATA
    # data = [strain, marine, soil]
    data = [strain, marine, soil]
    # Panel labels
    labels = ['', '', '', '']
    x_pos = 0
    y_pos = 1.04

    # Set x and y limits
    axs[0].set_xlim(0, 1.04)
    axs[0].set_ylim(0, 1.04)
    axs[0].xaxis.set_ticks(np.arange(0, 1.02, 0.2))
    axs[0].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
    # axs[0, 1].set_xlim(0.4, 1.02)

    # Subplot titles
    titles = ['Strain-madness', 'Marine', 'Plant-associated']
    # titles = ['CAMI2  Soil', 'Marine', 'Plant-associated']
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
        if i != 0:
            axs[i].legend_.remove()
        else:
            axs[i].legend(loc='lower right', markerscale=2, fontsize=12, edgecolor='black')
            handles, labels2 = axs[i].get_legend_handles_labels()
            print(handles)
            print(labels2)
            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[i].legend(handles[1:-3], labels2[1:-3], loc='lower right', markerscale=2,
                                         fontsize=11, edgecolor='black', ncol=1, handletextpad=0.5, handlelength=0.7)
            axs[i].add_artist(first_legend)
            axs[i].legend(handles[-2:], labels2[-2:], loc='lower left', markerscale=2, fontsize=11,
                          edgecolor='black', ncol=1, handletextpad=0.5, handlelength=0.7)

        # Zoom in [0, 1]
        if i == 1:
            axins = inset_axes(axs[i], loc='lower left', width=2, height=2, borderpad=0.5)
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
            axins.set_xlim(0.5, 1)
            axins.set_ylim(0.9, 1)

            # Draw a box around the inset axes in the parent axes and
            mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="0", lw=1)
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
            axins.set_yticklabels(['', ''])
            axins.set_xticklabels(['', '', '', '', '', ''])
            axins.text(0.99, 0.9, '1.0', ha='right', va='bottom', fontsize=15, fontfamily='Arial')
            # axins.text(0.51, 0.9, '0.5', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
            # axins.text(0.5, 0.9, '0.9', ha='left', va='bottom', fontsize=15, fontfamily='Arial')
            axins.text(0.505, 0.998, '1.0', ha='left', va='top', fontsize=15, fontfamily='Arial')
            axins.annotate('0.5', xy=(0.5, 0.9), xytext=(0.54, 0.9), fontsize=15, fontfamily='Arial', va='bottom',
                           ha='left', arrowprops=dict(arrowstyle='-', color='black'))
            axins.annotate('0.9', xy=(0.5, 0.9), xytext=(0.505, 0.908), fontsize=15, fontfamily='Arial', va='bottom',
                           ha='left', arrowprops=dict(arrowstyle='-', color='black'))



    # Remove plot in axs[1,1]
    axs[3].remove()

    # # Add a line for subspecies and species
    # axs[0].axvline(x=0.45, color='black', linestyle='-', linewidth=1.5)
    # axs[0].text(0.1, 0.3, 'Species', fontsize=12, fontweight='bold', fontfamily='Arial')
    # axs[0].text(0.65, 0.3, 'Genus', fontsize=12, fontweight='bold', fontfamily='Arial')

    # Add x tick labels to axs[0,1] and make visible
    axs[1].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    axs[1].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    axs[0].tick_params(axis='x', which='both', labelbottom=True)

    # Set common x and y labels
    # fig.text(0.28, 0.04, 'Sensitivity', ha='center', va='center', fontsize=15, fontweight='bold', fontfamily='Arial')
    # fig.text(0.03, 0.5, 'Precision', ha='center', va='center', rotation='vertical', fontsize=15, fontweight='bold',
    #          fontfamily='Arial')

    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9)
    plt.savefig('./plots/cami2_gtdb.png', dpi=300)
    plt.show()


def gtdb_longread():
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'

    # set figure size
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(7, 7))

    # Load data
    ont_in = pd.read_csv('./gtdb/longread/gtdb_inclusion_ont.tsv', sep='\t')
    ont_in_species = ont_in[ont_in['Rank'] == 'Species']
    ont_in_subspecies = ont_in[ont_in['Rank'] == 'Subspecies']
    ont_ex = pd.read_csv('./gtdb/longread/gtdb_exclusion_ont.tsv', sep='\t')
    ont_ex_family = ont_ex[ont_ex['Rank'] == 'Family']
    ont_ex_genus = ont_ex[ont_ex['Rank'] == 'Genus']
    sequel_in = pd.read_csv('./gtdb/longread/gtdb_inclusion_sequel.tsv', sep='\t')

    sequel_ex = pd.read_csv('./gtdb/longread/gtdb_exclusion_sequel.tsv', sep='\t')

    # Scatter plot parameters
    order = ["Metabuli", 'KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2']
    markers = ['o', 'H', 'D', 'v', 'P', 'X', 'd']
    colors = ['#D81B1B', '#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66']
    marker_size = 120

    data = [[ont_in, ont_ex],
            [ont_ex_family, ont_ex_genus]]
    titles = [['Prokaryote Inclusion ONT\n Species', 'Prokaryote Inclusion ONT\n Subspecies'],
              ['Prokaryote Exclusion ONT\n Family', 'Prokaryote Exclusion ONT\n genus']]
    # titles = [['ONT inclusion', 'ONT exclusion'],
    #           ['Sequel inclusion', 'Sequel exclusion']]
    labels = [[' ', ' '],
              [' ', ' ']]
    x_pos = 0
    y_pos = 1.06

    for i in range(2):
        for j in range(2):
            axs[i][j].set_xlim(0, 1.04)
            axs[i][j].set_ylim(0.0, 1.04)
            axs[i][j].xaxis.set_ticks(np.arange(0, 1.02, 0.2))
            axs[i][j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
            axs[i, j].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            axs[i, j].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            axs[i, j] = sns.scatterplot(x='Sensitivity', y='Precision',
                                 hue='Tool',  # different colors by group
                                 style='Tool',  # different shapes by group
                                 hue_order=order,
                                 style_order=order,
                                 edgecolor='black',
                                 palette=colors,
                                 markers=markers,
                                 s=marker_size,  # marker size
                                 data=data[i][j], ax=axs[i][j])
            # Set title
            axs[i][j].set_title(titles[i][j], fontsize=12, fontweight='bold', fontfamily='Arial')

            # Add panel labels
            axs[i][j].text(x_pos, y_pos, labels[i][j], fontsize=12, fontweight='bold', fontfamily='Arial')

            # Remove x and y labels
            axs[i][j].set_xlabel('')
            axs[i][j].set_ylabel('')

            # Remove top and right spines
            axs[i][j].spines[['right', 'top']].set_visible(False)

            # # Add F1 score contour
            # x = np.linspace(0, 1, 100)
            # y = np.linspace(0, 1, 100)
            # X, Y = np.meshgrid(x, y)
            # Z = 2 * X * Y / (X + Y)
            # axs[j].contour(X, Y, Z, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], colors='grey', linestyles='dashed', linewidths=0.5)

            # Remove legend
            if not (j == 0 and i == 0):
                axs[i][j].legend_.remove()
            else:
                axs[i][j].legend(loc='lower right', markerscale=2, edgecolor='black', fontsize=14)
                handles, labels2 = axs[i][j].get_legend_handles_labels()
                for h in handles:
                    h.set_edgecolor('black')
                first_legend = axs[i][j].legend(handles, labels2, loc='lower right', markerscale=2, fontsize=12,
                                                handletextpad=0.5, handlelength=0.7, edgecolor='black', ncol=2,
                                                columnspacing=-0.5)
                axs[i][j].add_artist(first_legend)

    # plt.subplots_adjust()
    fig.text(0.5, 0.05, 'Recall (TP / # of reads)', ha='center', va='center', fontsize=13, fontweight='bold', fontfamily='Arial')
    fig.text(0.05, 0.5, 'Precision (TP / TP+FP)', ha='center', va='center', rotation='vertical', fontsize=13, fontweight='bold',
             fontfamily='Arial')

    # Legend for DNA, AA, and both
    pa1 = Patch(facecolor=colors[3], edgecolor=colors[3])
    pb1 = Patch(facecolor=colors[5], edgecolor=colors[5])
    both1 = Patch(facecolor='#D81B1B', edgecolor='#D81B1B')
    both2 = Patch(facecolor='#D81B1B', edgecolor='#D81B1B')

    axs[0, 0].legend(handles=[pa1, pb1, both1, pa1, pb1, both2],
                     labels=['', '', '', 'DNA', 'AA', 'Both'],
                     ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                     loc='lower left', fontsize=12, edgecolor='black')

    # # Add a line for subspecies and species
    # axs[0].axvline(x=0.45, color='black', linestyle='-', linewidth=1.5)
    # axs[0].text(0.04, 0.3, 'Subspecies', fontsize=12, fontweight='bold', fontfamily='Arial')
    # axs[0].text(0.63, 0.3, 'Species', fontsize=12, fontweight='bold', fontfamily='Arial')
    #
    # axs[1].text(0.01, 0.93, 'Genus', fontsize=12, fontweight='bold', fontfamily='Arial')

    plt.show()
    # # Axis labels for GTDB

    # axs.xaxis.set_label_coords(1.1, -0.15)
    # axs.set_ylabel('')
    # axs.set_xlabel('')
    # axs.set_yticklabels(['', '', '', '', '', ''])
    # for tick in axs.get_xticklabels():
    #     tick.set_fontname('Arial')
    #     # tick.set_fontsize(12)
    # for tick in axs.get_yticklabels():
    #     tick.set_fontname('Arial')
    #     # tick.set_fontsize(12)
    #




if __name__ == '__main__':
    # gtdb_longread()
    figure2()
    # cami_gtdb()
