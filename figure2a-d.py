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


def figure2_AToD():
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'

    # set figure size
    plt.rcParams['figure.figsize'] = [14, 3.5]

    # Read gtdb exclusion and inclusion data
    gtdb_incl = pd.read_csv('gtdb/revision/inclusion/inclusion_short.tsv', sep='\t')
    gtdb_incl = gtdb_incl[gtdb_incl['Rank'] == 'Subspecies']
    gtdb_in_long = pd.read_csv('gtdb/revision/inclusion/inclusion_ont.tsv', sep='\t')
    gtdb_in_long = gtdb_in_long[gtdb_in_long['Rank'] == 'Subspecies']

    gtdb_excl = pd.read_csv('gtdb/revision/species_exclusion/sp-ex_short.tsv', sep='\t')
    gtdb_excl = gtdb_excl[gtdb_excl['Rank'] == 'Genus']
    gtdb_ex_long = pd.read_csv('gtdb/revision/species_exclusion/sp-ex_ont.tsv', sep='\t')
    gtdb_ex_long = gtdb_ex_long[gtdb_ex_long['Rank'] == 'Genus']

    # Load COVID-19 data
    covid_in = pd.read_csv('covid/covid19_in2.tsv', sep='\t')
    covid_ex_p = pd.read_csv('covid/covid19_ex_patient2.tsv', sep='\t')
    covid_ex_c = pd.read_csv('covid/covid19_ex_control2.tsv', sep='\t')

    # Scatter plot parameters
    order = ["Metabuli", "Metabuli-P", 'KrakenUniq', 'Kraken2',
             'Centrifuge', 'Metamaps', 'Metamaps_noEM', 'Kraken2X',
             'Kaiju', 'MMseqs2']
    markers = ['o', 's', 'H', 'D',
               'v', "<", ">", 'P',
               'X', 'd']
    colors = ['#D81B1B', '#E51EC3', '#FFC208', '#FFC208',
              '#FFC208', '#FFC208', '#FFC208', '#38BF66',
              '#38BF66', '#38BF66']
    marker_size = 100

    # fig, axs = plt.subplots(2, 4, figsize=(14, 3.5))
    # DATA
    data = [gtdb_incl, gtdb_in_long, gtdb_excl, gtdb_ex_long]
    titles = ['Prokaryote Inclusion Test', '', 'Prokaryote Exclusion Test', '']
    # Panel labels
    labels = ['', ' ', ' ', ' ']
    x_pos = 0
    y_pos = 1.06

    # Create subplots
    outer = gridspec.GridSpec(1, 4)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],  wspace=.05,
                                           width_ratios=[0.5, 0.83])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],  wspace=.05)

    # 1. GTDB inclusion & exclusion test
    gtdb_in_short = plt.subplot(gs1[0])
    gtdb_in_long = plt.subplot(gs1[1])
    gtdb_ex_short = plt.subplot(gs2[0])
    gtdb_ex_long = plt.subplot(gs2[1])

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

        for tick in axs[j].get_yticklabels() + axs[j].get_xticklabels():
            tick.set_fontname('Arial')

        # Axis range
        if j == 0:
            axs[j].set_xlim(0, 0.5)
            axs[j].set_ylim(0.0, 1.04)
            axs[j].xaxis.set_ticks(np.arange(0, 0.5, 0.2))
            axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
            axs[j].set_xticklabels(['0', '0.2', '0.4'])
            axs[j].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        elif j == 1:
            axs[j].set_xlim(0, 0.83)
            axs[j].set_ylim(0.0, 1.04)
            axs[j].xaxis.set_ticks(np.arange(0, 0.81, 0.2))
            axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
            axs[j].set_yticklabels(['', '', '', '', '', ''])
            axs[j].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8'])
        else:
            axs[j].set_xlim(0, 0.61)
            axs[j].set_ylim(0.0, 1.04)
            axs[j].xaxis.set_ticks(np.arange(0, 0.61, 0.2))
            axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))

        if j == 2:
            axs[j].set_xticklabels(['0', '0.2', '0.4', ''])
            axs[j].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        elif j == 3:
            axs[j].set_yticklabels(['', '', '', '', '', ''])
            axs[j].set_xticklabels(['0', '0.2', '0.4', '0.6'])

        # Add text
        if j == 0:
            axs[j].text(0.5, 0.65, 'Illumina', fontsize=12, fontfamily='Arial',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
            axs[j].text(0.5, 0.57, 'Subspecies', fontsize=10, fontweight='bold', fontfamily='Arial', color='dimgrey',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
        elif j == 1:
            axs[j].text(0.5, 0.65, 'ONT', fontsize=12, fontfamily='Arial',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
            axs[j].text(0.5, 0.57, 'Subspecies', fontsize=10, fontweight='bold', fontfamily='Arial', color='dimgrey',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
        elif j == 2:
            axs[j].text(0.5, 0.95, 'Illumina', fontsize=12, fontfamily='Arial',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
            axs[j].text(0.5, 0.87, 'Genus', fontsize=10, fontweight='bold', fontfamily='Arial', color='dimgrey',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
        elif j == 3:
            axs[j].text(0.5, 0.95, 'ONT', fontsize=12, fontfamily='Arial',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
            axs[j].text(0.5, 0.87, 'Genus', fontsize=10, fontweight='bold', fontfamily='Arial', color='dimgrey',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)

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

        axs[j].set_xlabel('')
        axs[j].set_ylabel('')

        # Set title
        if j == 0:
            axs[j].set_title(titles[j], fontsize=12, fontweight='bold', fontfamily='Arial', x=1.35)
        elif j == 2:
            axs[j].set_title(titles[j], fontsize=12, fontweight='bold', fontfamily='Arial', x=1.05)

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
            # axs[j].legend(bbox_to_anchor=(1.5, 1.05))
            handles, labels2 = axs[j].get_legend_handles_labels()
            print(handles)
            print(labels2)

            r = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)

            labels2.insert(0, "Both")
            labels2.insert(3, "DNA-based")
            labels2.insert(9, "AA-based")

            handles.insert(0, r)
            handles.insert(3, r)
            handles.insert(9, r)

            handles = handles[3:9] + handles[9:13] + handles[0:3]
            labels2 = labels2[3:9] + labels2[9:13] + labels2[0:3]
            print(labels2)
            labels2.insert(10, "")
            handles.insert(10, r)
            labels2.insert(10, "")
            handles.insert(10, r)
            labels2.insert(16, "")
            handles.insert(16, r)
            labels2.insert(16, "")
            handles.insert(16, r)

            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[j].legend(handles, labels2, loc='lower left', markerscale=2, fontsize=11,
                                         handletextpad=0.5, handlelength=0.7, edgecolor='black', ncol=3,
                                         columnspacing=0.2, framealpha=0.3, bbox_to_anchor=(-0.65, 0))
            axs[j].add_artist(first_legend)

    # Axis labels for GTDB
    gtdb_in_short.set_xlabel('Recall (TP / # of reads)', fontsize=14, fontweight='bold', fontfamily='Arial')
    gtdb_in_short.xaxis.set_label_coords(2.3, -0.15)
    gtdb_in_short.set_ylabel('Precision (TP / TP+FP)', fontsize=14, fontweight='bold', fontfamily='Arial')
    # gtdb_in_short.yaxis.set_label_coords(-0.15, 0)

    # gtdb_in_long.set_xlabel('')
    # gtdb_ex_long.set_ylabel('')
    # axs2.set_ylabel('')
    # axs2.set_xlabel('')
    # axs2.set_yticklabels(['', '', '', '', '', ''])
    # for tick in gtdb_in_short.get_xticklabels() + gtdb_in_long.get_xticklabels:
    #     tick.set_fontname('Arial')
    #     # tick.set_fontsize(12)
    # for tick in axs1.get_yticklabels() + axs2.get_yticklabels():
    #     tick.set_fontname('Arial')
        # tick.set_fontsize(12)

    # Legend for DNA, AA, and both
    pa1 = Patch(facecolor=colors[3], edgecolor=colors[3])
    pb1 = Patch(facecolor=colors[5], edgecolor=colors[5])
    both1 = Patch(facecolor='#D81B1B', edgecolor='#D81B1B')
    both2 = Patch(facecolor='#E51EC3', edgecolor='#E51EC3')

    # axs[2].legend(handles=[pa1, pb1, both1, pa1, pb1, both2],
    #               labels=['', '', '', 'DNA', 'AA', 'Both'],
    #               ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
    #               loc='lower left', fontsize=10, edgecolor='black', framealpha=0.3)

    # Add a line for subspecies and species
    # axs[0].

    # axs[1].text(0.01, 0.93, 'Genus', fontsize=12, fontweight='bold', fontfamily='Arial')
    plt.subplots_adjust(hspace=0)


    # ----------------------------------------------- COVID-19 --------------------------------------------------------#
    # Use rows with Kraken2, Kaiju, and Metabuli
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
    covid_in_b_axs.set_xticklabels(covid_order)
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
    plt.savefig('./plots/gtdb_covid.png', dpi=500, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    # gtdb_longread()
    figure2_AToD()
    # cami_gtdb()
