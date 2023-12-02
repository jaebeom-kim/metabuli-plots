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
    # gtdb_incl = pd.read_csv('gtdb/revision/inclusion/inclusion_short.tsv', sep='\t')
    gtdb_incl = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_short.tsv', sep='\t')
    gtdb_incl = gtdb_incl[gtdb_incl['Rank'] == 'Species']
    # gtdb_in_long = pd.read_csv('gtdb/revision/inclusion/inclusion_sequel_ccs.tsv', sep='\t')
    gtdb_in_long = pd.read_csv('gtdb/revision/subspecies_exclusion/ss-ex_sequel_ccs.tsv', sep='\t')
    gtdb_in_long = gtdb_in_long[gtdb_in_long['Rank'] == 'Species']
    gtdb_in_long = gtdb_in_long[gtdb_in_long['Tool'] != 'Metamaps_EM']

    gtdb_excl = pd.read_csv('gtdb/revision/species_exclusion/sp-ex_short.tsv', sep='\t')
    gtdb_excl = gtdb_excl[gtdb_excl['Rank'] == 'Genus']
    gtdb_ex_long = pd.read_csv('gtdb/revision/species_exclusion/sp-ex_sequel_ccs.tsv', sep='\t')
    gtdb_ex_long = gtdb_ex_long[gtdb_ex_long['Rank'] == 'Genus']
    gtdb_ex_long = gtdb_ex_long[gtdb_ex_long['Tool'] != 'Metamaps_EM']


    # Load COVID-19 data
    covid_in = pd.read_csv('covid/sars-cov-2_inclusion.tsv', sep='\t')
    covid_ex_p = pd.read_csv('covid/sars-cov-2_exclusion.tsv', sep='\t')
    covid_ex_c = pd.read_csv('covid/sars-cov-2_exclusion_NC.tsv', sep='\t')

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

    # fig, axs = plt.subplots(2, 4, figsize=(14, 3.5))
    # DATA
    data = [gtdb_incl, gtdb_in_long, gtdb_excl, gtdb_ex_long]
    titles = ['Subspecies Exclusion Test', '', 'Species Exclusion Test', '']
    # Panel labels
    labels = ['', ' ', ' ', ' ']
    x_pos = 0
    y_pos = 1.06

    # Create subplots
    outer = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.1, width_ratios=[0.9, 0.9, 1, 0.95])
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],  wspace=.1,
                                           width_ratios=[0.5, 0.5])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],  wspace=.1,
                                           width_ratios=[0.5, 0.5])

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
    covid_ex_p_axs = plt.subplot(1, 4, 4)  # , sharex=covid_in_t_axs)
    covid_ex_p_axs.spines[['right', 'top']].set_visible(False)
    # covid_ex_c_axs = plt.subplot(2, 4, 8)
    # covid_ex_c_axs.spines[['right']].set_visible(False)

    axs = [gtdb_in_short, gtdb_in_long, gtdb_ex_short, gtdb_ex_long]
    for j in range(4):

        for tick in axs[j].get_yticklabels() + axs[j].get_xticklabels():
            tick.set_fontname('Arial')

        axs[j].tick_params(axis='both', which='major', pad=0)

        # Axis range
        if j == 0:
            # axs[j].set_xlim(0, 0.5)
            axs[j].set_xlim(0.4, 1.00)
            axs[j].set_ylim(0.0, 1.04)
            # axs[j].xaxis.set_ticks(np.arange(0, 0.5, 0.2))
            axs[j].xaxis.set_ticks(np.arange(0.4, 1.02, 0.2))
            axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
            # axs[j].set_xticklabels(['0', '0.2', '0.4'])
            axs[j].set_xticklabels(['0.4', '0.6', '0.8', ''])
            axs[j].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        elif j == 1:
            axs[j].set_xlim(0.4, 1.00)
            axs[j].set_ylim(0.0, 1.04)
            axs[j].xaxis.set_ticks(np.arange(0.4, 1.02, 0.2))
            axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
            axs[j].set_yticklabels(['', '', '', '', '', ''])
            axs[j].set_xticklabels(['0.4', '0.6', '0.8', '1.0'])
        elif j == 2:
            axs[j].set_xlim(0, 0.6)
            axs[j].set_ylim(0.0, 1.04)
            axs[j].xaxis.set_ticks(np.arange(0, 0.61, 0.2))
            axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
        elif j == 3:
            axs[j].set_xlim(0.4, 1.0)
            axs[j].set_ylim(0.0, 1.04)
            axs[j].xaxis.set_ticks(np.arange(0.4, 1.02, 0.2))
            axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))

        if j == 2:
            axs[j].set_xticklabels(['0', '0.2', '0.4', ''])
            axs[j].set_yticklabels(['', '', '', '', '', ''])
            # axs[j].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        elif j == 3:
            axs[j].set_yticklabels(['', '', '', '', '', ''])
            axs[j].set_xticklabels(['0.4', '0.6', '0.8', '1.0'])

        # Add text
        if j == 0:
            axs[j].text(0.5, 0.65, 'Illumina', fontsize=12, fontfamily='Arial',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
            axs[j].text(0.5, 0.57, 'Species', fontsize=10, fontweight='bold', fontfamily='Arial', color='dimgrey',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
        elif j == 1:
            axs[j].text(0.5, 0.65, 'PacBio HiFi', fontsize=12, fontfamily='Arial',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
            axs[j].text(0.5, 0.57, 'Species', fontsize=10, fontweight='bold', fontfamily='Arial', color='dimgrey',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
        elif j == 2:
            axs[j].text(0.5, 0.4, 'Illumina', fontsize=12, fontfamily='Arial',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
            axs[j].text(0.5, 0.32, 'Genus', fontsize=10, fontweight='bold', fontfamily='Arial', color='dimgrey',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
        elif j == 3:
            axs[j].text(0.5, 0.4, 'PacBio HiFi', fontsize=12, fontfamily='Arial',
                        horizontalalignment='center', verticalalignment='top', transform=axs[j].transAxes)
            axs[j].text(0.5, 0.32, 'Genus', fontsize=10, fontweight='bold', fontfamily='Arial', color='dimgrey',
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
            axs[j].set_title(titles[j], fontsize=12, fontweight='bold', fontfamily='Arial', x=1.05)
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
            labels2.insert(8, "AA-based")

            handles.insert(0, r)
            handles.insert(3, r)
            handles.insert(8, r)

            handles = handles[3:8] + handles[8:12] + handles[0:3] + handles[12:]
            labels2 = labels2[3:8] + labels2[8:12] + labels2[0:3] + labels2[12:]
            print(labels2)

            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[j].legend(handles, labels2, loc='lower left', markerscale=1.8, fontsize=11,
                                         handletextpad=0.5, handlelength=0.7, edgecolor='black', ncol=3,
                                         columnspacing=0.2, framealpha=0.3, bbox_to_anchor=(-1.1, 0))
            axs[j].add_artist(first_legend)

    # Axis labels for GTDB
    gtdb_in_short.set_xlabel('Recall (TP / # of reads)', fontsize=14, fontfamily='Arial') #fontweight='bold', )
    gtdb_in_short.xaxis.set_label_coords(2.3, -0.15)
    gtdb_in_short.set_ylabel('Precision (TP / TP+FP)', fontsize=14, fontfamily='Arial') #fontweight='bold',
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
    covid_order = ['Centrifuge','Kraken2', 'KrakenUniq',  'Hybrid',
                   'Metabuli', 'Metabuli-P', 'Kraken2X', 'Kaiju', 'MMseqs2']
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
    marker_size_2 = 6
    alpha = 0.7
    jitter = 0.2
    colors = ['#FFC208', '#FFC208', '#FFC208', 'dimgray', '#D81B1B', '#D81B1B', '#38BF66', '#38BF66', '#38BF66']
    correct_colors = ['orangered', 'darkgreen']
    wrong_colors = ['sandybrown', 'sandybrown', 'sandybrown', 'silver', 'salmon', 'salmon',
                    'darkseagreen', 'darkseagreen', 'darkseagreen']
    covid_in_t_axs.set_ylim(0, 0.15)
    covid_in_b_axs.set_ylim(0.25, 0)

    # ------------ COVID-19 inclusion test TOP ------------------#
    covid_in_t_axs = sns.stripplot(x='Tool', y='Recall', hue='Tool', palette=colors, edgecolor='black',
                                   data=covid_in_omicron, order=covid_order, ax=covid_in_t_axs, linewidth=0.8,
                                   s=marker_size, jitter=jitter, hue_order=covid_order, alpha=alpha)
    covid_in_t_axs = sns.stripplot(x='Tool', y='Recall', hue='Tool', marker='X', palette=colors, linewidth=0.8,
                                   data=covid_in_beta, order=covid_order, ax=covid_in_t_axs, edgecolor='black',
                                   s=marker_size, jitter=jitter, hue_order=covid_order, alpha=alpha)
    covid_in_t_axs.set_title('SARS-CoV-2 Inclusion Test', fontsize=12, fontweight='bold', fontfamily='Arial')
    covid_in_t_axs.set_xlabel('')
    covid_in_t_axs.set_xticklabels(['', '', '', '', '', '', ''])
    covid_in_t_axs.set_ylabel('Recall')
    covid_in_t_axs.legend_.remove()
    covid_in_t_axs.set_yticklabels(['', '', '0.1', ''])
    covid_in_t_axs.tick_params(axis='both', which='major', pad=-3)

    # set font size for tick labels
    # for tick in covid_in_t_axs.get_yticklabels():
    #     tick.set_fontsize(10)

    # Add 'Sensitivity' text in the top right corner
    # covid_in_t_axs.text(1, 0.95, 'Recall', fontsize=14, fontfamily='Arial',
    #                     horizontalalignment='right', verticalalignment='top', transform=covid_in_t_axs.transAxes)
    # covid_in_t_axs.text(-0.5, 0.16, 'c', fontsize=12, fontweight='bold', fontfamily='Arial')

    # ------------ COVID-19 inclusion test BOTTOM ------------------#
    covid_in_b_axs = sns.stripplot(x='Tool', y='1-Precision', hue='Tool', alpha=alpha,
                                   data=covid_in_omicron, ax=covid_in_b_axs, order=covid_order, palette=wrong_colors,
                                   s=marker_size, edgecolor='black', linewidth=0.8, jitter=jitter, hue_order=covid_order)
    covid_in_b_axs = sns.stripplot(x='Tool', y='1-Precision', hue='Tool', marker='X', alpha=alpha,
                                   data=covid_in_beta, ax=covid_in_b_axs, order=covid_order, palette=wrong_colors,
                                   s=marker_size, edgecolor='black', linewidth=0.8, jitter=jitter, hue_order=covid_order)
    covid_in_b_axs.set_xlabel('')
    covid_in_b_axs.set_ylabel('1-Precision', fontfamily='Arial')
    covid_in_b_axs.legend_.remove()
    covid_in_b_axs.set_xticklabels(covid_order)
    covid_in_b_axs.tick_params(axis='both', which='major', pad=-3)

    # covid_in_b_axs.text(1, 0.0, '1-Precision', fontsize=14, fontfamily='Arial',
    #                     horizontalalignment='right', verticalalignment='bottom', transform=covid_in_b_axs.transAxes)
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
    covid_ex_p_axs.set_ylim(-0.1, 0.4)
    # covid_ex_c_axs.set_ylim(8, 0)

    # COVID-19 exclusion test TOP
    covid_ex_p_axs = sns.stripplot(x='Tool', y='Sentitivity', hue='Tool', palette=colors, edgecolor='black',
                                   data=covid_ex_p, order=covid_order, ax=covid_ex_p_axs, linewidth=0.8,
                                   s=6, hue_order=covid_order, jitter=0.2, alpha=0.7)
    covid_ex_p_axs.set_title('SARS-CoV-2 Exclusion Test', fontsize=12, fontweight='bold', fontfamily='Arial')

    # covid_ex_p_axs.text(0.5, 0.97, 'Patient samples', transform=covid_ex_p_axs.transAxes, fontsize=13, fontfamily='Arial',
    #                     verticalalignment='top', horizontalalignment='center', color='dimgray') #weight='bold', )
    covid_ex_p_axs.set_xlabel('')
    plt.setp(covid_ex_p_axs.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontfamily='Arial',
             fontsize=11)
    # covid_ex_p_axs.set_xticklabels(['', '', '', '', '', '', ''])
    covid_ex_p_axs.set_ylabel('')
    covid_ex_p_axs.yaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4])
    covid_ex_p_axs.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4'])
    covid_ex_p_axs.legend_.remove()
    covid_ex_p_axs.tick_params(axis='both', which='major', pad=-3)
    covid_ex_p_axs.text(0.01, 0.98, 'Recall', fontsize=14, fontfamily='Arial',
                        horizontalalignment='left', verticalalignment='top', transform=covid_ex_p_axs.transAxes)

    # axs[j].xaxis.set_ticks(np.arange(0.4, 1.02, 0.2))
    # axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.2))
    # axs[j].set_xticklabels(['0', '0.2', '0.4'])
    # axs[j].set_xticklabels(['0.4', '0.6', '0.8', ''])
    # axs[j].set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    # covid_ex_p_axs.text(-0.5, 0.8, 'd', fontsize=12, fontweight='bold', fontfamily='Arial')

    # covid_ex_p_axs.set_yticklabels(['', '0.25', '0.5', '0.75'])

    # COVID-19 exclusion test BOTTOM
    # covid_ex_c_axs = sns.stripplot(x='Tool', y='log(FP)', hue='Tool',
    #                                data=covid_ex_c, ax=covid_ex_c_axs, order=covid_order, palette=wrong_colors,
    #                                s=marker_size_2, edgecolor='black', linewidth=0.8, hue_order=covid_order,
    #                                jitter=jitter, alpha=alpha)
    #
    # covid_ex_c_axs.set_xlabel('')
    # covid_ex_c_axs.set_ylabel(r'$log_2(FP)$')
    # covid_ex_c_axs.tick_params(axis='both', which='major', pad=-3)
    # # covid_ex_c_axs.text(1, 0.0, r'$log_2(FP)$', fontsize=14, fontfamily='Arial', fontstyle='normal',
    # #                     horizontalalignment='right', verticalalignment='bottom', transform=covid_ex_c_axs.transAxes)
    # covid_ex_c_axs.legend_.remove()
    # covid_ex_c_axs.text(0.5, 0.5, 'Control samples', transform=covid_ex_c_axs.transAxes, fontsize=13, fontfamily='Arial',
    #                     verticalalignment='center', horizontalalignment='center', color='dimgray') # weight='bold')
    #
    # plt.setp(covid_ex_c_axs.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontfamily='Arial',
    #          fontsize=11)

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9)
    plt.savefig('./revision/figure2a-d.png', dpi=500, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    # gtdb_longread()
    figure2_AToD()
    # cami_gtdb()
