import os
import os.path
import matplotlib.pyplot as plt
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
    gtdb_incl = pd.read_csv('gtdb_inclusion.tsv', sep='\t')
    gtdb_excl = pd.read_csv('gtdb_exclusion.tsv', sep='\t')
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
    titles = ['GTDB Inclusion Test', 'GTDB Exclusion Test']
    # Panel labels
    labels = ['a', 'b', 'c', 'd']


    x_pos = 0
    y_pos = 1.06

    axs1 = plt.subplot(1, 4, 1)
    axs2 = plt.subplot(1, 4, 2)

    covid_in_t_axs = plt.subplot(2, 4, 3)
    covid_in_t_axs.spines[['right', 'top']].set_visible(False)
    covid_in_b_axs = plt.subplot(2, 4, 7)
    covid_in_b_axs.spines[['right', 'bottom']].set_visible(False)
    covid_ex_p_axs = plt.subplot(2, 4, 4)#, sharex=covid_in_t_axs)
    covid_ex_p_axs.spines[['right', 'top']].set_visible(False)
    covid_ex_c_axs = plt.subplot(2, 4, 8)
    covid_ex_c_axs.spines[['right', 'bottom']].set_visible(False)

    axs = [axs1, axs2]
    for j in range(2):
        axs[j].set_xlim(0, 1.04)
        axs[j].set_ylim(0.0, 1.04)
        axs[j].xaxis.set_ticks(np.arange(0, 1.02, 0.1))
        axs[j].yaxis.set_ticks(np.arange(0, 1.02, 0.1))
        axs[j].set_xticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'])
        axs[j].set_yticklabels(['0.0', '', '0.2', '', '0.4', '', '0.6', '', '0.8', '', '1.0'])

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
            axs[j].legend(loc='lower right', markerscale=2, edgecolor='black', fontsize=14)
            handles, labels2 = axs[j].get_legend_handles_labels()
            for h in handles:
                h.set_edgecolor('black')
            first_legend = axs[j].legend(handles, labels2, loc='lower right', markerscale=2, fontsize=12,
                                         handletextpad=0.5, handlelength=0.7, edgecolor='black')
            axs[j].add_artist(first_legend)

    # Axis labels for GTDB
    axs1.xaxis.set_label_coords(1.1, -0.12)
    axs2.set_ylabel('')
    axs2.set_xlabel('')
    for tick in axs1.get_xticklabels():
        tick.set_fontname('Arial')
    for tick in axs1.get_yticklabels():
        tick.set_fontname('Arial')

    # Legend for DNA, AA, and both
    pa1 = Patch(facecolor=colors[3], edgecolor=colors[3])
    pb1 = Patch(facecolor=colors[5], edgecolor=colors[5])
    both1 = Patch(facecolor='#D81B1B', edgecolor='#D81B1B')
    both2 = Patch(facecolor='#E51EC3', edgecolor='#E51EC3')

    axs[1].legend(handles=[pa1, pb1, both1, pa1, pb1, both2],
                  labels=['', '', '', 'DNA', 'AA', 'Both'],
                  ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                  loc='lower left', fontsize=12, edgecolor='black')

    # Add a line for subspecies and species
    axs[0].axvline(x=0.45, color='black', linestyle='-', linewidth=1.5)
    axs[0].text(0.04, 0.3, 'Subspecies', fontsize=12, fontweight='bold', fontfamily='Arial')
    axs[0].text(0.63, 0.3, 'Species', fontsize=12, fontweight='bold', fontfamily='Arial')




    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9)
    plt.subplots_adjust(hspace=0)

    # COVID-19 data
    # Use rows with Kraken2, Kaiju, and Metabuli
    covid_in = covid_in[covid_in['Tool'].isin(['Kraken2', 'Metabuli', 'Kaiju'])]
    covid_order = ['Kraken2', 'Metabuli', 'Kaiju']
    covid_in['Sensitivity'] = covid_in['Correct'] / covid_in['Reads']
    covid_in['Predicted'] = covid_in['Correct'] + covid_in['Wrong']
    covid_in['Precision'] = covid_in['Correct'] / covid_in['Predicted']
    covid_in['1-Precision'] = 1 - covid_in['Precision']

    covid_in['FP/TP'] = covid_in['Wrong'] / covid_in['Correct']
    covid_in['FP'] = covid_in['Wrong']
    covid_in_omicron = covid_in[covid_in['Variant'] == 'Omicron']
    covid_in_beta = covid_in[covid_in['Variant'] == 'Beta']

    marker_size = 10
    colors = ['#FFC208', '#38BF66', '#D81B1B']
    correct_colors = ['orangered', 'darkgreen']
    wrong_colors = ['sandybrown', 'darkseagreen', 'salmon']
    covid_in_t_axs.set_ylim(0, 0.15)
    covid_in_b_axs.set_ylim(0.15, 0)

    # COVID-19 inclusion test TOP
    covid_in_t_axs = sns.stripplot(x='Tool', y='Sensitivity', hue='Tool', palette=colors, edgecolor='black',
                                   data=covid_in_omicron, order=covid_order, ax=covid_in_t_axs, linewidth=0.8,
                                   s=marker_size, jitter=0.2)
    covid_in_t_axs = sns.stripplot(x='Tool', y='Sensitivity', hue='Tool', marker='X', palette=colors, linewidth=0.8,
                                   data=covid_in_beta, order=covid_order, ax=covid_in_t_axs, edgecolor='black',
                                   s=marker_size, jitter=0.2)
    covid_in_t_axs.set_title('SARS-CoV-2 Inclusion Test', fontsize=12, fontweight='bold', fontfamily='Arial')
    covid_in_t_axs.set_xlabel('')
    covid_in_t_axs.set_ylabel('')
    covid_in_t_axs.legend_.remove()
    covid_in_t_axs.set_yticklabels(['', '0.05', '0.10', '0.15'])
    # Add 'Sensitivity' text in the top right corner
    covid_in_t_axs.text(0.98, 0.95, 'Sensitivity', fontsize=14, fontfamily='Arial',
                        horizontalalignment='right', verticalalignment='top', transform=covid_in_t_axs.transAxes)


    # COVID-19 inclusion test BOTTOM
    covid_in_b_axs.sharex(covid_in_t_axs)
    covid_in_b_axs = sns.stripplot(x='Tool', y='1-Precision', hue='Tool', alpha=1,
                                   data=covid_in_omicron, ax=covid_in_b_axs, order=covid_order, palette=wrong_colors,
                                   s=marker_size, edgecolor='black', linewidth=0.8, jitter=0.2)
    covid_in_b_axs = sns.stripplot(x='Tool', y='1-Precision', hue='Tool', marker='X', alpha=1,
                                   data=covid_in_beta, ax=covid_in_b_axs, order=covid_order, palette=wrong_colors,
                                   s=marker_size, edgecolor='black', linewidth=0.8, jitter=0.2)
    covid_in_b_axs.set_xlabel('')
    covid_in_b_axs.set_ylabel('')
    covid_in_b_axs.text(0.98, 0.05, '1-Precision', fontsize=14, fontfamily='Arial',
                        horizontalalignment='right', verticalalignment='bottom', transform=covid_in_b_axs.transAxes)
    covid_in_b_axs.legend_.remove()

    # Add 'Incorrect variant / Correct variant' label in the top right corner
    # covid_in_b_axs.text(0.99, 0.1, r'$\frac{Incorrect\;variant}{Correct\;variant}$', transform=covid_in_b_axs.transAxes,
    #                     fontsize=18, fontfamily='Arial', verticalalignment='bottom', horizontalalignment='right')


    # COVID-19 exclusion test
    covid_ex_p = covid_ex_p[covid_ex_p['Tool'].isin(['Kraken2', 'Metabuli', 'Kaiju'])]
    covid_ex_c = covid_ex_c[covid_ex_c['Tool'].isin(['Kraken2', 'Metabuli', 'Kaiju'])]
    covid_ex_c['logValue'] = np.log2(covid_ex_c['Value'] + 1)
    covid_ex_p['Sentitivity'] = covid_ex_p['Value']
    covid_ex_c['FP'] = covid_ex_c['Value']
    covid_ex_p_axs.set_ylim(0, 0.6)
    covid_ex_c_axs.set_ylim(3, 0)

    # COVID-19 exclusion test TOP
    covid_ex_p_axs = sns.stripplot(x='Tool', y='Sentitivity', hue='Tool', palette=colors, edgecolor='black',
                                      data=covid_ex_p, order=covid_order, ax=covid_ex_p_axs, linewidth=0.8,
                                      s=marker_size)
    covid_ex_p_axs.set_title('SARS-CoV-2 Exclusion Test', fontsize=12, fontweight='bold', fontfamily='Arial')
    covid_ex_p_axs.set_xlabel('')
    covid_ex_p_axs.set_ylabel('')
    covid_ex_p_axs.legend_.remove()
    covid_ex_p_axs.text(0.02, 0.95, 'Sensitivity', fontsize=14, fontfamily='Arial',
                        horizontalalignment='left', verticalalignment='top', transform=covid_ex_p_axs.transAxes)

    covid_ex_p_axs.set_yticklabels(['', '0.2', '0.4', '0.6'])

    # COVID-19 exclusion test BOTTOM
    covid_ex_c_axs.sharex(covid_ex_p_axs)
    covid_ex_c_axs = sns.stripplot(x='Tool', y='FP', hue='Tool', alpha=1,
                                   data=covid_ex_c, ax=covid_ex_c_axs, order=covid_order, palette=wrong_colors,
                                   s=marker_size, edgecolor='black', linewidth=0.8)
    covid_ex_c_axs.set_xlabel('')
    covid_ex_c_axs.set_ylabel('')
    covid_ex_c_axs.text(0.02, 0.05, 'FP', fontsize=14, fontfamily='Arial',
                        horizontalalignment='left', verticalalignment='bottom', transform=covid_ex_c_axs.transAxes)
    covid_ex_c_axs.legend_.remove()



    plt.show()


if __name__ == '__main__':
    figure2()
