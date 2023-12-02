import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def gradeByCladeSize() -> None:
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5, rc={'lines.linewidth': 2.5})
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1
    plt.rcParams['grid.linestyle'] = '--'

    # Load Data
    data = pd.read_csv('./gtdb/revision/inclusion/inclusion_short_clade_size.tsv', sep='\t')

    # Set figure size
    fig, axs = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(7, 7))


    # color: tool
    # marker: tool
    # marker_size = 120
    # order = ['Kraken2', 'KrakenUniq', 'Centrifuge', 'Kaiju', 'Kraken2X', 'MMseqs2', 'Metabuli']
    # colors = ['#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66', '#D81B1B']
    #
    # markers = ['o', 's', 'H', 'D',
    #            'v', ">", 'P',
    #            'X', 'd']

    order = ['KrakenUniq', 'Kraken2', 'Centrifuge', 'Kraken2X', 'Kaiju', 'MMseqs2', 'Metabuli']
    markers = ['H', 'D', 'v', "P", 'X', 'd', 'o']
    colors = ['#FFC208', '#FFC208', '#FFC208', '#38BF66', '#38BF66', '#38BF66', '#D81B1B']
    marker_size = 15

    # line plot of tools
    # x-axis: clade size (Group)
    # y-axis: grade (F1)


    axs = sns.lineplot(x='Group',
                       y='F1', hue='Tools', style='Tools', hue_order=order, style_order=order, palette=colors, markers=markers, markersize=marker_size, data=data, ax=axs)
    axs.xaxis.set_ticks([1, 2, 3, 4, 5])
    axs.yaxis.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    axs.set_xlabel('Species Clade Size')
    axs.set_ylabel('F1 Score')
    # axs[j].set_xticklabels(['0', '0.2', '0.4'])
    axs.set_xticklabels(['1 - 2', '3 - 4', '5 - 8', '9 - 16', '17 -'])

    axs.legend(markerscale=2)
    plt.show()





if __name__ == '__main__':
    gradeByCladeSize()