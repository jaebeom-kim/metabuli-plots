import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def hiv_and_gtdb():
    # Read hiv exclusion and inclusion data
    hiv_incl = pd.read_csv('hiv-inclusion.tsv', sep='\t')
    hiv_excl = pd.read_csv('hiv-exclusion.tsv', sep='\t')

    # Read gtdb exclusion and inclusion data
    gtdb_incl = pd.read_csv('hiv-inclusion.tsv', sep='\t')
    gtdb_excl = pd.read_csv('hiv-exclusion.tsv', sep='\t')

    # Plot hiv and gtdb exclusion
    # Set figure size

    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(10, 10))
    marker_size = 100

    # Scatter Plot hiv inclusion

    axs[0, 0] = sns.scatterplot(x='Sensitivity', y='Precision',
                                hue='Tool',  # different colors by group
                                style='Tool',  # different shapes by group
                                s=marker_size,  # marker size
                                data=hiv_incl, ax=axs[0, 0])
    # Set x and y limits
    axs[0, 0].set_xlim(0, 1.1)
    axs[0, 0].set_ylim(0, 1.1)

    axs[0, 1] = sns.scatterplot(x='Sensitivity', y='Precision',
                                hue='Tool',  # different colors by group
                                style='Tool',  # different shapes by group
                                s=marker_size,  # marker size
                                data=hiv_excl, ax=axs[0, 1])

    plt.show()










if __name__ == '__main__':
    hiv_and_gtdb()