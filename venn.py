import os
import os.path
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes


def venn_hatch():
    # # Load data
    # metabuli_file_name = './venn/metabuli_' + str(0) + '.txt'
    # kraken2_file_name = './venn/kraken2_' + str(0) + '.txt'
    # kaiju_file_name = './venn/kaiju_' + str(0) + '.txt'
    #
    # metabuli = pd.read_csv(metabuli_file_name, sep='\t', header=None)[0].tolist()
    # kraken2 = pd.read_csv(kraken2_file_name, sep='\t', header=None)[0].tolist()
    # kaiju = pd.read_csv(kaiju_file_name, sep='\t', header=None)[0].tolist()
    # #
    # kaiju_set = set(kaiju)
    # kraken2_set = set(kraken2)
    # metabuli_set = set(metabuli)
    #

    # kaiju_kr2_union = kaiju_set.union(kraken2_set)
    # a = y * len(metabuli_set - kaiju_kr2_union) / len(metabuli_set.union(kaiju_kr2_union))
    # b = y - a
    # c = x * len(kaiju_set - kraken2_set) / len(kaiju_kr2_union)
    # d = x * len(kraken2_set - kaiju_set) / len(kaiju_kr2_union)
    # e = x - c - d
    # f = b * (1 - len(kaiju_set - kraken2_set - metabuli_set) / len(kaiju_set - kraken2_set))
    # h = b * (1 - len(kraken2_set - kaiju_set - metabuli_set) / len(kraken2_set - kaiju_set))
    # g = b * (1 - len(kraken2_set.intersection(kaiju_set) - metabuli_set) / len(kaiju_set.intersection(kraken2_set)))
    # print(a, b, c, d, e, f, g, h)

    x = 4  # x = 4
    y = 4
    a = 0.16346091498488724
    b = 3.8365390850151126
    c = 0.361830486455227
    d = 0.4530811342442936
    e = 3.1850883793004794
    f = 3.037901178095663
    g = 3.8232188293674514
    h = 3.656024225187836

    X = np.array([0, x])
    Y = np.array([0, y])
    plt.figure(figsize=(4, 4))
    plt.plot(X, Y, color='None')

    plt.rcParams['hatch.linewidth'] = 1
    plt.rcParams['hatch.color'] = 'black'
    hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

    al = 0.8
    metabuli_only = patches.Rectangle((0, x - a), x, a, edgecolor='black', facecolor='red', alpha=al,
                                      fill=False, hatch='o', color='red')
    plt.gca().add_patch(metabuli_only)

    kaiju_kr2 = patches.Rectangle((0, 0), c, b, edgecolor='black', facecolor='green', alpha=al,
                                  fill=False, hatch='o', color='green')
    plt.gca().add_patch(kaiju_kr2)

    kaiju_kr2_inter = patches.Rectangle((c, 0), e, b, edgecolor='black', facecolor='gray', alpha=al,
                                        fill=False, hatch='o', color='gray')
    plt.gca().add_patch(kaiju_kr2_inter)

    kr2_kaiju = patches.Rectangle((c + e, 0), d, b, edgecolor='black', facecolor='gold', alpha=al,
                                  fill=False, hatch='o', color='gold')
    plt.gca().add_patch(kr2_kaiju)
    # lightsalmon
    kaiju_coverd = patches.Rectangle((0, b - f), c, f, edgecolor='black', facecolor='red', alpha=al,
                                     fill=False, hatch='o', color='red')  # , hatch='///')
    plt.gca().add_patch(kaiju_coverd)
    plt.text(c / 2, b - f + 0.05, str(round(f / b * 100, 1)) + '%', transform=plt.gca().transData,
             color='black', weight='bold', fontsize=13,
             fontfamily='Arial', verticalalignment='bottom',
             horizontalalignment='center', rotation=90)

    kr2_coverd = patches.Rectangle((c + e, b - h), d, h, edgecolor='black', facecolor='red', alpha=al,
                                   fill=False, hatch='o', color='red')  # ,hatch='///')
    plt.gca().add_patch(kr2_coverd)
    plt.text(c + e + d / 2, b - h + 0.05, str(round(h / b * 100, 1)) + '%', transform=plt.gca().transData,
             color='black', weight='bold', fontsize=13,
             fontfamily='Arial', verticalalignment='bottom',
             horizontalalignment='center', rotation=90)

    kaiju_kr2_coverd = patches.Rectangle((c, b - g), e, g, edgecolor='black', facecolor='red',
                                         alpha=al, fill=False, hatch='o', color='red')  # , hatch='///')
    plt.gca().add_patch(kaiju_kr2_coverd)
    plt.text(c + e / 2, b - g + 0.05, str(round(g / b * 100, 1)) + '%', transform=plt.gca().transData,
             color='black', weight='bold', fontsize=13,
             fontfamily='Arial', verticalalignment='bottom',
             horizontalalignment='center', rotation=0)
    plt.show()


def venn():

    # # Load data
    # metabuli_file_name = './venn/metabuli_' + str(0) + '.txt'
    # kraken2_file_name = './venn/kraken2_' + str(0) + '.txt'
    # kaiju_file_name = './venn/kaiju_' + str(0) + '.txt'
    #
    # metabuli = pd.read_csv(metabuli_file_name, sep='\t', header=None)[0].tolist()
    # kraken2 = pd.read_csv(kraken2_file_name, sep='\t', header=None)[0].tolist()
    # kaiju = pd.read_csv(kaiju_file_name, sep='\t', header=None)[0].tolist()
    # #
    # kaiju_set = set(kaiju)
    # kraken2_set = set(kraken2)
    # metabuli_set = set(metabuli)
    #

    # kaiju_kr2_union = kaiju_set.union(kraken2_set)
    # a = y * len(metabuli_set - kaiju_kr2_union) / len(metabuli_set.union(kaiju_kr2_union))
    # b = y - a
    # c = x * len(kaiju_set - kraken2_set) / len(kaiju_kr2_union)
    # d = x * len(kraken2_set - kaiju_set) / len(kaiju_kr2_union)
    # e = x - c - d
    # f = b * (1 - len(kaiju_set - kraken2_set - metabuli_set) / len(kaiju_set - kraken2_set))
    # h = b * (1 - len(kraken2_set - kaiju_set - metabuli_set) / len(kraken2_set - kaiju_set))
    # g = b * (1 - len(kraken2_set.intersection(kaiju_set) - metabuli_set) / len(kaiju_set.intersection(kraken2_set)))
    # print(a, b, c, d, e, f, g, h)

    x = 4 # x = 4
    y = 4
    a = 0.16346091498488724
    b = 3.8365390850151126
    c = 0.361830486455227
    d = 0.4530811342442936
    e = 3.1850883793004794
    f = 3.037901178095663
    g = 3.8232188293674514
    h = 3.656024225187836

    X = np.array([0, x])
    Y = np.array([0, y])
    plt.figure(figsize=(4, 4))
    plt.plot(X, Y, color='None')

    hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

    al = 0.5
    metabuli_only = patches.Rectangle((0, x - a), x, a, color='red', edgecolor='black', facecolor='red', alpha=al, fill=False, hatch='///')
    plt.gca().add_patch(metabuli_only)

    kaiju_kr2 = patches.Rectangle((0, 0), c, b, edgecolor='black', facecolor='green', alpha=al)
    plt.gca().add_patch(kaiju_kr2)

    kaiju_kr2_inter = patches.Rectangle((c, 0), e, b, edgecolor='black', facecolor='gray', alpha=al)
    plt.gca().add_patch(kaiju_kr2_inter)

    kr2_kaiju = patches.Rectangle((c + e, 0), d, b, edgecolor='black', facecolor='gold', alpha=al)
    plt.gca().add_patch(kr2_kaiju)
#lightsalmon
    kaiju_coverd = patches.Rectangle((0, b - f), c, f, edgecolor='black', facecolor='red', alpha=al) #, hatch='///')
    plt.gca().add_patch(kaiju_coverd)
    plt.text(c / 2, b - f + 0.05, str(round(f / b * 100, 1)) + '%', transform=plt.gca().transData,
             color='black', weight='bold', fontsize=13,
             fontfamily='Arial', verticalalignment='bottom',
             horizontalalignment='center', rotation=90)

    kr2_coverd = patches.Rectangle((c + e, b - h), d, h, edgecolor='black', facecolor='red', alpha=al) #,hatch='///')
    plt.gca().add_patch(kr2_coverd)
    plt.text(c + e + d / 2, b - h + 0.05, str(round(h / b * 100, 1)) + '%', transform=plt.gca().transData,
             color='black', weight='bold', fontsize=13,
             fontfamily='Arial', verticalalignment='bottom',
             horizontalalignment='center', rotation=90)

    kaiju_kr2_coverd = patches.Rectangle((c, b - g), e, g, edgecolor='black', facecolor='red', alpha=al) #, hatch='///')
    plt.gca().add_patch(kaiju_kr2_coverd)
    plt.text(c + e / 2, b - g + 0.05, str(round(g / b * 100, 1)) + '%', transform=plt.gca().transData,
                color='black', weight='bold', fontsize=13,
                fontfamily='Arial', verticalalignment='bottom',
                horizontalalignment='center', rotation=0)
    plt.show()


def venn_multi():

    # fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    # Load data
    list = [0, 10, 12, 14, 16, 18, 20, 2, 4, 6, 8]
    # metabuli_only_total = 0
    meta_kaiju_kr2_total = 0
    kaiju_kr2_union_total = 0
    kaiju_kr2_total = 0
    kr2_kaiju_total = 0
    universe_total = 0
    kaiju_kr2_meta_total = 0
    kr2_kaiju_meta_total = 0
    kr2_and_kaiju_meta_total = 0
    kr2_and_kaiju_total = 0

    for i in list:
        metabuli_file_name = './venn/metabuli_' + str(i) + '.txt'
        kraken2_file_name = './venn/kraken2_' + str(i) + '.txt'
        kaiju_file_name = './venn/kaiju_' + str(i) + '.txt'

        metabuli = pd.read_csv(metabuli_file_name, sep='\t', header=None)[0].tolist()
        kraken2 = pd.read_csv(kraken2_file_name, sep='\t', header=None)[0].tolist()
        kaiju = pd.read_csv(kaiju_file_name, sep='\t', header=None)[0].tolist()

        kaiju_set = set(kaiju)
        kraken2_set = set(kraken2)
        metabuli_set = set(metabuli)

        # venn3([kaiju_set, kraken2_set, metabuli_set], set_labels=('Kaiju', 'Kraken2', 'Metabuli'), ax=axs[0])

        plt.show()

        kaiju_kr2_union = kaiju_set.union(kraken2_set)

        meta_kaiju_kr2_total += len(metabuli_set - kaiju_kr2_union)
        kaiju_kr2_union_total += len(kaiju_kr2_union)
        kaiju_kr2_total += len(kaiju_set - kraken2_set)
        kr2_kaiju_total += len(kraken2_set - kaiju_set)
        universe_total += len(metabuli_set.union(kaiju_kr2_union))
        kaiju_kr2_meta_total += len(kaiju_set - kraken2_set - metabuli_set)
        kr2_kaiju_meta_total += len(kraken2_set - kaiju_set - metabuli_set)
        kr2_and_kaiju_meta_total += len(kraken2_set.intersection(kaiju_set) - metabuli_set)
        kr2_and_kaiju_total += len(kaiju_set.intersection(kraken2_set))

    x = 4
    y = 4
    a = y * meta_kaiju_kr2_total / universe_total
    b = y - a
    c = x * kaiju_kr2_total / kaiju_kr2_union_total
    d = x * kr2_kaiju_total / kaiju_kr2_union_total
    e = x - c - d
    f = b * (1 - kaiju_kr2_meta_total / kaiju_kr2_total)
    h = b * (1 - kr2_kaiju_meta_total / kr2_kaiju_total)
    g = b * (1 - kr2_and_kaiju_meta_total / kr2_and_kaiju_total)
    print(a, b, c, d, e, f, g, h)
    print(meta_kaiju_kr2_total)
    # a = 0.15420469224599534
    # b = 3.845795307754005
    # c = 0.3679362130582213
    # d = 0.44474044799400575
    # e = 3.187323338947773
    # f = 3.204708673952227
    # g = 3.817218484933487
    # h = 3.6061251213481746

    X = np.array([0, x])
    Y = np.array([0, y])
    plt.figure(figsize=(4, 3.7))
    plt.plot(X, Y, color='None')

    hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

    metabuli_only = patches.Rectangle((0, x - a), x, a, edgecolor='black', facecolor='red', alpha=1) # fill=False, hatch='///')
    plt.gca().add_patch(metabuli_only)

    kaiju_kr2 = patches.Rectangle((0, 0), c, b, edgecolor='black', facecolor='green', alpha=1)
    plt.gca().add_patch(kaiju_kr2)

    kaiju_kr2_inter = patches.Rectangle((c, 0), e, b, edgecolor='black', facecolor='gray', alpha=1)
    plt.gca().add_patch(kaiju_kr2_inter)

    kr2_kaiju = patches.Rectangle((c + e, 0), d, b, edgecolor='black', facecolor='gold', alpha=1)
    plt.gca().add_patch(kr2_kaiju)

    kaiju_coverd = patches.Rectangle((0, b - f), c, f, edgecolor='black', facecolor='lightsalmon', alpha=1) #, hatch='///')
    plt.gca().add_patch(kaiju_coverd)
    plt.text(c / 2, b - f + 0.05, str(round(f / b * 100, 1)) + '%', transform=plt.gca().transData,
             color='black', weight='bold', fontsize=13,
             fontfamily='Arial', verticalalignment='bottom',
             horizontalalignment='center', rotation=90)

    kr2_coverd = patches.Rectangle((c + e, b - h), d, h, edgecolor='black', facecolor='lightsalmon', alpha=1) #, hatch='///')
    plt.gca().add_patch(kr2_coverd)
    plt.text(c + e + d / 2, b - h + 0.05, str(round(h / b * 100, 1)) + '%', transform=plt.gca().transData,
             color='black', weight='bold', fontsize=13,
             fontfamily='Arial', verticalalignment='bottom',
             horizontalalignment='center', rotation=90)

    kaiju_kr2_coverd = patches.Rectangle((c, b - g), e, g, edgecolor='black', facecolor='lightsalmon', alpha=1) #, hatch='///')
    plt.gca().add_patch(kaiju_kr2_coverd)
    plt.text(c + e / 2, b - g + 0.05, str(round(g / b * 100, 1)) + '%', transform=plt.gca().transData,
                color='black', weight='bold', fontsize=13,
                fontfamily='Arial', verticalalignment='bottom',
                horizontalalignment='center', rotation=0)

    plt.show()


if __name__ == '__main__':
    venn_multi()
    # venn_hatch()
    # venn()
