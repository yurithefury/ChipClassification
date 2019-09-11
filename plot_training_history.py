import matplotlib.pyplot as plt
import pandas as pd
import sys

def plots(fname):
    h = pd.read_csv(fname, index_col=0)
    fig, axs = plt.subplots(2,1,sharex=True)

    axs[0].plot(h.index.values, h['loss'], 'r')
    axs[0].plot(h.index.values, h['val_loss'], 'b')
    axs[1].plot(h.index.values, h['balanced_accuracy_score'], 'r')
    axs[1].plot(h.index.values, h['val_balanced_accuracy_score'], 'b')

    axs[1].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[1].set_ylabel('BA')

    plt.show()

try:
    plots(sys.argv[1])
except IndexError:
    print('usage: plot_history path/to/history.csv')
