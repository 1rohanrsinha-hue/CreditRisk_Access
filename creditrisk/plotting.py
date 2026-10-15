"""Figure style: fixed-order categorical hues (validated reference palette),
thin marks, recessive grid, 300 dpi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#4a3aa7', '#e34948', '#eda100', '#008300']
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e4e3dc'


def style():
    plt.rcParams.update({
        'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 9, 'axes.edgecolor': MUTED,
        'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'lines.linewidth': 1.6, 'legend.frameon': False, 'axes.titlesize': 9.5,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    })
