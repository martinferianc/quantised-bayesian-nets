import matplotlib.pyplot as plt
import matplotlib as mlp
import brewer2mpl

LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot']
bmap = brewer2mpl.get_map('Set1', 'qualitative', 9)

COLORS = bmap.mpl_colors

params = {
    'figure.dpi':150,
    'axes.labelsize': 10,
    'font.size': 8,
    'legend.fontsize': 10,
    'legend.fancybox': False,
    'legend.framealpha': .5,
    'legend.frameon':False,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': True,
    'figure.figsize': [7.5, 5.5],
    'mathtext.default':'regular',
}

mlp.rcParams.update(params)
mlp.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

MLP = mlp 
PLT = plt