import string
import warnings
from math import *

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

warnings.simplefilter('ignore')


def plot_curves(figname, betas_m, betas_p, crit_wavelengths, crit_strains, axes, mode='hetero'):
    """ Plot crit_strain vs. wavelength for each combination of beta_m and beta_p

    Parameters
    ----------
    figname : string
        figure number corresponding to paper
    betas_m : array of floats
        stiffness ratios of lower matrix (matrix/layer) 
    betas_p : array of floats
        stiffness ratios of upper matrix (matrix/layer) 
    crit_wavelengths : list of floats
        list of wavelengths
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength
    axes : list
        list of axes limits (x_min, x_max, y_min, y_max)
    mode : string
        type of instability - 'hetero', 'pinched', or 'serpentine'

    Returns
    -------
    None
    """

    plt.figure(figname)

    for i in range(len(betas_m)):
        beta_m = betas_m[i]

        for j in range(len(betas_p)):

            # for homogeneous problems, only consider beta_m = beta_p
            if mode in ['serpentine', 'pinched]']:
                beta_p = beta_m
            else:
                beta_p = betas_p[j]

            # change width of major lines
            if beta_p in [0., 0.5, 1., 2.]:
                width = 3
            else:
                width = 1

            # set colors to match regions A, B, C in Fig. 3
            if beta_m == 1. or beta_p == 1.:  # line
                color = 'k'
            elif beta_m < 1. and beta_p < 1.:  # A
                color = 'red'
            elif beta_m > 1. and beta_p > 1.:  # C
                color = 'blue'
            else:  # B
                color = 'darkviolet'

            if mode in ['serpentine', 'pinched']:
                color = 'k'

            # change linestyle for pinched vs. symmetric 
            if mode == 'pinched':
                linestyle = '--'
            else:
                linestyle = '-'

            plt.plot(crit_wavelengths, crit_strains[i][j], color=color, linestyle=linestyle, linewidth=width)

    plt.xlabel('normalized wavenumber $L/H$')
    plt.ylabel('critical axial strain $\\epsilon_1$')
    plt.gca().set_xlim(axes[0], axes[1])
    plt.gca().set_ylim(axes[2], axes[3])
    plt.savefig('{filename}.pdf'.format(filename=figname))


def plot_wavelengths(figname, mode, betas_m, betas_p, thresh_wavelengths, color, axes):
    """ Plot threshold wavelengths vs. beta_p for each beta_m

    Parameters
    ----------
    figname : string
        figure number corresponding to paper
    mode : string
        type of instability - 'hetero', 'pinched', or 'serpentine'
    betas_m : array of floats
        stiffness ratios of lower matrix (matrix/layer) 
    betas_p : array of floats
        stiffness ratios of upper matrix (matrix/layer) 
    thresh_wavelengths : array of floats
        array of threshold wavelengths
    color : string 
        line colors, set to match regions A, B, C in Fig. 3
    axes : list
        list of axes limits (x_min, x_max, y_min, y_max)

    Returns
    -------
    None
    """

    plt.figure(figname)

    for i in range(len(betas_m)):

        # change width of major lines
        if betas_m[i] in [0.0, 0.5, 0.98, 1.0, 1.2, 3., 5.]:
            width = 3
        else:
            width = 1

        # for homogeneous problems, only consider beta_m = beta_p
        if mode in ['pinched', 'serpentine']:
            wavelengths = numpy.zeros(len(thresh_wavelengths))
            for j in range(len(thresh_wavelengths)):
                wavelengths[j] = thresh_wavelengths[j][j]
        else:
            wavelengths = thresh_wavelengths[i]

        plt.plot(betas_p, wavelengths, color=color, linestyle='-', linewidth=width)

    plt.gca().set_xlim(axes[0], axes[1])
    plt.gca().set_ylim(axes[2], axes[3])
    plt.savefig('{figname}.pdf'.format(figname=figname))


def plot_strains(figname, mode, betas_m, betas_p, thresh_strains, color, axes):
    """ Plot threshold strains vs. beta_p for each beta_m

    Parameters
    ----------
    figname : string
        figure number corresponding to paper
    mode : string
        type of instability - 'hetero', 'pinched', or 'serpentine'
    betas_m : array of floats
        stiffness ratios of lower matrix (matrix/layer) 
    betas_p : array of floats
        stiffness ratios of upper matrix (matrix/layer) 
    thresh_strains : array of floats
        array of threshold strains
    color : string 
        line colors, set to match regions A, B, C in Fig. 3
    axes : list
        list of axes limits (x_min, x_max, y_min, y_max)

    Returns
    -------
    None
    """

    plt.figure(figname)

    for i in range(len(betas_m)):

        # change width of major lines
        if betas_m[i] in [0.0, 0.5, 0.98, 1.0, 1.2, 3., 5.]:
            width = 3
        else:
            width = 1

        # for homogeneous problems, only consider beta_m = beta_p
        if mode in ['pinched', 'serpentine']:
            strains = numpy.zeros(len(thresh_strains))
            for j in range(len(thresh_strains)):
                strains[j] = thresh_strains[j][j]
        else:
            strains = thresh_strains[i]

        plt.plot(betas_p, strains, color=color, linestyle='-', linewidth=width)

    plt.gca().set_xlim(axes[0], axes[1])
    plt.gca().set_ylim(axes[2], axes[3])
    plt.savefig('{figname}.pdf'.format(figname=figname))


def heat_maps(figname, betas_m, betas_p, thresh_strains, thresh_wavelengths): 
    """ Plot heat maps of threshold strains and wavelengths

    Parameters
    ----------
    figname : string
        figure number corresponding to paper
    betas_m : array of floats
        stiffness ratios of lower matrix (matrix/layer) 
    betas_p : array of floats
        stiffness ratios of upper matrix (matrix/layer) 
    thresh_strains : array of floats
        array of threshold strains

    Returns
    -------
    None
    """

    fig = plt.figure(2, figsize=(6, 6))  # default is (8,6)
    ax = fig.add_subplot(111, aspect='equal')
    ax.contourf(betas_m, betas_p, thresh_strains, 20, cmap='jet', vmax=1., vmin=0., interpolation='none')
    ax.axvline(x=1, color='w', linewidth=4)
    ax.axhline(y=1, color='w', linewidth=4)
    ax.plot([0., 2.], [0., 2.], color='k', linewidth=4)
    ax.axis([0., 2., 0., 2.])
    plt.savefig('{figname}A.png'.format(figname=figname))

    fig = plt.figure(1, figsize=(6, 6))  # default is (8,6)
    ax = fig.add_subplot(111, aspect='equal')
    ax.contourf(betas_m, betas_p, thresh_wavelengths, 20, cmap='jet', vmax=100., vmin=0., interpolation='none')
    ax.axvline(x=1, color='w', linewidth=4)
    ax.axhline(y=1, color='w', linewidth=4)
    ax.plot([0., 2.], [0., 2.], color='k', linewidth=4)
    ax.axis([0., 2., 0., 2.])
    plt.savefig('{figname}B.png'.format(figname=figname))

    fig = plt.figure(figsize=(10., 1.))
    ax = fig.add_axes([0.05, 0.25, 0.9, 0.5])  # (left, bottom, width, height )
    cb = mpl.colorbar.ColorbarBase(ax, cmap='jet', orientation='horizontal')
    plt.savefig('{figname}_colorbar.png'.format(figname=figname))


def read_thresh_values(filename, mode, bs_m, bs_p):
    """ Read threshold values from files

    Parameters
    ----------
    filename : string
        name of results file
    mode : string
        type of instability - 'hetero', 'pinched', or 'serpentine'
    bs_m : list of floats
        stiffness ratios of lower matrix (matrix/layer) 
    bs_p : list of floats
        stiffness ratios of upper matrix (matrix/layer)     

    Returns
    -------
    thresh_strains : list of floats
        list of threshold strain values 
    thresh_wavelengths : list of floats
        list of threshold wavelengths
    
    Notes
    -----
    Results were only written for homogeneous cases when beta_m = beta_p
    """

    m = len(bs_m)
    n = len(bs_p)

    thresh_wavelengths = numpy.zeros((m, n))
    thresh_strains = numpy.zeros((m, n))

    with open('results_{filename}.txt'.format(filename=filename), 'r') as f:

        lines = f.readlines()

        k = 0

        for i in range(m):

            for j in range(n):

                diagonal = True
                if mode in ['serpentine', 'pinched']:
                    if i != j:
                        diagonal = False

                if diagonal:
                    values = lines[k].split()
                    beta_m = float(values[0])
                    beta_p = float(values[1])
                    strain = float(values[2])
                    wavelength = float(values[3])

                    # check that correct line is being read
                    if abs(bs_m[i] - beta_m) > 0.0001:
                        print("something is wrong with minus", bs_m[i], beta_m)
                    if abs(bs_p[j] - beta_p) > 0.0001:
                        print("something is wrong with plus", bs_p[j], beta_p)

                    thresh_wavelengths[i][j] = wavelength
                    thresh_strains[i][j] = strain

                    k = k + 1

    return thresh_strains, thresh_wavelengths


def read_crit_values(filename, mode, bs_m, bs_p):
    """ Read critical values from files

    Parameters
    ----------
    filename : string
        name of results file
    mode : string
        type of instability - 'hetero', 'pinched', or 'serpentine'
    bs_m : list of floats
        stiffness ratios of lower matrix (matrix/layer) 
    bs_p : list of floats
        stiffness ratios of upper matrix (matrix/layer)     

    Returns
    -------
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength
    crit_wavelengths : list of floats
        list of wavelengths
    
    Notes
    -----
    Results were only written for homogeneous cases when beta_m = beta_p
    """

    crit_wavelengths = numpy.loadtxt('results_{filename}_wavelengths.txt'.format(filename=filename), dtype=float)

    m = len(bs_m)
    n = len(bs_p)
    p = len(crit_wavelengths)

    crit_strains = numpy.zeros((m, n, p))

    with open('results_{filename}_strains.txt'.format(filename=filename), 'r') as f:

        lines = f.readlines()

        l = 0

        for i in range(m):

            for j in range(n):

                diagonal = True
                if mode in ['serpentine', 'pinched']:
                    if i != j:
                        diagonal = False

                if diagonal:

                    values = lines[l].split()
                    beta_m = float(values[0])
                    beta_p = float(values[1])

                    # check that correct line is being read
                    if abs(bs_m[i] - beta_m) > 0.0001:
                        print("something is wrong with minus", bs_m[i], beta_m)
                    if abs(bs_p[j] - beta_p) > 0.0001:
                        print("something is wrong with plus", bs_p[j], beta_p)

                    for k in range(p):
                        # print(i,j,k,l)
                        crit_strains[i][j][k] = float(values[k + 2])

                    l = l + 1

    return crit_strains, crit_wavelengths
