
from IJSS19_subroutines_plot import *

warnings.simplefilter('ignore')

if __name__ == '__main__':

    ####################################################################################
    # Fig 4 - heat maps
    ####################################################################################
    filename = 'Fig4'
    mode = 'hetero'
    betas_m = numpy.linspace(0., 2., 41)
    betas_p = betas_m

    [thresh_strains, thresh_wavelengths] = read_thresh_values(filename, mode, betas_m, betas_p)
    heat_maps(filename, betas_m, betas_p, thresh_strains, thresh_wavelengths)


    ###################################################################################
    # Fig. 5 - beta_plus = 0.1
    ####################################################################################
    filename = 'Fig5'
    mode = 'hetero'
    betas_m = [0.1]
    a = numpy.linspace(0., 1., 21)
    b = numpy.linspace(1., 1.5, 6)
    c = numpy.array([1.75, 2.])
    betas_p = numpy.concatenate((a, b, c))

    [crit_strains, crit_wavelengths] = read_crit_values(filename, mode, betas_m, betas_p)
    plot_curves(filename, betas_m, betas_p, crit_wavelengths, crit_strains, [0., 200., 0., 0.65], mode)

    ####################################################################################
    # Fig. 6 - regions A, B, C
    ####################################################################################
    mode = 'hetero'

    # Fig 6, 1 and 3
    a = numpy.linspace(0., 0.9, 10)
    b = numpy.array([0.98])
    bs_m = numpy.concatenate((a, b))

    filename = 'Fig6A'
    a = numpy.linspace(0.01, 0.1, 10)
    b = numpy.linspace(0.1, 0.9, 17)
    c = numpy.linspace(0.90, 0.98, 9)
    bs_p = numpy.concatenate((a, b, c))
    
    [thresh_strains, thresh_wavelengths] = read_thresh_values(filename, mode, bs_m, bs_p)
    plot_strains('Fig6_1', mode, bs_m, bs_p, thresh_strains, 'r', [0.0, 5.0, 0.0, 1.0])
    plot_wavelengths('Fig6_3', mode, bs_m, bs_p, thresh_wavelengths, 'r', [0.0, 1.0, 0.0, 100.])

    filename = 'Fig6C1'
    bs_p = numpy.linspace(1.0, 5.0, 81)
    
    [thresh_strains, thresh_wavelengths] = read_thresh_values(filename, mode, bs_m, bs_p)
    plot_strains('Fig6_1', mode, bs_m, bs_p, thresh_strains, 'darkviolet', [0.0, 5.0, 0.0, 1.0])
    plot_wavelengths('Fig6_3', mode, bs_m, bs_p, thresh_wavelengths, 'darkviolet', [0.0, 1.0, 0.0, 100.])

    # Fig 6, 2 and 4
    bs_m = numpy.array([1.2, 2.0, 3.0, 4.0, 5.0])

    filename = 'Fig6C2'
    bs_p = numpy.linspace(0.05, 1.0, 20)
    
    [thresh_strains, thresh_wavelengths] = read_thresh_values(filename, mode, bs_m, bs_p)
    plot_strains('Fig6_2', mode, bs_m, bs_p, thresh_strains, 'darkviolet', [0.0, 5.0, 0.0, 1.0])
    plot_wavelengths('Fig6_4', mode, bs_m, bs_p, thresh_wavelengths, 'darkviolet', [1.0, 5.0, 0.0, 100.])

    filename = 'Fig6B'
    bs_p = numpy.linspace(1.05, 5.0, 80)
    
    [thresh_strains, thresh_wavelengths] = read_thresh_values(filename, mode, bs_m, bs_p)
    plot_strains('Fig6_2', mode, bs_m, bs_p, thresh_strains, 'b', [0.0, 5.0, 0.0, 1.0])
    plot_wavelengths('Fig6_4', mode, bs_m, bs_p, thresh_wavelengths, 'b', [1.0, 5.0, 0.0, 100.])

    ####################################################################################
    # Fig. 7 - serpentine vs. pinched
    ####################################################################################
    modes = ['serpentine', 'pinched']
    bs = numpy.array([0.1, 10])
    bs_inset = numpy.array([10.])

    for mode in modes:
        filename = 'Fig7_{mode}'.format(mode=mode)
        [crit_strains, crit_wavelengths] = read_crit_values(filename, mode, bs, bs)
        plot_curves('Fig7', bs, bs, crit_wavelengths, crit_strains, [0., 40., 0., 1.75], mode)

        filename = 'Fig7_inset_{mode}'.format(mode=mode)
        [crit_strains, crit_wavelengths] = read_crit_values(filename, mode, bs_inset, bs_inset)
        plot_curves('Fig7_inset', bs_inset, bs_inset, crit_wavelengths, crit_strains, [0., 30., 0.525, 0.535], mode)

    ####################################################################################
    # Fig. 8 - homogeneous matrix 
    ####################################################################################
    filename = 'Fig8'
    mode = 'serpentine'

    a = numpy.linspace(0.0, 0.975, 40)
    b = numpy.linspace(1.0, 10., 91)
    bs = numpy.concatenate((a, b))

    [thresh_strains, thresh_wavelengths] = read_thresh_values('Fig8'.format(mode=mode), mode, bs, bs)
    plot_strains('Fig8A', mode, [0.1], bs, thresh_strains, 'k', [0.0, 10.0, 0.0, 1.0])
    plot_wavelengths('Fig8B', mode, [0.1], bs, thresh_wavelengths, 'k', [0.0, 10.0, 0.0, 100.])
