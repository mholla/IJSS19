from IJSS19_subroutines_calculate import *

warnings.simplefilter('ignore')

if __name__ == '__main__':

    ####################################################################################
    # parameters
    ####################################################################################

    # parameters for root finding (Lbar = 2.*pi/Kbar = L/H_l = 2.*pi/KH_l)
    n_wavelengths = 999
    wavelengths = numpy.logspace(-1., 3., num=n_wavelengths)
    wavelengths = wavelengths[::-1]  # start at right end of the graph (where the distance between roots is larger)

    # parameters for output
    findroots = True  # only set to false for troubleshooting, using plotroots below
    plotroots = False  # save plot of absolute value of determinant at each n_wavelengths
    plotindcurves = False  # save plot of individual curves for each beta
    printoutput = False  # print every root found at every n_wavelengths

    options = [findroots, plotroots, plotindcurves, printoutput]

    ####################################################################################
    # # Fig 4 - heat maps
    # ####################################################################################
    description = 'Fig4'
    mode = 'hetero'
    bs_m = numpy.linspace(0., 2., 41)
    bs_p = bs_m

    solve_and_write(description, mode, bs_m, bs_p, wavelengths, options)

    ###################################################################################
    # Fig. 5 - beta_plus = 0.1
    ###################################################################################
    description = 'Fig5'
    mode = 'hetero'
    bs_m = [0.1]
    a = numpy.linspace(0., 1., 21)
    b = numpy.linspace(1., 1.5, 6)
    c = numpy.array([1.75, 2.])
    bs_p = numpy.concatenate((a, b, c))

    solve_and_write(description, mode, bs_m, bs_p, wavelengths, options, crit=True)

    ####################################################################################
    # Fig. 6 - regions A, B, C
    ####################################################################################
    mode = 'hetero'

    # Fig 6, 1 and 3
    a = numpy.linspace(0., 0.9, 10)
    b = numpy.array([0.98])
    bs_m = numpy.concatenate((a, b))

    description = 'Fig6A'
    a = numpy.linspace(0.01, 0.1, 10)
    b = numpy.linspace(0.1, 0.9, 17)
    c = numpy.linspace(0.90, 0.98, 9)
    bs_p = numpy.concatenate((a, b, c))
    solve_and_write(description, mode, bs_m, bs_p, wavelengths, options)

    description = 'Fig6C1'
    bs_p = numpy.linspace(1.0, 5.0, 81)
    solve_and_write(description, mode, bs_m, bs_p, wavelengths, options)

    # Fig 6, 2 and 4
    bs_m = numpy.array([1.2, 2.0, 3.0, 4.0, 5.0])

    description = 'Fig6C2'
    bs_p = numpy.linspace(0.05, 1.0, 20)
    solve_and_write(description, mode, bs_m, bs_p, wavelengths, options)

    description = 'Fig6B'
    bs_p = numpy.linspace(1.05, 5.0, 80)
    solve_and_write(description, mode, bs_m, bs_p, wavelengths, options)

    ####################################################################################
    # Fig. 7 - serpentine vs. pinched
    ####################################################################################

    modes = ['serpentine', 'pinched']

    bs = numpy.array([0.1, 10.])
    for mode in modes:
        description = 'Fig7_{mode}'.format(mode=mode)
        solve_and_write(description, mode, bs, bs, wavelengths, options, crit=True)

    bs = numpy.array([10.])
    for mode in modes:
        description = 'Fig7_inset_{mode}'.format(mode=mode)
        solve_and_write(description, mode, bs, bs, wavelengths, options, crit=True)

    ####################################################################################
    # Fig. 8 - homogeneous matrix 
    ####################################################################################
    mode = 'serpentine'
    a = numpy.linspace(0.0, 0.975, 40)
    b = numpy.linspace(1.0, 10., 91)
    bs = numpy.concatenate((a, b))

    description = 'Fig8'
    solve_and_write(description, mode, bs, bs, wavelengths, options)
