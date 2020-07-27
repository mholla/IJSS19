import string
import warnings
from math import *

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

warnings.simplefilter('ignore')


def determinant_H(lam, beta_m, beta_p, Kbar):
    """ Return determinant of 8x8 coefficient matrix for heterogeneous case 

    Parameters
    ----------
    lam : float
        applied axial compression
    beta_m : float
        stiffness ratio of lower matrix (matrix/layer) 
    beta_p : float
        stiffness ratio of upper matrix (matrix/layer) 
    Kbar : float
        wavenumber, normalized by layer thickness

    Returns
    -------
    dd : float
        determinant of coefficient matrix

    Notes
    -----
    If the determinant is too large to compute, returns None.
    """

    LAMBDA = 1. + lam ** 4.
    AA = numpy.zeros((8, 8), dtype='float64')

    # Eq. 22 in paper
    try:
        AA[0][0] = +exp(-Kbar / lam ** 2.)
        AA[0][1] = -exp(+Kbar / lam ** 2.)
        AA[0][2] = +lam ** 2. * exp(-Kbar)
        AA[0][3] = -lam ** 2. * exp(+Kbar)
        AA[0][4] = -1.
        AA[0][5] = -lam ** 2.
        AA[0][6] = 0.
        AA[0][7] = 0.

        AA[1][0] = exp(-Kbar / lam ** 2.)
        AA[1][1] = exp(+Kbar / lam ** 2.)
        AA[1][2] = exp(-Kbar)
        AA[1][3] = exp(+Kbar)
        AA[1][4] = -1.
        AA[1][5] = -1.
        AA[1][6] = 0.
        AA[1][7] = 0.

        AA[2][0] = -2. * exp(-Kbar / lam ** 2.)
        AA[2][1] = -2. * exp(+Kbar / lam ** 2.)
        AA[2][2] = -LAMBDA * exp(-Kbar)
        AA[2][3] = -LAMBDA * exp(+Kbar)
        AA[2][4] = 2. * beta_p
        AA[2][5] = LAMBDA * beta_p
        AA[2][6] = 0.
        AA[2][7] = 0.

        AA[3][0] = +LAMBDA * exp(-Kbar / lam ** 2.)
        AA[3][1] = -LAMBDA * exp(+Kbar / lam ** 2.)
        AA[3][2] = +2. * lam ** 2. * exp(-Kbar)
        AA[3][3] = -2. * lam ** 2. * exp(+Kbar)
        AA[3][4] = -LAMBDA * beta_p
        AA[3][5] = -2. * lam ** 2. * beta_p
        AA[3][6] = 0.
        AA[3][7] = 0.

        AA[4][0] = +exp(+Kbar / lam ** 2.)
        AA[4][1] = -exp(-Kbar / lam ** 2.)
        AA[4][2] = +lam ** 2. * exp(+Kbar)
        AA[4][3] = -lam ** 2. * exp(-Kbar)
        AA[4][4] = 0.
        AA[4][5] = 0.
        AA[4][6] = 1.
        AA[4][7] = lam ** 2.

        AA[5][0] = exp(+Kbar / lam ** 2.)
        AA[5][1] = exp(-Kbar / lam ** 2.)
        AA[5][2] = exp(+Kbar)
        AA[5][3] = exp(-Kbar)
        AA[5][4] = 0.
        AA[5][5] = 0.
        AA[5][6] = -1.
        AA[5][7] = -1.

        AA[6][0] = -2. * exp(+Kbar / lam ** 2.)
        AA[6][1] = -2. * exp(-Kbar / lam ** 2.)
        AA[6][2] = -LAMBDA * exp(+Kbar)
        AA[6][3] = -LAMBDA * exp(-Kbar)
        AA[6][4] = 0.
        AA[6][5] = 0.
        AA[6][6] = 2. * beta_m
        AA[6][7] = LAMBDA * beta_m

        AA[7][0] = +LAMBDA * exp(+Kbar / lam ** 2.)
        AA[7][1] = -LAMBDA * exp(-Kbar / lam ** 2.)
        AA[7][2] = +2. * lam ** 2. * exp(+Kbar)
        AA[7][3] = -2. * lam ** 2. * exp(-Kbar)
        AA[7][4] = 0.
        AA[7][5] = 0.
        AA[7][6] = LAMBDA * beta_m
        AA[7][7] = 2. * lam ** 2. * beta_m

        dd = numpy.linalg.det(AA)

        if isinf(dd):
            dd = None
            # print("infinity at lam = ", lam)

    except (OverflowError):
        dd = None
        # print("overflow at lam = ", lam)

    return dd


def determinant_S(lam, beta, beta_check, Kbar):
    """ Return determinant of 4x4 coefficient matrix for serpentine/symmetric case 

    Parameters
    ----------
    lam : float
        applied axial compression
    beta : float
        stiffness ratio (matrix/layer) 
    beta_check : float
        stiffness ratio passed in as a check
    Kbar : float
        wavenumber, normalized by layer thickness

    Returns
    -------
    dd : float
        determinant of coefficient matrix

    Notes
    -----
    If the determinant is too large to compute, returns None.
    """

    LAMBDA = 1. + lam ** 6.
    AA = numpy.zeros((4, 4), dtype='float64')

    if beta != beta_check:
        print("beta = {m} != {p} for the homogeneous case")

    # Eq. 24 in paper
    try:
        AA[0][0] = numpy.sinh(Kbar / lam ** 2.)
        AA[0][1] = lam ** 2. * numpy.sinh(Kbar)
        AA[0][2] = 1.
        AA[0][3] = lam ** 2.

        AA[1][0] = numpy.cosh(Kbar / lam ** 2.)
        AA[1][1] = numpy.cosh(Kbar)
        AA[1][2] = -1.
        AA[1][3] = -1.

        AA[2][0] = +2. * numpy.cosh(Kbar / lam ** 2.)
        AA[2][1] = +LAMBDA * numpy.cosh(Kbar)
        AA[2][2] = -2. * beta
        AA[2][3] = -LAMBDA * beta

        AA[3][0] = LAMBDA * numpy.sinh(Kbar / lam ** 2.)
        AA[3][1] = 2. * lam ** 2. * numpy.sinh(Kbar)
        AA[3][2] = LAMBDA * beta
        AA[3][3] = 2. * lam ** 2. * beta

        dd = numpy.linalg.det(AA)

        if isinf(dd):
            dd = None
            # print("infinity at lam = ", lam)

    except (OverflowError):
        dd = None
        # print("overflow at lam = ", lam)

    return dd


def determinant_P(lam, beta, beta_check, Kbar):
    """ Return determinant of 4x4 coefficient matrix for pinched/antisymmetric case 

    Parameters
    ----------
    lam : float
        applied axial compression
    beta : float
        stiffness ratio (matrix/layer) 
    beta_check : float
        stiffness ratio passed in as a check
    Kbar : float
        wavenumber, normalized by layer thickness

    Returns
    -------
    dd : float
        determinant of coefficient matrix

    Notes
    -----
    If the determinant is too large to compute, returns None.
    """

    LAMBDA = 1. + lam ** 6.
    AA = numpy.zeros((4, 4), dtype='float64')

    if beta != beta_check:
        print("beta = {m} != {p} for the homogeneous case")

    # Eq. 26 in paper
    try:
        AA[0][0] = numpy.cosh(Kbar / lam ** 2.)
        AA[0][1] = lam ** 2. * numpy.cosh(Kbar)
        AA[0][2] = -1.
        AA[0][3] = -lam ** 2.

        AA[1][0] = numpy.sinh(Kbar / lam ** 2.)
        AA[1][1] = numpy.sinh(Kbar)
        AA[1][2] = 1.
        AA[1][3] = 1.

        AA[2][0] = 2. * numpy.sinh(Kbar / lam ** 2.)
        AA[2][1] = LAMBDA * numpy.sinh(Kbar)
        AA[2][2] = 2. * beta
        AA[2][3] = LAMBDA * beta

        AA[3][0] = LAMBDA * numpy.cosh(Kbar / lam ** 2.)
        AA[3][1] = 2. * lam ** 2. * numpy.cosh(Kbar)
        AA[3][2] = -LAMBDA * beta
        AA[3][3] = -2. * lam ** 2. * beta

        dd = numpy.linalg.det(AA)

        if isinf(dd):
            dd = None
            # print("infinity at lam = ", lam)

    except (OverflowError):
        dd = None
        # print("overflow at lam = ", lam)

    return dd


def Ridder(a, b, determinant, beta_m, beta_p, Kbar, tol=1.e-12, nmax=50):
    """ Uses Ridders' method to find critical strain (between a and b)

    Parameters
    ----------
    a, b : floats
        upper and lower brackets of axial compression, lambda, for Ridders' method
    determinant : function
        name of function that returns relevant determinant
    beta_m : float
        stiffness ratio of lower matrix (matrix/layer) 
    beta_p : float
        stiffness ratio of upper matrix (matrix/layer) 
    Kbar : float
        wavenumber, normalized by layer thickness
    tol : float
        tolerance for Ridders' method; solution will be returned when the value of the function is within the tolerance
    nmax : int
        maximum number of iterations before exiting

    Returns
    -------
    lambda_crit : float
        value of axial compression, lambda, that satisfies eigenvalue problem
    n_iter : int
        number of iterations before lambda_crit was found

    Notes
    -----
    Based on based on https://en.wikipedia.org/wiki/Ridders%27_method
    """

    nmax = 50

    fa = determinant(a, beta_m, beta_p, Kbar)
    fb = determinant(b, beta_m, beta_p, Kbar)

    if fa == 0.0:
        # print("lower bracket is root")
        return a, 0
    if fb == 0.0:
        # print("upper bracket is root")
        return b, 0
    if fa * fb > 0.0:
        # print("Root is not bracketed between a = {a} and b = {b}".format(a=a, b=b))
        return None, None

    # iterate to find lambda_crit
    for i in range(nmax):
        c = 0.5 * (a + b)
        fc = determinant(c, beta_m, beta_p, Kbar)

        if fc == None:
            return None, i

        s = sqrt(fc ** 2. - fa * fb)
        if s == 0.0:
            return None, i

        dx = (c - a) * fc / s
        if (fa - fb) < 0.0:
            dx = -dx
        x = c + dx

        fx = determinant(x, beta_m, beta_p, Kbar)

        # check for convergence
        if i > 0:
            if abs(x - xOld) < tol * max(abs(x), 1.0):
                return x, i
        xOld = x

        # rebracket root
        if fc * fx > 0.0:
            if fa * fx < 0.0:
                b = x
                fb = fx
            else:
                a = x
                fa = fx
        else:
            a = c
            b = x
            fa = fc
            fb = fx

    res = abs(x - xOld) / max(abs(x), 1.0)

    print('Too many iterations, res = {res:e}'.format(res=res))
    return None, nmax


def check_roots(determinant, lams, beta_m, beta_p, Kbar, plotroots):
    """ Calculates determinant at every lambda to check for existence of roots

    Parameters
    ----------
    determinant : function
        name of function that returns relevant determinant
    lams : array of floats
        strains to consider when checking for existence of roots
    beta_m : float
        stiffness ratio of lower matrix (matrix/layer) 
    beta_p : float
        stiffness ratio of upper matrix (matrix/layer) 
    Kbar : float
        wavenumber, normalized by layer thickness
    plotroots : boolean
        plot lines showing positive or negative value at all npts for each wavelength
    
    Returns
    -------
    root_exists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    a : float
        smallest lambda value for which a real determinant was calculated (used as lower bound for Ridder algorithm)
    c : float
        largest lambda value for which a real determinant was calculated (used as lower bound for Ridder algorithm)
    """

    n = len(lams)

    dds = numpy.zeros(n, dtype='float64')
    dds_abs = numpy.zeros(n, dtype='float64')

    rootexists = False
    a = 0.
    c = 0.

    # move backwards, in order of decreasing strain and report values bracketing highest root
    for i in range(n):
        lam = lams[i]
        dds[i] = determinant(lam, beta_m, beta_p, Kbar)

        # if encountering NaN before finding root:
        if isnan(dds[i]) and not rootexists:
            rootexists = False
            a = 0.
            c = 0.

        # otherwise, step through
        if i > 0 and not rootexists:
            dds_abs[i] = dds[i] / abs(dds[i])
            # if encountering root
            if dds[i] * dds[i - 1] < 0.:  # sign change
                rootexists = True
                a = lams[i - 1]
                c = lams[i]

    if plotroots:
        plt.figure()
        plt.axis([0, 1.1, -1.5, 1.5])
        Lbar = 2. * pi / Kbar
        title = 'Lbar = {Lbar}'.format(Lbar=Lbar)
        plt.title(title)
        plt.xlabel('$\lambda$')
        plt.ylabel('energy')

        plt.axvline(x=1., linestyle='--', color='k')
        plt.axhline(y=0., linestyle='--', color='k')
        plt.plot(lams, dds_abs, color='b', linestyle='-')
        plt.savefig('Lbar_{Lbar}.png'.format(Lbar=Lbar))

    return rootexists, a, c


def find_roots(rootexists, determinant, crit_strains, a, b, c, beta_m, beta_p, Kbar, printoutput, tol=1.e-12):
    """ Calls Ridder algorithm to find roots

    Parameters
    ----------
    root_exists : boolean
        boolean value indicating whether or not a root (sign change) was detected
    determinant : function
        name of function that returns relevant determinant
    crit_strains : list of floats
        list of critical strain values corresponding to given wavelengths (appended each time this is called)
    a, b, c : floats
        lower, mid, and upper brackets for Ridder's algorithm
    beta_m : float
        stiffness ratio of lower matrix (matrix/layer) 
    beta_p : float
        stiffness ratio of upper matrix (matrix/layer) 
    Kbar : float
        wavenumber, normalized by layer thickness
    printoutput : boolean
        whether or not to print every root found at every wavelength
    tol : float
        tolerance for Ridders' method; solution will be returned when the value of the function is within the tolerance

    Returns
    -------
    crit_strains : list of floats
        list of critical strain values corresponding to given wavelengths (appended each time this is called)
    b : float
        midpoint value of axial compression lambda, used to distinguish between double roots.  Updated each time function is called, and used for the next call

    Notes
    -----
    If root_exists = False, then the strain value of 0 is appended
    """

    minimum = a

    if printoutput:
        Lbar = 2. * pi / Kbar
        print("Lbar = {Lbar:0.2f}, a = {a}, c = {c}".format(Lbar=Lbar, a=a, c=c))

    if rootexists:
        [lam, n] = Ridder(a, c, determinant, beta_m, beta_p, Kbar, tol)

        # in the case of double roots
        if lam is None:
            [lam_min, n] = Ridder(a, b, determinant, beta_m, beta_p, Kbar, tol)
            [lam_max, n] = Ridder(b, c, determinant, beta_m, beta_p, Kbar, tol)
            if lam_max is None:
                if lam_min is None:
                    lam = 1.0
                else:
                    lam = lam_min
            else:
                b = 0.5 * (lam_min + lam_max)
                lam = lam_max
        else:
            b = 0.5 * (lam + minimum)

        if printoutput:
            print(" lam = {lam:0.5f}, n = {n}".format(lam=lam, n=n))

    else:  # no root means there is no strain that can cause this system to buckle.  Give it a high strain
        lam = 0.
        n = 1.

    if printoutput:
        print("Lbar = {Lbar:0.2f}, no root".format(Lbar=Lbar))

    crit_strains.append(1. - lam)

    return crit_strains, b


def find_crit_values(mode, beta_m, beta_p, wavelengths, options, tol=1.e-12):
    """ Finds critical strain for each specified wavelength

    Parameters
    ----------
    mode : string
        type of instability - 'hetero', 'pinched', or 'serpentine'
    beta_m : float
        stiffness ratio of lower matrix (matrix/layer) 
    beta_p : float
        stiffness ratio of upper matrix (matrix/layer) 
    wavelengths : list of floats
        list of wavelengths for which to calculate determinant
        number of strain values to consider when checking for existence of roots
    options : list of booleans
        findroots, plotroots, plotindcurves, printoutput options
    tol : float
        tolerance for Ridders' method; solution will be returned when the value of the function is within the tolerance
    
    Returns
    -------
    crit_strains : list of floats
        list of crit_strain associated with each wavelength    
    """

    [findroots, plotroots, plotindcurves, printoutput] = options

    a = 0.1  # lower bracket
    c = 0.9999  # upper bracket
    b = 0.5 * (a + c)  # initial middle point

    lam_0 = 0.1
    lam_f = 1.05
    n = 100
    lams = numpy.linspace(lam_0, lam_f, n)[::-1]

    crit_strains = []

    if mode == 'hetero':
        print("{mode}, beta- = {bm:0.2f}, beta+ = {bp:0.2f}".format(mode=mode, bm=beta_m, bp=beta_p))
        determinant = determinant_H
    elif mode == 'serpentine':
        print("{mode}, beta = {bm:0.2f} = {bp:0.2f}".format(mode=mode, bm=beta_m, bp=beta_p))
        determinant = determinant_S
    elif mode == 'pinched':
        print("{mode}, beta = {bm:0.2f} = {bp:0.2f}".format(mode=mode, bm=beta_m, bp=beta_p))
        determinant = determinant_P

    for Lbar in wavelengths:

        Kbar = 2. * pi / Lbar

        [rootexists, a, c] = check_roots(determinant, lams, beta_m, beta_p, Kbar, plotroots)

        if findroots:
            [crit_strains, b] = find_roots(
                rootexists,
                determinant,
                crit_strains,
                a,
                b,
                c,
                beta_m,
                beta_p,
                Kbar,
                printoutput,
                tol
            )

    if plotindcurves and findroots:
        plt.figure()
        plt.plot(wavelengths, crit_strains, linestyle='-')
        plt.gca().axis([-20.0, 1000., 0.0, 1.0])
        plt.savefig('curves_{bm}_{bp}.png'.format(bm=beta_m, bp=beta_p))

    if findroots:
        return crit_strains


def find_minimum(wavelengths, crit_strains):
    """ Finds minimum critical strain and corresponding min_wavelength

    Parameters
    ----------
    wavelengths : list of floats
        list of wavelengths
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength

    Returns
    -------
    min_wavelength : float
        critical wavelength (corresponding to critical strain)
    min_strain : float
        minimum critical strain
    
    Notes
    -----
    If there is no true minimum, it returns the zero for the wavelength and the zero wavelength strain
    """

    index = 0

    if crit_strains[0] == min(crit_strains):  # we're done
        start = False
    else:
        start = True

    for i in range(1, len(wavelengths) - 1):
        if crit_strains[i] == 1.0 and start:
            index = i + 1
        elif crit_strains[i] == 1.0:
            index = index
        elif crit_strains[i] < crit_strains[index]:
            index = i
            start = False

    # remove no-root results
    strains_masked = numpy.ma.masked_greater(crit_strains, 0.999)
    if max(strains_masked) - min(strains_masked) < 0.001:
        index = 0

    # if very small crit_wavelength is dominating, check to see if infinity is equally favorable
    if index > 0.95 * len(wavelengths):
        if abs(crit_strains[0] - crit_strains[index]) < 0.001:
            index = 0

    min_wavelength = wavelengths[index]
    min_crit_strain = crit_strains[index]

    return min_crit_strain, min_wavelength


def find_threshold_values(i, j, crit_strains, wavelengths, thresh_strains, thresh_wavelengths, findroots):
    """ find the threshold (min and max) critical min_strains, and corresponding wavelengths and stiffness ratios

    Parameters
    ----------
    i, j: ints
        indices corresponding to relevant element of beta arrays 
    crit_strains : list of floats
        list of critical strains associated with each wavelength
    wavelengths : list of floats
        list of wavelengths
    thresh_strains : list of floats
        list of threshold strain values
    thresh_wavelengths : list of floats
        list of wavelengths associated with threshold strains
    find_roots : boolean
        whether or not to seek values of roots (set to False to only see root plots)

    Returns
    -------
    thresh_strains : array of floats
        updated array of threshold strain values
    thresh_wavelengths : array of floats
        updated array of wavelengths associated with threshold strains
    
    Notes
    -----
    Large wavelengths are set to the maximum (110)
    """

    if findroots:

        crit_strains = numpy.array(crit_strains)

        [thresh_strain, thresh_wavelength] = find_minimum(wavelengths, crit_strains)

        if thresh_wavelength < 100.:
            thresh_wavelengths[i][j] = thresh_wavelength
            thresh_strains[i][j] = thresh_strain

        else:
            thresh_wavelengths[i][j] = 110.
            thresh_strains[i][j] = thresh_strain

        return thresh_strains, thresh_wavelengths


def solve_and_write(filename, mode, bs_m, bs_p, wavelengths, options, tol=1.e-12, crit=False):
    """ Solve for critical and threshold values and write them to files

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
    wavelengths : list of floats
        list of wavelengths
    options : list of booleans
        findroots, plotroots, plotindcurves, printoutput options
    tol : float
        tolerance for Ridders' method; solution will be returned when the absolute value of the function is below the tolerance
    crit : boolean
        whether or not to print out all critical values in addition to threshold values    

    Returns
    -------
    None

    Notes
    -----
    In homogeneous cases (pinched and symmetric), beta_p and beta_m are checked;
    if they are not equal, the calculation is skipped

    In heterogeneous cases, beta_p and beta_m are checked; 
    if the opposite case has already been calculated, those values are copied
    """

    print(filename)

    [findroots, plotroots, plotindcurves, printoutput] = options

    if crit == True:
        crit_strain_file = open('results_{filename}_strains.txt'.format(filename=filename), 'w')

        with open('results_{filename}_wavelengths.txt'.format(filename=filename), 'w') as crit_wave_file:
            for i in range(len(wavelengths)):
                crit_wave_file.write('{wavelength:0.5f}\n'.format(wavelength=wavelengths[i]))

    thresh_wavelengths = numpy.zeros((len(bs_m), len(bs_p)))
    thresh_strains = numpy.zeros((len(bs_m), len(bs_p)))
    betas_m = numpy.zeros((len(bs_m), len(bs_p)))
    betas_p = numpy.zeros((len(bs_m), len(bs_p)))

    # build beta arrays
    for i in range(len(bs_m)):
        for j in range(len(bs_p)):
            betas_m[i][j] = bs_m[i]
            betas_p[i][j] = bs_p[j]

    # fill in threshold arrays
    for i in range(len(bs_m)):
        for j in range(len(bs_p)):
            beta_m = betas_m[i][j]
            beta_p = betas_p[i][j]

            # check for symmetry to see if this calculation can be avoided
            symmetric = False
            if beta_m in bs_p and beta_p in bs_m and beta_p < beta_m:
                symmetric = True
                j_sym = numpy.where(bs_p == beta_m)[0][0]
                i_sym = numpy.where(bs_m == beta_p)[0][0]

            # if homogenous, check to see if beta_m == beta_p   
            diagonal = True
            if mode in ['serpentine', 'pinched']:
                if i != j:
                    diagonal = False

            if diagonal:

                # if calculation has already been done, copy results
                if symmetric:
                    thresh_wavelengths[i][j] = thresh_wavelengths[i_sym][j_sym]
                    thresh_strains[i][j] = thresh_strains[i_sym][j_sym]

                # otherwise, calculate values
                else:
                    crit_strains = find_crit_values(
                        mode,
                        beta_m,
                        beta_p,
                        wavelengths,
                        options,
                        tol
                    )

                    [thresh_strains, thresh_wavelengths] = find_threshold_values(
                        i,
                        j,
                        crit_strains,
                        wavelengths,
                        thresh_strains,
                        thresh_wavelengths,
                        findroots
                    )

                if crit == True:
                    write_crit_values(filename, beta_m, beta_p, crit_strains)

    write_thresh_values(filename, mode, betas_m, betas_p, thresh_strains, thresh_wavelengths)


def write_thresh_values(filename, mode, betas_m, betas_p, crit_strains, crit_wavelengths):
    """ Write threshold values to files

    Parameters
    ----------
    filename : string
        name of results file
    mode : string
        type of instability - 'hetero', 'pinched', or 'serpentine'
    betas_m : array of floats
        stiffness ratios of lower matrix (matrix/layer) 
    betas_p : array of floats
        stiffness ratios of upper matrix (matrix/layer)     
    crit_wavelengths : list of floats
        list of wavelengths
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength

    Returns
    -------
    None

    Notes
    -----
    Results are only written for homogeneous cases when beta_m = beta_p
    """

    [m, n] = betas_m.shape

    with open('results_{filename}.txt'.format(filename=filename), 'w') as f:
        for i in range(m):
            for j in range(n):
                if mode == 'hetero' or i == j:
                    f.write(
                        '{beta_m:0.5f} {beta_p:0.5f} {strain:0.5f} {wavelength:0.5f} \n'.format(
                            beta_m=betas_m[i][j],
                            beta_p=betas_p[i][j],
                            strain=crit_strains[i][j],
                            wavelength=crit_wavelengths[i][j])
                    )


def write_crit_values(filename, beta_m, beta_p, crit_strains):
    """ Write threshold values to files

    Parameters
    ----------
    filename : string
        name of results file
    beta_m : float
        stiffness ratio of lower matrix (matrix/layer) 
    beta_p : float
        stiffness ratio of upper matrix (matrix/layer) 
    crit_strains : list of floats
        list of critical strain values corresponding to each wavelength

    Returns
    -------
    None
    """

    with open('results_{filename}_strains.txt'.format(filename=filename), 'a') as f:
        f.write('{beta_m:0.5f} {beta_p:0.5f} '.format(beta_m=beta_m, beta_p=beta_p))

        for k in range(len(crit_strains)):
            f.write('{strain:0.5f} '.format(strain=crit_strains[k]))

        f.write('\n')
