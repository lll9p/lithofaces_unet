import numpy as np
import scipy.ndimage as scind
"""https://github.com/CellProfiler/centrosome/blob/master/centrosome/filter.py"""

def poisson_equation(
    image, gradient=1, max_iter=100, convergence=0.01, percentile=95.0
):
    """Estimate the solution to the Poisson Equation
    The Poisson Equation is the solution to gradient(x) = h^2/4 and, in this
    context, we use a boundary condition where x is zero for background
    pixels. Also, we set h^2/4 = 1 to indicate that each pixel is a distance
    of 1 from its neighbors.
    The estimation exits after max_iter iterations or if the given percentile
    of foreground pixels differ by less than the convergence fraction
    from one pass to the next.
    Some ideas taken from Gorelick, "Shape representation and classification
    using the Poisson Equation", IEEE Transactions on Pattern Analysis and
    Machine Intelligence V28, # 12, 2006
    image - binary image with foreground as True
    gradient - the target gradient between 4-adjacent pixels
    max_iter - maximum # of iterations at a given level
    convergence - target fractional difference between values from previous
                  and next pass
    percentile - measure convergence at this percentile
    """
    # Evaluate the poisson equation with zero-padded boundaries
    pe = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    if image.shape[0] > 64 and image.shape[1] > 64:
        #
        # Sub-sample to get seed values
        #
        sub_image = image[::2, ::2]
        sub_pe = poisson_equation(
            sub_image,
            gradient=gradient * 2,
            max_iter=max_iter,
            convergence=convergence)
        coordinates = (
            np.mgrid[0: (sub_pe.shape[0] * 2),
                     0: (sub_pe.shape[1] * 2)].astype(float) / 2)
        pe[
            1: (sub_image.shape[0] * 2 + 1), 1: (sub_image.shape[1] * 2 + 1)
        ] = scind.map_coordinates(sub_pe, coordinates, order=1)
        pe[: image.shape[0], : image.shape[1]][~image] = 0
    else:
        pe[1:-1, 1:-1] = image
    #
    # evaluate only at i and j within the foreground
    #
    i, j = np.mgrid[0: pe.shape[0], 0: pe.shape[1]]
    mask = (i > 0) & (i < pe.shape[0] - 1) & (j > 0) & (j < pe.shape[1] - 1)
    mask[mask] = image[i[mask] - 1, j[mask] - 1]
    i = i[mask]
    j = j[mask]
    if len(i) == 0:
        return pe[1:-1, 1:-1]
    if len(i) == 1:
        # Just in case "percentile" can't work when unable to interpolate
        # between a single value... Isolated pixels have value = 1
        #
        pe[mask] = 1
        return pe[1:-1, 1:-1]

    for itr in range(max_iter):
        next_pe = (pe[i + 1, j] + pe[i - 1, j] + pe[i, j + 1] + pe[i, j - 1]) / 4 + 1
        difference = np.abs((pe[mask] - next_pe) / next_pe)
        pe[mask] = next_pe
        if np.percentile(difference, percentile) <= convergence:
            break
    return pe[1:-1, 1:-1]
