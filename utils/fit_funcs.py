import numpy
from scipy.special import erf

def line(x, m, b):
    return m * x + b

def gauss(x, amp, mu, sigma):
    return amp * numpy.exp(-0.5 * ((x - mu) / sigma) ** 2)

def sine(x, amp, freq, phase, offset, m):
    return m*x + amp * numpy.sin(freq * x + phase) + offset

def erf_edge(x, x0, sigma):
    """Smeared step-edge turn-on: rises through 0.5 at x0, over width sigma."""
    return 0.5 * (1 + erf((x - x0) / (numpy.sqrt(2) * sigma)))

def erf_disk(xy, x0, y0, r, sigma, eff0):
    """Efficiency footprint of a disk of radius r centered at (x0, y0),
    smeared by an isotropic Gaussian resolution sigma.

    This is the standard edge-response model for a hard circular boundary
    convolved with 2D Gaussian smearing: exact along the boundary, and an
    excellent approximation everywhere sigma << r.
    """
    x, y = xy
    rr = numpy.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return eff0 * 0.5 * (1 - erf((rr - r) / (numpy.sqrt(2) * sigma)))

def erf_box(xy, x0, y0, hx, hy, sigmax, sigmay, eff0):
    """Efficiency footprint of an axis-aligned rectangle (half-widths hx,
    hy, centered at x0, y0), as the product of two independent smeared
    step-edges, one per axis. hx/hy are fit independently rather than
    forced equal, so a real x/y tracker scale mismatch shows up as
    hx != hy instead of being averaged away.
    """
    x, y = xy
    fx = erf_edge(x, x0 - hx, sigmax) - erf_edge(x, x0 + hx, sigmax)
    fy = erf_edge(y, y0 - hy, sigmay) - erf_edge(y, y0 + hy, sigmay)
    return eff0 * fx * fy