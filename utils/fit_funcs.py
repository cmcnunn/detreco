import numpy

def line(x, m, b):
    return m * x + b

def gauss(x, amp, mu, sigma):
    return amp * numpy.exp(-0.5 * ((x - mu) / sigma) ** 2)

def sine(x, amp, freq, phase, offset, m):
    return m*x + amp * numpy.sin(freq * x + phase) + offset