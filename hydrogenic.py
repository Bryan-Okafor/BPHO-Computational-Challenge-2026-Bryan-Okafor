import numpy as np
from scipy.special import lpmv, factorial

# Physical constants
eps0 = 8.854e-12
hbar = 1.055e-34
me   = 9.109e-31
e    = 1.602e-19
u    = 1.661e-27          # atomic mass unit
a0   = 5.292e-11          # Bohr radius


def reducedMass(A):
    # The nucleus is not infinitely heavy, so the electron and nucleus both
    # orbit their common centre of mass. Using the reduced mass corrects for this.
    M = A * u
    return (me * M) / (me + M)


def atomRadius(Z, A):
    # Characteristic radius of a hydrogenic atom with Z protons
    mu = reducedMass(A)
    return (me * a0) / (mu * Z)


def orbitalEnergy(Z, A, n):
    # Energy of the nth level in eV
    mu = reducedMass(A)
    return -(mu * e**4 * Z**2) / (8 * eps0**2 * (2*np.pi*hbar)**2 * n**2) / e


def laguerreSum(x, n, l):
    # Associated Laguerre polynomial written out as its explicit finite sum
    total = np.zeros_like(x, dtype=float)
    for k in range(n - l):
        coeff = (factorial(l + n) * (-x)**k) / (
            factorial(2*l + 1 + k) * factorial(n - l - 1 - k) * factorial(k)
        )
        total = total + coeff
    return total


def radialPart(r, Z, A, n, l):
    # Radial part of the wavefunction, R(r,n,l)
    a = atomRadius(Z, A)
    x = (2 * r) / (a * n)

    normalisation = np.sqrt(factorial(n - l - 1) / (2 * n * factorial(n + l)))
    scale = (2 / (a * n))**1.5

    return normalisation * scale * x**l * np.exp(-x / 2) * laguerreSum(x, n, l)


def sphericalHarmonic(theta, phi, l, m):
    # Y_l^m using the associated Legendre function
    mAbs = abs(m)
    norm = np.sqrt(
        ((2*l + 1) / (4*np.pi)) * (factorial(l - mAbs) / factorial(l + mAbs))
    )
    legendre = lpmv(mAbs, l, np.cos(theta))
    return ((-1)**mAbs) * norm * legendre * np.exp(1j * mAbs * phi)


def angularPart(theta, phi, l, m):
    # Real-valued angular part built from combinations of spherical harmonics,
    # so the orbitals come out in the familiar real lobed shapes rather than complex
    if m == 0:
        return sphericalHarmonic(theta, phi, l, 0)
    elif m < 0:
        return (sphericalHarmonic(theta, phi, l, -m)
                - sphericalHarmonic(theta, phi, l, m))
    else:
        return (sphericalHarmonic(theta, phi, l, m)
                + sphericalHarmonic(theta, phi, l, -m))


def hydrogenicWavefunction(r, theta, phi, Z, A, n, l, m):
    # Full wavefunction is the product of the radial and angular parts
    if not (0 <= l <= n - 1 and abs(m) <= l):
        raise ValueError(f'Invalid quantum numbers: n={n}, l={l}, m={m}')

    return radialPart(r, Z, A, n, l) * angularPart(theta, phi, l, m)


def probabilityDensity(r, theta, phi, Z, A, n, l, m):
    # Born interpretation: |psi|^2 is the probability density
    psi = hydrogenicWavefunction(r, theta, phi, Z, A, n, l, m)
    return np.real(psi * np.conj(psi))


def orbitalLetter(l):
    return ['S', 'P', 'D', 'F', 'G', 'H'][l]
