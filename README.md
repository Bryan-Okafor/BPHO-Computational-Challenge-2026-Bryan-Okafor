# BPhO Computational Challenge 2026

Interactive quantum mechanics simulations, built for the British Physics
Olympiad Computational Challenge 2026.

Every page is driven by the same Python models used for the written tasks,
so the plots are generated live rather than being stored images.

Bryan Okafor, Littleover Community School

## Pages

- **Brownian motion** — temperature and coefficient of restitution drive a
  live collision simulation, showing how the large particle's drift responds
- **Black body radiation** — Planck spectrum with a temperature slider and
  Wien peak tracking, plus the Einstein model of molar heat capacity
- **Electron diffraction** — simulated phosphor screen where the rings move
  as the accelerating voltage changes, with the linearised check that
  recovers the graphite atomic spacings from the gradient
- **Quantum cryptography** — classical and quantum mismatch probabilities
  for polarised entangled photons, with both detector angles adjustable
- **Particle in a box** — energy levels and probability densities with an
  adjustable well width, plus a live check of the uncertainty principle
- **Compton scattering** — wavelength shift, recoil speed and recoil angle
  with an accompanying sketch of the collision geometry
- **Hydrogenic orbitals** — 2D slices and 3D stacked-slice visualisations
  of the probability density for any valid n, l, m

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```
