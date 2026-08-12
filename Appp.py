import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from hydrogenic import (probabilityDensity, orbitalEnergy,
                        orbitalLetter, atomRadius)

st.set_page_config(page_title='BPhO Computational Challenge 2026',
                   page_icon='atom', layout='wide')

# Physical constants used across the pages
h    = 6.626e-34
c    = 2.998e8
kB   = 1.381e-23
e    = 1.602e-19
me   = 9.109e-31
hbar = 1.055e-34
R    = 8.314


def darkFigure(width=9, height=5.5):
    # Every plot uses the same dark styling so the pages look consistent
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    return fig, ax


# ----------------------------------------------------------------------
# Task 3: Planck spectrum and Einstein heat capacity
# ----------------------------------------------------------------------
def planckPage():
    st.header('Black Body Radiation and Heat Capacity')

    left, right = st.columns([1, 2.4])

    with left:
        temperature = st.slider('Temperature / K', 1000, 10000, 5772, step=50)
        showWien = st.checkbox('Mark the Wien peak', value=True)
        showVisible = st.checkbox('Shade the visible band', value=True)
        compare = st.multiselect('Compare against',
                                 [3000, 4000, 5000, 6000, 7000],
                                 default=[4000, 6000])

        peak = 2.898e-3 / temperature
        st.metric('Peak wavelength', f'{peak*1e9:.0f} nm')

        # Integrating Planck over all wavelengths gives the Stefan-Boltzmann law,
        # so total power radiated climbs with the fourth power of temperature
        sigma = 5.670e-8
        st.metric('Total emitted power', f'{sigma * temperature**4:.3e} W/m²')

        if abs(temperature - 5772) < 60:
            st.caption('5772 K is the surface temperature of the Sun.')

    wavelengths = np.linspace(50e-9, 3000e-9, 1400)

    def planck(wavelength, T):
        # Dividing by 1e9 converts from per metre to per nanometre of wavelength
        exponent = (h * c) / (wavelength * kB * T)
        return (2*h*c**2 / wavelength**5) * (1/(np.exp(exponent) - 1)) * 1e-9

    with right:
        fig, ax = darkFigure()

        for T in sorted(compare):
            ax.plot(wavelengths*1e9, planck(wavelengths, T),
                    linewidth=1, alpha=0.45, color='#888888')
            ax.annotate(f'{T}K', xy=(2900, planck(2900e-9, T)),
                        color='#888888', fontsize=7)

        ax.plot(wavelengths*1e9, planck(wavelengths, temperature),
                color='#ff6b35', linewidth=2.2, label=f'T = {temperature} K')

        if showVisible:
            ax.axvspan(380, 700, color='cyan', alpha=0.10)

        if showWien:
            ax.plot(peak*1e9, planck(peak, temperature), 'o',
                    color='white', markersize=6)
            ax.annotate(f'{peak*1e9:.0f} nm',
                        xy=(peak*1e9, planck(peak, temperature)),
                        xytext=(peak*1e9 + 160, planck(peak, temperature)),
                        color='white', fontsize=8)

        ax.set_xlabel('Wavelength / nm')
        ax.set_ylabel('Spectral irradiance / W m⁻² nm⁻¹')
        ax.set_title('Planck black body spectrum')
        ax.set_xlim(0, 3000)
        ax.legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader('Einstein model of molar heat capacity')

    controls, plotArea = st.columns([1, 2.4])

    with controls:
        materials = {'Gold': 165, 'Copper': 343, 'Iron': 470,
                     'Aluminium': 394, 'Diamond': 1320}
        chosen = st.multiselect('Materials', list(materials),
                                default=['Gold', 'Copper', 'Iron'])
        maxTemp = st.slider('Temperature range / K', 200, 2000, 1000, step=100)
        st.caption('Each curve rises toward the classical Dulong–Petit value '
                   'of 3R, but falls away at low temperature where the '
                   'vibrational quanta can no longer be excited.')

    with plotArea:
        fig, ax = darkFigure()
        tempRange = np.linspace(1, maxTemp, 900)

        for name in chosen:
            einsteinTemp = materials[name]
            u = einsteinTemp / tempRange

            # At very low temperature u becomes large and exp(u) overflows once
            # squared. The heat capacity is vanishingly small there anyway, so
            # the exponent is capped and those points set straight to zero.
            safe = u < 300
            heatCapacity = np.zeros_like(tempRange)
            uSafe = u[safe]
            heatCapacity[safe] = (3*R * uSafe**2 * np.exp(uSafe)
                                  / (np.exp(uSafe) - 1)**2)

            ax.plot(tempRange, heatCapacity, linewidth=1.6,
                    label=f'{name}  (T_E = {einsteinTemp} K)')

        ax.axhline(3*R, color='#888888', linestyle='--', linewidth=1)
        ax.annotate(f'3R = {3*R:.1f}', xy=(maxTemp*0.72, 3*R + 0.6),
                    color='#888888', fontsize=8)
        ax.set_xlabel('Temperature / K')
        ax.set_ylabel('Molar heat capacity / J mol⁻¹ K⁻¹')
        ax.set_ylim(0, 30)
        ax.legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------
# Task 6: Electron diffraction
# ----------------------------------------------------------------------
def diffractionPage():
    st.header('Electron Diffraction Through Graphite')

    tubeRadius = 65e-3
    spacings = {'d₁ = 0.123 nm': 0.123e-9, 'd₂ = 0.213 nm': 0.213e-9}
    colours = {'d₁ = 0.123 nm': '#00e5ff', 'd₂ = 0.213 nm': '#ff4fd8'}

    def ringRadius(V, d):
        deBroglie = h / np.sqrt(2 * me * e * V)
        sinHalfPhi = np.clip(deBroglie / (2*d), 0, 0.999)
        phi = 2 * np.arcsin(sinHalfPhi)
        return tubeRadius * np.sin(2*phi)

    left, right = st.columns([1, 2.4])

    with left:
        voltage = st.slider('Accelerating voltage / kV', 1.0, 5.0, 2.5, 0.05)
        V = voltage * 1000

        deBroglie = h / np.sqrt(2 * me * e * V)
        st.metric('de Broglie wavelength', f'{deBroglie*1e12:.2f} pm')

        for label, d in spacings.items():
            st.metric(f'Ring radius, {label}',
                      f'{ringRadius(V, d)*1000:.1f} mm')

        st.caption('Raising the voltage shortens the electron wavelength, '
                   'which narrows the Bragg angle and pulls the rings inward.')

    with right:
        screen, graphs = st.columns([1, 1.25])

        with screen:
            fig, ax = darkFigure(5, 5)
            ax.set_facecolor('black')

            # Central undiffracted beam
            for glow, alpha in [(9, 0.20), (6, 0.35), (3.5, 0.9)]:
                ax.add_patch(plt.Circle((0, 0), glow, color='#39ff14',
                                        alpha=alpha, zorder=3))

            for label, d in spacings.items():
                radius = ringRadius(V, d) * 1000
                ax.add_patch(plt.Circle((0, 0), radius, fill=False,
                                        color=colours[label],
                                        linewidth=2.5, alpha=0.9))
                ax.add_patch(plt.Circle((0, 0), radius, fill=False,
                                        color=colours[label],
                                        linewidth=7, alpha=0.18))

            ax.set_xlim(-65, 65)
            ax.set_ylim(-65, 65)
            ax.set_aspect('equal')
            ax.set_title(f'Phosphor screen at {voltage:.2f} kV')
            ax.set_xlabel('mm')
            st.pyplot(fig)
            plt.close(fig)

        with graphs:
            voltages = np.linspace(1000, 5000, 500)

            fig, ax = darkFigure(5.4, 2.5)
            for label, d in spacings.items():
                ax.plot(voltages/1000, ringRadius(voltages, d)*1000,
                        color=colours[label], linewidth=1.6, label=label)
            ax.axvline(voltage, color='white', linestyle=':', linewidth=1)
            ax.set_xlabel('Voltage / kV')
            ax.set_ylabel('Ring radius / mm')
            ax.legend(facecolor='#0e1117', labelcolor='white', fontsize=7)
            st.pyplot(fig)
            plt.close(fig)

            # Plotting against 1/sqrt(V) linearises the relationship, and the
            # gradient of each line recovers the atomic spacing that produced it
            fig, ax = darkFigure(5.4, 2.5)
            invSqrtV = 1/np.sqrt(voltages)
            for label, d in spacings.items():
                sinHalfPhi = h/np.sqrt(2*me*e*voltages) / (2*d)
                ax.plot(invSqrtV*1000, sinHalfPhi, color=colours[label],
                        linewidth=1.6)
                gradient = np.polyfit(invSqrtV, sinHalfPhi, 1)[0]
                recovered = h / (2*gradient*np.sqrt(2*me*e)) * 1e9
                ax.annotate(f'recovers d = {recovered:.3f} nm',
                            xy=(invSqrtV[len(invSqrtV)//2]*1000,
                                sinHalfPhi[len(sinHalfPhi)//2]),
                            color=colours[label], fontsize=7)
            ax.axvline(1/np.sqrt(V)*1000, color='white',
                       linestyle=':', linewidth=1)
            ax.set_xlabel('1/√V   (×10⁻³ V⁻¹ᐟ²)')
            ax.set_ylabel('sin(φ/2)')
            ax.set_title('Linearised check', fontsize=9)
            st.pyplot(fig)
            plt.close(fig)


# ----------------------------------------------------------------------
# Task 8: Quantum cryptography
# ----------------------------------------------------------------------
def cryptographyPage():
    st.header('Polarised Entangled Photons: Classical against Quantum')

    def classicalMismatch(theta, phi):
        return 1 - np.cos(theta)**2*np.cos(phi)**2 - np.sin(theta)**2*np.sin(phi)**2

    def quantumMismatch(theta, phi):
        return np.sin(phi - theta)**2

    left, right = st.columns([1, 2.4])

    with left:
        thetaDeg = st.slider('Detector A angle θ / °', -180, 180, -30)
        phiDeg = st.slider('Detector B angle φ / °', -180, 180, 30)

        theta = np.radians(thetaDeg)
        phi = np.radians(phiDeg)

        classical = classicalMismatch(theta, phi)
        quantum = quantumMismatch(theta, phi)

        st.metric('Classical mismatch', f'{classical:.4f}')
        st.metric('Quantum mismatch', f'{quantum:.4f}')
        st.metric('Difference', f'{quantum - classical:+.4f}')

        if abs(thetaDeg + 30) < 1 and abs(phiDeg - 30) < 1:
            st.success('This is the worked case: 3/8 classical, 6/8 quantum.')

        st.caption('The two models agree at some angles and diverge at others. '
                   'Where they diverge, experiment sides with the quantum '
                   'prediction, which is what rules out local hidden variables.')

    with right:
        difference = np.linspace(-np.pi, np.pi, 900)

        fig, ax = darkFigure(9, 5)
        ax.plot(np.degrees(difference),
                classicalMismatch(theta, theta + difference),
                color='#00e5ff', linewidth=2, label='Classical')
        ax.plot(np.degrees(difference),
                quantumMismatch(theta, theta + difference),
                color='#ff4fd8', linewidth=2, label='Quantum')

        ax.axvline(phiDeg - thetaDeg, color='white', linestyle=':', linewidth=1)
        ax.plot(phiDeg - thetaDeg, classical, 'o', color='#00e5ff', markersize=9)
        ax.plot(phiDeg - thetaDeg, quantum, 'o', color='#ff4fd8', markersize=9)

        ax.set_xlabel('φ − θ   / degrees')
        ax.set_ylabel('Probability of mismatch')
        ax.set_ylim(-0.02, 1.02)
        ax.legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------
# Task 10: Hydrogenic orbitals
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def orbitalExtent(Z, A, n, l, m, threshold):
    # Orbital size grows roughly with n squared but depends on l and m too,
    # so a coarse probe grid sizes the axes to the orbital being drawn
    trial = 2.5 * n**2 * atomRadius(Z, A)

    for _ in range(14):
        probe = np.linspace(-trial, trial, 34)
        X, Y, ZZ = np.meshgrid(probe, probe, probe, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + ZZ**2)
        r[r == 0] = 1e-20
        theta = np.arccos(np.clip(ZZ/r, -1, 1))
        phi = np.arctan2(Y, X)

        density = probabilityDensity(r, theta, phi, Z, A, n, l, m)
        density = density / density.max()
        keep = density > threshold

        if keep.sum() == 0:
            trial *= 1.4
            continue

        reach = max(abs(X[keep]).max(), abs(Y[keep]).max(), abs(ZZ[keep]).max())
        if reach < 0.72 * trial:
            return reach * 1.3
        trial *= 1.35

    return trial


@st.cache_data(show_spinner=False)
def sliceData(Z, A, n, l, m, extent, resolution):
    axis = np.linspace(-extent, extent, resolution)
    X, ZZ = np.meshgrid(axis, axis, indexing='ij')
    r = np.sqrt(X**2 + ZZ**2)
    r[r == 0] = 1e-20
    theta = np.arccos(np.clip(ZZ/r, -1, 1))

    # In the y = 0 plane arctan2 cannot be used, so phi is set directly:
    # zero on the positive x side and pi on the negative x side
    phi = np.where(X >= 0, 0.0, np.pi)

    density = probabilityDensity(r, theta, phi, Z, A, n, l, m)
    return X, ZZ, density / density.max()


@st.cache_data(show_spinner=False)
def volumeData(Z, A, n, l, m, extent, threshold, sliceCount, resolution):
    probe = np.linspace(-extent, extent, 44)
    X, Y, ZZ = np.meshgrid(probe, probe, probe, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + ZZ**2)
    r[r == 0] = 1e-20
    theta = np.arccos(np.clip(ZZ/r, -1, 1))
    phi = np.arctan2(Y, X)
    peak = probabilityDensity(r, theta, phi, Z, A, n, l, m).max()

    planeAxis = np.linspace(-extent, extent, resolution)
    X2, Y2 = np.meshgrid(planeAxis, planeAxis, indexing='ij')

    xs, ys, zs, cs = [], [], [], []
    for zValue in np.linspace(-extent, extent, sliceCount):
        r = np.sqrt(X2**2 + Y2**2 + zValue**2)
        r[r == 0] = 1e-20
        theta = np.arccos(np.clip(zValue/r, -1, 1))
        phi = np.arctan2(Y2, X2)

        density = probabilityDensity(r, theta, phi, Z, A, n, l, m) / peak
        keep = density > threshold
        if keep.sum() == 0:
            continue

        xs.append(X2[keep])
        ys.append(Y2[keep])
        zs.append(np.full(keep.sum(), zValue))
        cs.append(density[keep])

    if not xs:
        return None
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(zs), np.concatenate(cs))


def orbitalPage():
    st.header('Hydrogenic Orbitals')

    left, right = st.columns([1, 2.4])

    with left:
        Z = st.slider('Protons Z', 1, 10, 1)
        A = st.slider('Mass number A', 1, 20, 1)

        n = st.slider('Principal quantum number n', 1, 6, 3)

        # l is restricted to below n, and m cannot exceed l in magnitude.
        # When only one value is allowed there is nothing to slide, so the
        # value is shown as fixed text instead of an unusable slider.
        if n == 1:
            l = 0
            st.caption('l is fixed at 0, since l must be less than n.')
        else:
            l = st.slider('Orbital quantum number l', 0, n-1, min(2, n-1))

        if l == 0:
            m = 0
            st.caption('m is fixed at 0, since m cannot exceed l.')
        else:
            m = st.slider('Magnetic quantum number m', 0, l, 0)

        view = st.radio('View', ['2D slice', '3D stacked slices'])
        threshold = st.slider('Density cut-off', 0.05, 0.50, 0.15, 0.01)

        st.metric('Orbital', f'{n}{orbitalLetter(l)}, m = {m}')
        st.metric('Energy', f'{orbitalEnergy(Z, A, n):.4f} eV')
        st.metric('Atom radius scale', f'{atomRadius(Z, A)*1e12:.2f} pm')

        st.caption('Density below the cut-off is discarded, so what remains '
                   'is the region the electron actually occupies rather than '
                   'a solid block.')

    with right:
        extent = orbitalExtent(Z, A, n, l, m, threshold)

        if view == '2D slice':
            X, ZZ, density = sliceData(Z, A, n, l, m, extent, 320)

            fig, ax = darkFigure(7.5, 6.5)
            image = ax.contourf(X*1e10, ZZ*1e10, density,
                                levels=60, cmap='jet', vmin=0, vmax=1)
            ax.set_xlabel('x / Å')
            ax.set_ylabel('z / Å')
            ax.set_aspect('equal')
            ax.set_title(f'y = 0 plane   Z={Z}   {n}{orbitalLetter(l)}   '
                         f'l={l}   m={m}')
            bar = fig.colorbar(image, ax=ax, fraction=0.046)
            bar.set_label('|ψ|² / max|ψ|²', color='white')
            bar.ax.tick_params(colors='white', labelsize=7)
            st.pyplot(fig)
            plt.close(fig)

        else:
            data = volumeData(Z, A, n, l, m, extent, threshold, 24, 110)

            if data is None:
                st.warning('Nothing exceeds the cut-off. Try lowering it.')
            else:
                xs, ys, zs, cs = data

                fig = plt.figure(figsize=(7.5, 6.5))
                fig.patch.set_facecolor('#0e1117')
                ax = fig.add_subplot(111, projection='3d')
                ax.set_facecolor('#0e1117')

                scatter = ax.scatter(xs*1e10, ys*1e10, zs*1e10, c=cs,
                                     cmap='jet', s=4, alpha=0.16,
                                     vmin=threshold, vmax=1,
                                     edgecolors='none', depthshade=False)

                limit = extent*1e10
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.set_zlim(-limit, limit)
                ax.set_box_aspect([1, 1, 1])
                ax.set_xlabel('x / Å', color='white')
                ax.set_ylabel('y / Å', color='white')
                ax.set_zlabel('z / Å', color='white')
                ax.tick_params(colors='white', labelsize=7)
                ax.set_title(f'Z={Z}  {n}{orbitalLetter(l)}  m={m}',
                             color='white')

                bar = fig.colorbar(scatter, ax=ax, shrink=0.65)
                bar.set_label('|ψ|² / max|ψ|²', color='white')
                bar.ax.tick_params(colors='white', labelsize=7)
                bar.solids.set_alpha(1)

                st.pyplot(fig)
                plt.close(fig)

                st.caption('The volume is drawn as a stack of separate planes '
                           'rather than a filled grid. Gaps between the planes '
                           'let you see through the outer surface to the '
                           'high density core, which a solid volume hides.')


# ----------------------------------------------------------------------
# Task 2: Brownian motion
# ----------------------------------------------------------------------
def brownianPage():
    st.header('Brownian Motion')

    left, right = st.columns([1, 2.4])

    with left:
        temperature = st.slider('Temperature / K', 50, 1200, 300, step=25)
        restitution = st.slider('Coefficient of restitution C', 0.0, 1.0, 1.0, 0.05)
        particleCount = st.slider('Number of small particles', 50, 600, 300, step=50)
        massRatio = st.slider('Mass ratio M/m', 10, 500, 100, step=10)
        frames = st.slider('Simulation length / frames', 200, 3000, 1200, step=200)
        seed = st.number_input('Random seed', 0, 9999, 0)

        st.caption('The large particle starts at rest. Its motion comes '
                   'entirely from the imbalance of random collisions, so a '
                   'higher temperature drives the small particles faster and '
                   'the drift grows.')

    smallMass = 1e-20
    bigMass = massRatio * smallMass
    smallRadius, bigRadius = 1.0, 8.0
    boxSize = 100.0
    knudsen = 15

    # Speed comes from the Maxwell-Boltzmann result, so temperature feeds
    # straight through to how hard the large particle gets hit
    speed = np.sqrt(3 * kB * temperature / smallMass) / 1000
    dt = 0.1 * knudsen * smallRadius / speed

    @st.cache_data(show_spinner='Running the collision simulation...')
    def runSimulation(temperature, restitution, particleCount, massRatio,
                      frames, seed):
        rng = np.random.default_rng(seed)

        smallX = rng.uniform(-boxSize, boxSize, particleCount)
        smallY = rng.uniform(-boxSize, boxSize, particleCount)
        angles = rng.uniform(0, 2*np.pi, particleCount)
        smallVX = speed * np.cos(angles)
        smallVY = speed * np.sin(angles)
        stepsSince = np.zeros(particleCount)

        bigX = bigY = 0.0
        bigVX = bigVY = 0.0
        pathX, pathY, collisions = [0.0], [0.0], 0

        for _ in range(frames):
            smallX += smallVX * dt
            smallY += smallVY * dt
            stepsSince += 1

            # A real molecule travels in a straight line between collisions
            # with other molecules, so directions are only rerolled after
            # a set number of steps rather than every frame
            due = stepsSince >= knudsen
            if due.any():
                fresh = rng.uniform(0, 2*np.pi, due.sum())
                smallVX[due] = speed * np.cos(fresh)
                smallVY[due] = speed * np.sin(fresh)
                stepsSince[due] = 0

            smallX = np.where(smallX > boxSize, -boxSize, smallX)
            smallX = np.where(smallX < -boxSize, boxSize, smallX)
            smallY = np.where(smallY > boxSize, -boxSize, smallY)
            smallY = np.where(smallY < -boxSize, boxSize, smallY)

            dx = smallX - bigX
            dy = smallY - bigY
            distances = np.sqrt(dx**2 + dy**2)

            # A fast particle can cross the large one entirely between frames,
            # so any overlap is pushed back out to the surface before the
            # collision is resolved
            overlapping = distances < (smallRadius + bigRadius)
            if overlapping.any():
                contact = smallRadius + bigRadius
                smallX[overlapping] = bigX + contact*dx[overlapping]/distances[overlapping]
                smallY[overlapping] = bigY + contact*dy[overlapping]/distances[overlapping]
                dx = smallX - bigX
                dy = smallY - bigY
                distances = np.sqrt(dx**2 + dy**2)

            hits = np.where(distances < (smallRadius + bigRadius))[0]
            collisions += len(hits)

            for i in hits:
                # Momentum is only exchanged along the line joining the two
                # centres, so both velocities are projected onto that normal
                normalX = (bigX - smallX[i]) / distances[i]
                normalY = (bigY - smallY[i]) / distances[i]

                u1 = smallVX[i]*normalX + smallVY[i]*normalY
                u2 = bigVX*normalX + bigVY*normalY

                # Shifting into the zero momentum frame reduces the collision
                # to a simple reversal scaled by the restitution coefficient
                V = (smallMass*u1 + bigMass*u2) / (smallMass + bigMass)
                v1 = restitution*(V - u1) + V
                v2 = restitution*(V - u2) + V

                smallVX[i] += (v1 - u1) * normalX
                smallVY[i] += (v1 - u1) * normalY
                bigVX += (v2 - u2) * normalX
                bigVY += (v2 - u2) * normalY

                fresh = rng.uniform(0, 2*np.pi)
                smallVX[i] = speed * np.cos(fresh)
                smallVY[i] = speed * np.sin(fresh)
                stepsSince[i] = 0

            bigX += bigVX * dt
            bigY += bigVY * dt
            pathX.append(bigX)
            pathY.append(bigY)

        return (np.array(pathX), np.array(pathY), smallX, smallY,
                collisions, np.hypot(bigVX, bigVY))

    pathX, pathY, smallX, smallY, collisions, finalSpeed = runSimulation(
        temperature, restitution, particleCount, massRatio, frames, seed)

    drift = np.hypot(pathX[-1], pathY[-1])

    with left:
        st.metric('Net drift from start', f'{drift:.1f} nm')
        st.metric('Collisions resolved', f'{collisions:,}')
        st.metric('Molecular speed', f'{speed*1000:.0f} m/s')

    with right:
        plot, stats = st.columns([1.15, 1])

        with plot:
            fig, ax = darkFigure(6, 6)
            ax.set_facecolor('black')
            ax.plot(smallX, smallY, '.', color='#4a9eff', markersize=1.5, alpha=0.6)
            ax.plot(pathX, pathY, '-', color='#ff4444', linewidth=0.8, alpha=0.9)
            ax.add_patch(plt.Circle((pathX[-1], pathY[-1]), bigRadius,
                                    color='#ff2222', alpha=0.85, zorder=5))
            ax.plot(0, 0, '*', color='#39ff14', markersize=15, zorder=6)
            ax.set_xlim(-boxSize, boxSize)
            ax.set_ylim(-boxSize, boxSize)
            ax.set_aspect('equal')
            ax.set_xlabel('x / nm')
            ax.set_ylabel('y / nm')
            ax.set_title(f'T = {temperature} K,  C = {restitution:.2f}')
            st.pyplot(fig)
            plt.close(fig)

        with stats:
            # Averaging the squared displacement over many start points gives
            # a much cleaner trend than following a single trajectory
            steps = np.arange(1, len(pathX))
            displacement = np.hypot(pathX - pathX[0], pathY - pathY[0])[1:]

            fig, ax = darkFigure(5.4, 2.7)
            ax.plot(steps, displacement, color='#ff6b35', linewidth=0.8)
            ax.plot(steps, displacement[-1]*np.sqrt(steps/steps[-1]),
                    '--', color='#888888', linewidth=1.2,
                    label='square root growth')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Distance from start / nm')
            ax.legend(facecolor='#0e1117', labelcolor='white', fontsize=7)
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = darkFigure(5.4, 2.7)
            ax.plot(pathX, pathY, color='#ff4444', linewidth=0.7)
            ax.plot(0, 0, '*', color='#39ff14', markersize=12)
            ax.plot(pathX[-1], pathY[-1], 'o', color='#ff2222', markersize=7)
            ax.set_aspect('equal')
            ax.set_xlabel('x / nm')
            ax.set_ylabel('y / nm')
            ax.set_title('Path of the large particle', fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

    st.caption('Lowering C below 1 makes the collisions inelastic, so kinetic '
               'energy is lost on every impact and the large particle is '
               'pushed around less for the same temperature.')


# ----------------------------------------------------------------------
# Task 7: Particle in a box
# ----------------------------------------------------------------------
def boxPage():
    st.header('Particle in an Infinite Potential Well')

    left, right = st.columns([1, 2.4])

    with left:
        boxWidth = st.slider('Box width / Angstrom', 0.1, 5.0, 0.53, 0.01)
        maxLevel = st.slider('Highest level shown', 1, 8, 3)
        showAll = st.checkbox('Overlay all levels', value=True)
        showWave = st.checkbox('Show the wavefunction as well', value=False)

        L = boxWidth * 1e-10

        def levelEnergy(n):
            return (n**2 * np.pi**2 * hbar**2) / (2 * me * L**2 * e)

        st.metric(f'Ground state energy', f'{levelEnergy(1):.2f} eV')
        st.metric(f'Level {maxLevel} energy', f'{levelEnergy(maxLevel):.2f} eV')

        # Position and momentum spreads follow from the standard integrals,
        # and their product is what the uncertainty principle constrains
        n = maxLevel
        deltaX = L * np.sqrt(1/12 - 1/(2*n**2*np.pi**2))
        deltaP = n * np.pi * hbar / L
        product = deltaX * deltaP / hbar

        st.metric(f'Delta x Delta p at n={n}', f'{product:.4f} hbar')
        if product >= 0.5:
            st.success(f'Above the limit of 0.5 hbar, as required.')
        else:
            st.error('Below the limit, which should not happen.')

        st.caption('Narrowing the box confines the particle more tightly, '
                   'which raises every energy level as one over width squared.')

    x = np.linspace(0, L, 900)

    with right:
        fig, ax = darkFigure(9, 5.5)

        colours = ['#00e5ff', '#39ff14', '#ff4fd8', '#ff6b35',
                   '#ffd700', '#ff4444', '#9d4edd', '#4a9eff']

        levels = range(1, maxLevel+1) if showAll else [maxLevel]

        for n in levels:
            psi = np.sqrt(2/L) * np.sin(n*np.pi*x/L)
            curve = psi if showWave else psi**2
            ax.plot(x*1e10, curve, color=colours[(n-1) % 8], linewidth=1.8,
                    label=f'n={n}   E={levelEnergy(n):.1f} eV')

        if showWave:
            ax.axhline(0, color='#555555', linewidth=0.8)
            ax.set_ylabel('Wavefunction psi')
        else:
            ax.set_ylabel('Probability density |psi|^2')

        ax.set_xlabel('Position / Angstrom')
        ax.set_title(f'Box width {boxWidth:.2f} Angstrom')
        ax.legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        energyPlot, uncertaintyPlot = st.columns(2)

        with energyPlot:
            fig, ax = darkFigure(4.6, 3)
            levelNumbers = np.arange(1, 9)
            ax.plot(levelNumbers, [levelEnergy(k) for k in levelNumbers],
                    'o-', color='#00e5ff', linewidth=1.4, markersize=5)
            ax.plot(maxLevel, levelEnergy(maxLevel), 'o',
                    color='#ff4444', markersize=10)
            ax.set_xlabel('Quantum number n')
            ax.set_ylabel('Energy / eV')
            ax.set_title('Energy scales as n squared', fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

        with uncertaintyPlot:
            fig, ax = darkFigure(4.6, 3)
            levelNumbers = np.arange(1, 21)
            products = [(L*np.sqrt(1/12 - 1/(2*k**2*np.pi**2))) *
                        (k*np.pi*hbar/L) / hbar for k in levelNumbers]
            ax.plot(levelNumbers, products, 'o-', color='#39ff14',
                    linewidth=1.4, markersize=4)
            ax.axhline(0.5, color='#ff4444', linestyle='--', linewidth=1.2)
            ax.annotate('limit 0.5', xy=(13, 0.62), color='#ff4444', fontsize=8)
            ax.set_xlabel('Quantum number n')
            ax.set_ylabel('Delta x Delta p  / hbar')
            ax.set_title('Uncertainty principle holds for all n', fontsize=9)
            st.pyplot(fig)
            plt.close(fig)


# ----------------------------------------------------------------------
# Task 9: Compton scattering
# ----------------------------------------------------------------------
def comptonPage():
    st.header('Compton Scattering')

    comptonWavelength = h / (me * c)
    theta = np.linspace(1e-6, np.pi, 900)

    left, right = st.columns([1, 2.4])

    with left:
        photonEnergy = st.slider('Incident photon energy / keV',
                                 10, 2000, 500, step=10)
        angleDeg = st.slider('Scattering angle / degrees', 0, 180, 90)

        E0 = photonEnergy * 1000 * e
        lambda0 = h * c / E0
        angle = np.radians(angleDeg)

        shift = comptonWavelength * (1 - np.cos(angle))
        lambda1 = lambda0 + shift
        electronEnergy = E0 - h*c/lambda1
        gamma = 1 + electronEnergy / (me*c**2)
        recoilSpeed = c * np.sqrt(1 - 1/gamma**2)

        st.metric('Incident wavelength', f'{lambda0*1e12:.3f} pm')
        st.metric('Wavelength shift', f'{shift*1e12:.3f} pm')
        st.metric('Energy given to electron', f'{electronEnergy/e/1000:.1f} keV')
        st.metric('Electron recoil speed', f'{recoilSpeed/c:.4f} c')

        st.caption('The wavelength shift depends only on the scattering angle, '
                   'never on the incident energy. What does change with energy '
                   'is how large that fixed shift is relative to the original '
                   'wavelength, which is why hard photons lose a far greater '
                   'fraction of their energy.')

    with right:
        shiftAll = comptonWavelength * (1 - np.cos(theta))
        lambda1All = lambda0 + shiftAll
        electronAll = E0 - h*c/lambda1All
        gammaAll = 1 + electronAll/(me*c**2)
        speedAll = c*np.sqrt(1 - 1/gammaAll**2)
        recoilAngle = np.arctan(np.sin(theta) /
                                (1 + (h/(me*c*lambda0))*(1-np.cos(theta)) - np.cos(theta)))

        top = st.columns(3)

        with top[0]:
            fig, ax = darkFigure(4.2, 3.4)
            for E in [50, 200, 1000]:
                l0 = h*c/(E*1000*e)
                ax.plot(np.degrees(theta), shiftAll/l0, linewidth=1,
                        alpha=0.4, color='#888888')
            ax.plot(np.degrees(theta), shiftAll/lambda0,
                    color='#00e5ff', linewidth=2)
            ax.plot(angleDeg, shift/lambda0, 'o', color='#ff4444', markersize=8)
            ax.set_xlabel('Scattering angle / deg')
            ax.set_ylabel('Fractional shift')
            ax.set_title('Wavelength shift', fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

        with top[1]:
            fig, ax = darkFigure(4.2, 3.4)
            ax.plot(np.degrees(theta), speedAll/c, color='#39ff14', linewidth=2)
            ax.plot(angleDeg, recoilSpeed/c, 'o', color='#ff4444', markersize=8)
            ax.set_xlabel('Scattering angle / deg')
            ax.set_ylabel('Recoil speed / c')
            ax.set_ylim(0, 1)
            ax.set_title('Electron recoil speed', fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

        with top[2]:
            fig, ax = darkFigure(4.2, 3.4)
            ax.plot(np.degrees(theta), np.degrees(recoilAngle),
                    color='#ff4fd8', linewidth=2)
            currentRecoil = np.degrees(np.arctan(
                np.sin(angle) /
                (1 + (h/(me*c*lambda0))*(1-np.cos(angle)) - np.cos(angle))))
            ax.plot(angleDeg, currentRecoil, 'o', color='#ff4444', markersize=8)
            ax.set_xlabel('Photon scattering angle / deg')
            ax.set_ylabel('Electron recoil angle / deg')
            ax.set_title('Recoil direction', fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

        # A simple sketch of the geometry, which makes the angles concrete
        fig, ax = darkFigure(9, 3)
        ax.arrow(-1, 0, 0.85, 0, head_width=0.06, color='#ffd700', linewidth=2)
        ax.text(-0.6, 0.1, f'{photonEnergy} keV', color='#ffd700', fontsize=9)
        ax.plot(0, 0, 'o', color='#4a9eff', markersize=11)
        ax.text(0, -0.16, 'electron at rest', color='#4a9eff',
                fontsize=8, ha='center')
        ax.arrow(0, 0, 0.9*np.cos(angle), 0.9*np.sin(angle),
                 head_width=0.06, color='#ff6b35', linewidth=2)
        ax.text(0.95*np.cos(angle), 0.95*np.sin(angle),
                f'scattered  {angleDeg}deg', color='#ff6b35', fontsize=8)
        recoilRad = np.radians(currentRecoil)
        ax.arrow(0, 0, 0.7*np.cos(-recoilRad), 0.7*np.sin(-recoilRad),
                 head_width=0.06, color='#39ff14', linewidth=2)
        ax.text(0.75*np.cos(-recoilRad), 0.75*np.sin(-recoilRad),
                f'recoil  {currentRecoil:.0f}deg', color='#39ff14', fontsize=8)
        ax.set_xlim(-1.2, 1.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------
# Task 4: Photoelectric effect
# ----------------------------------------------------------------------
def photoelectricPage():
    st.header('The Photoelectric Effect')

    workFunctions = {
        'Sodium (Na)':    2.4,
        'Silver (Ag)':    4.3,
        'Aluminium (Al)': 4.3,
        'Lead (Pb)':      4.3,
        'Tin (Sn)':       4.4,
        'Tungsten (W)':   4.5,
        'Nickel (Ni)':    4.6,
        'Copper (Cu)':    4.7,
        'Gold (Au)':      5.1,
    }

    left, right = st.columns([1, 2.4])

    with left:
        chosen = st.multiselect('Metals', list(workFunctions),
                                default=['Sodium (Na)', 'Silver (Ag)',
                                         'Copper (Cu)', 'Gold (Au)'])
        frequency = st.slider('Incident frequency / PHz', 0.0, 2.0, 1.5, 0.01)
        intensity = st.slider('Beam power / microwatt', 0.1, 10.0, 1.0, 0.1)

        f = frequency * 1e15
        photonEnergy = h * f

        st.metric('Photon energy',
                  f'{photonEnergy/e:.3f} eV' if f > 0 else '0 eV')
        if f > 0:
            st.metric('Wavelength', f'{c/f*1e9:.1f} nm')

        st.caption('Raising the intensity sends more photons per second, '
                   'which raises the current but never the energy of the '
                   'individual electrons. Only frequency does that.')

    with right:
        frequencies = np.linspace(0, 2e15, 1600)

        fig, ax = darkFigure(9, 5)
        palette = ['#00e5ff', '#39ff14', '#ff4fd8', '#ff6b35',
                   '#ffd700', '#ff4444', '#9d4edd', '#4a9eff', '#ffffff']

        for index, metal in enumerate(chosen):
            phi = workFunctions[metal]
            threshold = (phi * e) / h

            # Below the threshold no electrons leave the metal at all, so
            # there is no stopping voltage to plot rather than a negative one
            stopping = np.where(frequencies > threshold,
                                (h*frequencies - phi*e) / e, np.nan)
            ax.plot(frequencies*1e-15, stopping,
                    color=palette[index % 9], linewidth=1.6,
                    label=f'{metal}   phi={phi} eV')
            ax.plot(threshold*1e-15, 0, 'o',
                    color=palette[index % 9], markersize=6)

        ax.axvline(frequency, color='white', linestyle=':', linewidth=1)
        ax.axhline(0, color='#666666', linewidth=0.8)
        ax.set_xlabel('Frequency / PHz')
        ax.set_ylabel('Stopping voltage / V')
        ax.set_xlim(0, 2)
        ax.set_ylim(-1, 5)
        ax.legend(facecolor='#0e1117', labelcolor='white', fontsize=7,
                  loc='upper left')
        ax.set_title('Every line has gradient h/e, so only the intercept '
                     'depends on the metal', fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader('Threshold data')

    table, notes = st.columns([1.5, 1])

    with table:
        rows = []
        for metal, phi in workFunctions.items():
            threshold = (phi * e) / h
            emitting = (frequency*1e15) > threshold
            current = 0.0
            if emitting:
                # One electron per absorbed photon sets an upper bound
                # on the current the beam can produce
                current = (intensity*1e-6 / photonEnergy) * e * 1e6

            rows.append({
                'Metal': metal,
                'Work function / eV': f'{phi:.1f}',
                'Threshold / PHz': f'{threshold*1e-15:.4f}',
                'Max wavelength / nm': f'{h*c/(phi*e)*1e9:.0f}',
                'Emitting now': 'yes' if emitting else 'no',
                'Stopping voltage / V':
                    f'{(photonEnergy - phi*e)/e:.3f}' if emitting else '-',
                'Photocurrent / microA':
                    f'{current:.4f}' if emitting else '-',
            })
        st.dataframe(rows, hide_index=True, width='stretch')

    with notes:
        emittingCount = sum(1 for phi in workFunctions.values()
                            if (frequency*1e15) > (phi*e)/h)
        st.metric('Metals currently emitting',
                  f'{emittingCount} of {len(workFunctions)}')
        st.metric('Gradient of every line', f'{h/e:.4e} V/Hz')
        st.caption('The gradient is Planck constant over electron charge, '
                   'a combination of two fundamental constants with no '
                   'reference to the material. Measuring the gradient of '
                   'any one of these lines is a way of measuring h.')


# ----------------------------------------------------------------------
# Task 5: Hydrogenic emission spectrum
# ----------------------------------------------------------------------
def spectrumPage():
    st.header('Hydrogenic Emission Spectrum')

    seriesInfo = {
        'Lyman':    {'nFinal': 1, 'colour': '#c77dff'},
        'Balmer':   {'nFinal': 2, 'colour': '#ff4444'},
        'Paschen':  {'nFinal': 3, 'colour': '#4a9eff'},
        'Brackett': {'nFinal': 4, 'colour': '#39ff14'},
        'Pfund':    {'nFinal': 5, 'colour': '#ffd700'},
    }

    left, right = st.columns([1, 2.4])

    with left:
        Z = st.slider('Protons Z', 1, 6, 1)
        A = st.slider('Mass number A', 1, 14, 1)
        chosen = st.multiselect('Series', list(seriesInfo),
                                default=list(seriesInfo))
        nMax = st.slider('Highest level followed', 6, 60, 40)

        st.metric('Ionisation energy', f'{-orbitalEnergy(Z, A, 1):.3f} eV')

        # Every level scales as Z squared, so the whole spectrum shifts
        # to shorter wavelength as the nuclear charge is increased
        lymanLimit = h*c / (-orbitalEnergy(Z, A, 1) * e) * 1e9
        st.metric('Lyman series limit', f'{lymanLimit:.2f} nm')

        if Z > 1:
            st.caption(f'At Z={Z} every level is about {Z**2} times deeper '
                       f'than in hydrogen, so every wavelength is shorter '
                       f'by roughly the same factor.')

    with right:
        fig, (ax, axZoom) = plt.subplots(
            1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1.35, 1]})
        fig.patch.set_facecolor('#0e1117')

        for panel in (ax, axZoom):
            panel.set_facecolor('#0e1117')
            panel.tick_params(colors='white', labelsize=8)
            for spine in panel.spines.values():
                spine.set_edgecolor('#555555')
            panel.xaxis.label.set_color('white')
            panel.yaxis.label.set_color('white')
            panel.title.set_color('white')

        limitRows = []
        longest = 0

        for name in chosen:
            nFinal = seriesInfo[name]['nFinal']
            colour = seriesInfo[name]['colour']

            wavelengths, energies = [], []
            for nInitial in range(nFinal+1, nMax+1):
                photonEnergy = (orbitalEnergy(Z, A, nInitial)
                                - orbitalEnergy(Z, A, nFinal))
                wavelength = h*c / (photonEnergy * e) * 1e9
                wavelengths.append(wavelength)
                energies.append(photonEnergy)

                for panel in (ax, axZoom):
                    panel.plot([wavelength, wavelength], [0, photonEnergy],
                               color=colour, linewidth=0.9, alpha=0.75)

            ax.plot(wavelengths, energies, 'o', color=colour,
                    markersize=3.5, label=name)
            axZoom.plot(wavelengths, energies, 'o', color=colour,
                        markersize=4.5)

            longest = max(longest, wavelengths[0])
            limitEnergy = -orbitalEnergy(Z, A, nFinal)
            limitRows.append(f'{name:9s} n->{nFinal}  '
                             f'{h*c/(limitEnergy*e)*1e9:8.1f} nm')

        ax.set_xlabel('Wavelength / nm')
        ax.set_ylabel('Photon energy / eV')
        ax.set_title(f'Photon emissions, Z = {Z}')
        ax.set_xlim(0, max(longest*1.1, 100))
        ax.set_ylim(0, -orbitalEnergy(Z, A, 1) * 1.05)
        legend = ax.legend(facecolor='#0e1117', labelcolor='white',
                           loc='upper right', fontsize=8)
        legend.get_frame().set_edgecolor('#555555')

        if limitRows:
            ax.text(0.985, 0.55, 'Series limits\n' + '\n'.join(limitRows),
                    transform=ax.transAxes, color='white', fontsize=7,
                    family='monospace', ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='#0e1117',
                              edgecolor='#555555'))

        # The short wavelength lines crowd together, so the second panel
        # zooms in far enough to separate them
        zoomLimit = h*c / (-orbitalEnergy(Z, A, 1) * e) * 1e9
        axZoom.set_xlabel('Wavelength / nm')
        axZoom.set_ylabel('Photon energy / eV')
        axZoom.set_title('Short wavelength detail')
        axZoom.set_xlim(zoomLimit*0.85, zoomLimit*8)
        axZoom.set_ylim(0, -orbitalEnergy(Z, A, 1) * 1.05)

        if Z == 1:
            axZoom.axvspan(380, 700, color='cyan', alpha=0.10)
            axZoom.text(540, -orbitalEnergy(Z, A, 1)*0.94, 'visible',
                        color='cyan', fontsize=8, ha='center')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption('Each vertical line is one transition. Within a series the '
               'lines crowd together toward the series limit, because the '
               'energy levels themselves converge as one over n squared.')


# ----------------------------------------------------------------------
pages = {
    'Brownian motion': brownianPage,
    'Black body radiation': planckPage,
    'Photoelectric effect': photoelectricPage,
    'Hydrogenic spectrum': spectrumPage,
    'Electron diffraction': diffractionPage,
    'Particle in a box': boxPage,
    'Compton scattering': comptonPage,
    'Quantum cryptography': cryptographyPage,
    'Hydrogenic orbitals': orbitalPage,
}

st.sidebar.title('BPhO Challenge 2026')
st.sidebar.caption('Quantum Mechanics')
choice = st.sidebar.radio('Simulation', list(pages))
st.sidebar.divider()
st.sidebar.caption('Every page is driven by the same Python models used for '
                   'the written tasks, so the plots here are generated live '
                   'rather than being stored images.')
st.sidebar.divider()
st.sidebar.markdown('**Bryan Okafor**')
st.sidebar.caption('Littleover Community School')

pages[choice]()
