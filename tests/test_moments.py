"""Moment integration tests: grid convergence, no halfway truncation of
the tail, consistency of the derived statistics, and the scaling laws
every characteristic period must obey when omega_p is rescaled."""

import math

import numpy as np
import pytest

from app.constants import OMEGA_MAX_FACTOR
from app.moments import derived_statistics, omega_grid, spectral_moments
from app.spectrum import jonswap

ALPHA = 0.0081
OMEGA_P = 0.5
GAMMA = 3.3


def _moments(n_points, omega_p=OMEGA_P, max_factor=None):
    if max_factor is None:
        omega = omega_grid(omega_p, n_points)
    else:
        omega = np.linspace(0.05 * omega_p, max_factor * omega_p, n_points)
    return spectral_moments(omega, jonswap(omega, omega_p, ALPHA, GAMMA))


def test_moments_converge_under_grid_refinement():
    coarse = _moments(1024)
    fine = _moments(4096)
    finest = _moments(16384)

    for name in ("m0", "m1", "m2"):
        assert getattr(fine, name) == pytest.approx(getattr(coarse, name), rel=1e-4)
        assert getattr(finest, name) == pytest.approx(getattr(fine, name), rel=1e-5)


def test_integration_tail_not_truncated():
    """Extending the cut-off beyond the default multiple of omega_p must
    barely change m0: the default grid already captures the tail."""
    default = _moments(4096)
    extended = _moments(16384, max_factor=2 * OMEGA_MAX_FACTOR)

    # m0 (the quantity the Hs constraint closes on) must be stable to well
    # under a permille. Higher moments weight the tail by omega and omega**2
    # respectively, so their residual tail sensitivity is looser -- but a
    # grid truncated halfway to the peak would move them by tens of percent,
    # far outside these bounds.
    assert extended.m0 == pytest.approx(default.m0, rel=1e-3)
    assert extended.m1 == pytest.approx(default.m1, rel=5e-3)
    assert extended.m2 == pytest.approx(default.m2, rel=2e-2)


def test_moments_are_positive_and_ordered():
    m = _moments(4096)
    assert m.m0 > 0 and m.m1 > 0 and m.m2 > 0


def test_derived_statistics_consistency():
    m = _moments(4096)
    stats = derived_statistics(m, OMEGA_P)

    assert stats.hs == pytest.approx(4.0 * math.sqrt(m.m0), rel=1e-15)
    assert stats.tp == pytest.approx(2.0 * math.pi / OMEGA_P, rel=1e-15)
    assert stats.tz == pytest.approx(2.0 * math.pi * math.sqrt(m.m0 / m.m2))
    assert stats.t01 == pytest.approx(2.0 * math.pi * m.m0 / m.m1)
    # Physically sensible ordering for a narrow-band spectrum.
    assert stats.t01 < stats.tp


def test_halving_omega_p_scales_all_periods_by_two():
    """With alpha and gamma fixed, shifting omega_p -> omega_p/k rescales
    the whole spectrum in frequency only, so EVERY characteristic period
    must stretch by the same factor k: Tz/Tp is a shape constant of the
    spectrum family, not a function of omega_p. Regression test for the
    second-moment integrand (m2 needs omega**2 * s, not omega * s -- with
    the wrong weight Tz degenerates to 2*pi*sqrt(m0/m1) and scales with
    sqrt(k) instead of k, e.g. 1.41x instead of 2x for k=2)."""
    stats1 = derived_statistics(_moments(4096, omega_p=0.6), 0.6)
    stats2 = derived_statistics(_moments(4096, omega_p=0.3), 0.3)

    assert stats2.tp == pytest.approx(2.0 * stats1.tp, rel=1e-12)
    assert stats2.tz == pytest.approx(2.0 * stats1.tz, rel=1e-9)
    assert stats2.t01 == pytest.approx(2.0 * stats1.t01, rel=1e-9)
    # Equivalently: Tz/Tp must be the same constant for both spectra.
    assert stats2.tz / stats2.tp == pytest.approx(
        stats1.tz / stats1.tp, rel=1e-9
    )


def test_tz_never_exceeds_mean_period_t01():
    """Cauchy-Schwarz on the spectral measure gives m1**2 <= m0*m2, hence
    Tz = 2*pi*sqrt(m0/m2) <= 2*pi*m0/m1 = T01 for ANY spectrum. With a
    wrong m2 weight this ordering flips whenever m0/m1 < 1 s."""
    for omega_p in (0.3, 0.6, 1.0, 1.5):
        for gamma in (1.0, 3.3, 6.0):
            omega = omega_grid(omega_p, 4096)
            m = spectral_moments(omega, jonswap(omega, omega_p, ALPHA, gamma))
            stats = derived_statistics(m, omega_p)
            assert stats.tz <= stats.t01
