"""Period-scaling invariants of the derived statistics.

Scaling the peak frequency omega_p -> omega_p / k (all other parameters
unchanged) shifts the whole spectrum to lower frequencies, so *every*
characteristic period must stretch by the same factor k. In particular
the zero-crossing period Tz = 2*pi*sqrt(m0/m2) must scale exactly like
the peak period Tp = 2*pi/omega_p: the spectral moments scale as
m_n -> k**(4-n) * m_n under the self-similar JONSWAP form, hence
sqrt(m0/m2) -> k * sqrt(m0/m2).

A regression that computes m2 as the first moment (omega * S instead of
omega**2 * S) makes Tz scale only with sqrt(k) while Tp scales with k,
and is caught by the comparison below.

The second group pins the universal ordering Tz <= T01 (equivalently
m1**2 <= m0*m2, Cauchy-Schwarz), which must hold for any spectrum.
"""

import pytest

from app.service import compute_spectrum

ALPHA = 0.0081
GAMMA = 3.3
OMEGA_P = 0.6


@pytest.mark.parametrize("k", [2.0, 3.0])
@pytest.mark.parametrize("gamma", [GAMMA, 1.0])
def test_all_periods_scale_together_with_omega_p(k, gamma):
    """omega_p -> omega_p/k must stretch Tp, Tz and T01 by the same k."""
    base = compute_spectrum(omega_p=OMEGA_P, gamma=gamma, alpha=ALPHA)
    shifted = compute_spectrum(omega_p=OMEGA_P / k, gamma=gamma, alpha=ALPHA)

    assert shifted.stats.tp == pytest.approx(k * base.stats.tp, rel=1e-12)
    assert shifted.stats.tz == pytest.approx(k * base.stats.tz, rel=1e-9)
    assert shifted.stats.t01 == pytest.approx(k * base.stats.t01, rel=1e-9)

    # The Tz/Tp ratio is a shape constant: it must not drift with omega_p.
    ratio_base = base.stats.tz / base.stats.tp
    ratio_shifted = shifted.stats.tz / shifted.stats.tp
    assert ratio_shifted == pytest.approx(ratio_base, rel=1e-9)


@pytest.mark.parametrize("omega_p", [0.3, 0.5, 0.6, 0.9, 1.4])
@pytest.mark.parametrize("gamma", [1.0, 2.0, 3.3, 6.0])
def test_tz_never_exceeds_mean_period(omega_p, gamma):
    """Tz <= T01 for any spectrum (Cauchy-Schwarz: m1**2 <= m0*m2)."""
    result = compute_spectrum(omega_p=omega_p, gamma=gamma, alpha=ALPHA)
    m = result.moments
    assert m.m1**2 <= m.m0 * m.m2 * (1.0 + 1e-12)
    assert result.stats.tz <= result.stats.t01 * (1.0 + 1e-12)


def test_scaling_holds_under_hs_inversion():
    """The same omega_p scaling must hold when alpha is inverted from a
    target wave height instead of given directly."""
    k = 2.0
    base = compute_spectrum(omega_p=OMEGA_P, gamma=GAMMA, hs_target=3.0)
    shifted = compute_spectrum(omega_p=OMEGA_P / k, gamma=GAMMA, hs_target=3.0)

    assert shifted.stats.tp == pytest.approx(k * base.stats.tp, rel=1e-12)
    assert shifted.stats.tz == pytest.approx(k * base.stats.tz, rel=1e-9)
