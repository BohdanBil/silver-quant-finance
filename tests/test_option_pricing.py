import numpy as np
import pytest

from option_pricing import monte_carlo_option_price, option_greeks, option_price

# Hull's textbook example: S = K = 100, r = 5%, sigma = 20%, T = 1y.
HULL = dict(s=100.0, k=100.0, r=0.05, sigma=0.20, T=1.0)


def test_black_scholes_matches_hull_example():
    assert option_price(**HULL, option_type="call", underlying="spot") == pytest.approx(10.4506, abs=1e-4)
    assert option_price(**HULL, option_type="put", underlying="spot") == pytest.approx(5.5735, abs=1e-4)


def test_black_scholes_greeks_match_hull_example():
    g = option_greeks(**HULL, option_type="call", underlying="spot")
    assert g.delta == pytest.approx(0.6368, abs=1e-4)
    assert g.gamma == pytest.approx(0.018762, abs=1e-6)
    assert g.vega == pytest.approx(37.524, abs=1e-3)
    assert g.theta == pytest.approx(-6.414, abs=1e-3)
    assert g.rho == pytest.approx(53.232, abs=1e-3)


@pytest.mark.parametrize("underlying", ["spot", "futures"])
def test_put_call_parity(underlying):
    s, k, r, sigma, T = 54.0, 50.0, 0.04, 0.38, 0.25
    call = option_price(s, k, r, sigma, T, "call", underlying)
    put = option_price(s, k, r, sigma, T, "put", underlying)
    forward_pv = s if underlying == "spot" else s * np.exp(-r * T)
    assert call - put == pytest.approx(forward_pv - k * np.exp(-r * T), abs=1e-10)


def test_black76_equals_black_scholes_on_the_discounted_futures_price():
    f, k, r, sigma, T = 54.0, 52.0, 0.04, 0.38, 0.25
    for kind in ("call", "put"):
        b76 = option_price(f, k, r, sigma, T, kind, "futures")
        bs = option_price(f * np.exp(-r * T), k, r, sigma, T, kind, "spot")
        assert b76 == pytest.approx(bs, abs=1e-10)


def test_black76_is_cheaper_than_treating_futures_as_spot():
    args = dict(s=53.71, k=53.71, r=0.04, sigma=0.39, T=63 / 252)
    assert option_price(**args, underlying="spot") / option_price(**args, underlying="futures") > 1.05


@pytest.mark.parametrize("underlying", ["spot", "futures"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(underlying, kind):
    s, k, r, sigma, T = 54.0, 50.0, 0.04, 0.38, 0.5
    g = option_greeks(s, k, r, sigma, T, kind, underlying)

    def price(**over):
        p = dict(s=s, k=k, r=r, sigma=sigma, T=T) | over
        return option_price(p["s"], p["k"], p["r"], p["sigma"], p["T"], kind, underlying)

    h = 1e-4
    assert g.delta == pytest.approx((price(s=s + h) - price(s=s - h)) / (2 * h), rel=1e-5)
    assert g.gamma == pytest.approx((price(s=s + h) - 2 * price() + price(s=s - h)) / h**2, rel=1e-3)
    assert g.vega == pytest.approx((price(sigma=sigma + h) - price(sigma=sigma - h)) / (2 * h), rel=1e-5)
    assert g.theta == pytest.approx(-(price(T=T + h) - price(T=T - h)) / (2 * h), rel=1e-5)
    assert g.rho == pytest.approx((price(r=r + h) - price(r=r - h)) / (2 * h), rel=1e-5)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        option_price(100, 100, 0.05, 0.2, 0.0)
    with pytest.raises(ValueError):
        option_price(100, 100, 0.05, 0.0, 1.0)


@pytest.mark.parametrize("underlying", ["spot", "futures"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_converges_to_closed_form(underlying, kind):
    args = dict(s=54.0, k=56.0, r=0.04, sigma=0.38, T=0.25)
    exact = option_price(**args, option_type=kind, underlying=underlying)
    mc = monte_carlo_option_price(**args, n_sims=200_000, option_type=kind, underlying=underlying, seed=1)
    assert abs(mc.price - exact) < 4 * mc.std_error


@pytest.mark.parametrize("antithetic,control_variate", [(False, False), (True, False), (False, True), (True, True)])
def test_reported_standard_error_matches_spread_across_replications(antithetic, control_variate):
    """The SE one run reports should equal the actual std of its price across independent runs."""
    args = dict(s=54.0, k=54.0, r=0.04, sigma=0.38, T=0.25, n_sims=4_000,
                antithetic=antithetic, control_variate=control_variate)
    runs = [monte_carlo_option_price(**args, seed=seed) for seed in range(300)]
    empirical_sd = np.std([run.price for run in runs], ddof=1)
    reported_se = np.mean([run.std_error for run in runs])
    assert reported_se / empirical_sd == pytest.approx(1.0, abs=0.15)
    assert abs(np.mean([run.price for run in runs]) - option_price(54.0, 54.0, 0.04, 0.38, 0.25)) \
        < 4 * empirical_sd / np.sqrt(len(runs))
