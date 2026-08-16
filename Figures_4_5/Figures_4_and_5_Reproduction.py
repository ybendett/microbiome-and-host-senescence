import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import integrate
from scipy.optimize import brentq, curve_fit
from svgpath2mpl import parse_path


# Reproducibility notes
# ---------------------
# This script implements the multiple-age-class simulations used for the
# empirical guppy and yellow-baboon analyses. The current manuscript numbers 
# the corresponding main-text panels as Figs. 4 and 5, although the local code
# labels are kept as Fig. 1 and Fig. 2.
#
# Required packages include NumPy, pandas, Matplotlib, seaborn, SciPy,
# svgpath2mpl, and an Excel engine that reads legacy .xls files (for example,
# xlrd). Required local files are:
# 1. The Jones et al. (2014) life-table workbook used below is avaliable for
#    download at: 
#    https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnature12789/MediaObjects/41586_2014_BFnature12789_MOESM42_ESM.xls
# 2. The guppy and baboon SVG files referenced in the plotting functions, are 
#    avaliable at: 
#    XXX link to github
#
# The manuscript reports Python 3.14. Record package versions and preserve the
# random-free deterministic settings below when archiving final results.
# Suppressing all warnings reproduces the original execution behavior, but
# removing this line temporarily is useful when validating a new environment.
warnings.filterwarnings("ignore")


# Load empirical life-history data.
# Update this path to the local copy of the Jones et al. (2014) supplementary
# workbook or run the script from the same folder it is saved at. 
# Sheet 42 contains the guppy data and sheet 37 contains the yellow
# baboon data used by the fitting functions below. 
path = (r"41586_2014_BFnature12789_MOESM42_ESM.xls")
life_tables = pd.read_excel(path, [x for x in range(1, 90)])


# Empirical fitting functions
# ---------------------------
def l_x_fit(x, M, G, R):
    """Return Gompertz-Makeham survival for curve fitting.

    Parameters follow the order used by ``curve_fit``: M, G, and R.
    """
    if abs(R) < 1e-12:
        return np.exp(-(G + M) * x)

    return np.exp(-(G / R) * (np.exp(R * x) - 1.0) - M * x)


def mortality_fit(x, M, G, R):
    """Return Gompertz-Makeham mortality for curve fitting."""
    return G * np.exp(R * x) + M


def m_x_fit(x, alpha, beta, gamma):
    """Return Gamma-type reproduction, m(x) = alpha*x^beta*exp(-gamma*x)."""
    return alpha * (x**beta) * np.exp(-gamma * x)


# Model parameters
# ----------------
@dataclass(frozen=True)
class Params:
    """Store demographic, microbial, and heatmap parameters."""

    age_classes: int
    
    # vertical transmission parameter
    v: float

    # Horizontal transmission probabilities.
    T_A: float
    T_B: float

    # Initial population sizes used to construct stable age distributions.
    N_A: float
    N_B: float

    # Gamma-type reproduction: m(x) = alpha*x^beta*exp(-gamma*x).
    alpha_A: float
    alpha_B: float
    beta_A: float
    beta_B: float
    gamma_A: float
    gamma_B: float

    # Gompertz-Makeham mortality: mu(x) = G*exp(R*x) + M.
    G_A: float
    G_B: float
    R_A: float
    R_B: float
    M_A: float
    M_B: float

    # When True, heatmaps vary T_A/T_B while keeping T_B fixed. When False,
    # both transmission probabilities are set to the same absolute value.
    T_ratio: bool = True

    # Gompertz-Makeham parameter modified by microbe A: "R", "G", or "M".
    mortality_parameter: str = "R"

    # Upper bound of 1 - theta_A/theta_B shown in mortality-benefit heatmaps.
    # Values below 1 avoid setting theta_A exactly to zero. The G-based
    # supplementary analysis uses 0.95 to prevent extremely long age grids.
    mortality_benefit_max: float = 1.0

    # Safety limit for the adaptive number of age classes.
    max_age_classes: int = 5000

    # Heatmap dimension held fixed:
    # "m" varies mortality benefit and transmission;
    # "mortality" varies reproduction and transmission;
    # "T" varies reproduction and mortality benefit.
    # The aliases "R", "G", and "M" are accepted for "mortality".
    exclude: str = "m"


VALID_MORTALITY_PARAMETERS = ("R", "G", "M")


def selected_mortality_parameter(p: Params):
    """Return the selected Gompertz-Makeham parameter: R, G, or M."""
    trait = str(p.mortality_parameter).upper()

    if trait not in VALID_MORTALITY_PARAMETERS:
        raise ValueError(
            "p.mortality_parameter must be one of: 'R', 'G', or 'M'."
        )

    return trait


def normalized_exclude(p: Params):
    """Normalize current and backward-compatible heatmap-mode names."""
    exclude = str(p.exclude)

    if exclude == "m":
        return "m"

    if exclude == "T":
        return "T"

    if (
        exclude.lower() == "mortality"
        or exclude.upper() in VALID_MORTALITY_PARAMETERS
    ):
        return "mortality"

    raise ValueError(
        "p.exclude must be one of: 'm', 'mortality', 'T', "
        "or the backward-compatible aliases 'R', 'G', and 'M'."
    )


def mortality_parameter_values(p: Params):
    """Return (parameter_A, parameter_B) for the selected mortality trait."""
    trait = selected_mortality_parameter(p)
    return getattr(p, f"{trait}_A"), getattr(p, f"{trait}_B")


def with_A_mortality_value(p: Params, value):
    """Return a copy of p with A's selected mortality parameter changed."""
    trait = selected_mortality_parameter(p)
    return replace(p, **{f"{trait}_A": float(value)})


def with_A_mortality_ratio(p: Params, ratio):
    """Set selected_parameter_A to ratio*selected_parameter_B."""
    trait = selected_mortality_parameter(p)
    baseline = getattr(p, f"{trait}_B")
    return replace(p, **{f"{trait}_A": float(ratio) * baseline})


def mortality_parameter_ratio(p: Params):
    """Return selected_parameter_A/selected_parameter_B."""
    value_A, value_B = mortality_parameter_values(p)

    if value_B == 0:
        return np.nan

    return value_A / value_B


def mortality_relative_benefit(p: Params):
    """Return 1 - selected_parameter_A/selected_parameter_B."""
    ratio = mortality_parameter_ratio(p)
    return 1.0 - ratio if np.isfinite(ratio) else np.nan


def mortality_benefit_column(p: Params):
    """Return the trait-specific relative-benefit column name."""
    trait = selected_mortality_parameter(p)
    return f"{trait}_relative_benefit"


def mortality_benefit_axis_label(p: Params):
    """Return a biologically explicit axis label for the selected trait."""
    trait = selected_mortality_parameter(p)

    labels = {
        "R": (
            r"Relative decrease in senescence rate "
            r"$(1 - R_A/R_B)$"
        ),
        "G": (
            r"Relative decrease in age-dependent mortality "
            r"$(1 - G_A/G_B)$"
        ),
        "M": (
            r"Relative decrease in age-independent mortality "
            r"$(1 - M_A/M_B)$"
        ),
    }

    return labels[trait]


# Gompertz-Makeham survival
# -------------------------
def cumulative_hazard_gm(x, G, R, M):
    """Return the cumulative Gompertz-Makeham hazard.

    H(x) = integral from 0 to x of [G*exp(R*t) + M] dt.

    The implementation handles G = 0 explicitly and converts positive-R
    exponential overflow into an effectively infinite cumulative hazard.
    """
    x = np.asarray(x, dtype=float)

    if G < 0 or M < 0:
        raise ValueError("G and M must be non-negative.")

    # Handle an absent Gompertz component before evaluating exp(R*x), thereby
    # avoiding the undefined product 0*inf.
    if G == 0:
        result = M * x
    elif abs(R) < 1e-12:
        result = (G + M) * x
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            result = (G / R) * np.expm1(R * x) + M * x

        if R > 0:
            result = np.where(np.isnan(result), np.inf, result)

    if result.ndim == 0:
        return float(result)

    return result


def l_x_gm(x, G, R, M):
    """Return Gompertz-Makeham survival, l(x) = exp[-H(x)]."""
    return np.exp(-cumulative_hazard_gm(x, G, R, M))


def l_x_A(x, p: Params):
    """Return survival to age x for hosts carrying microbe A."""
    return l_x_gm(x, p.G_A, p.R_A, p.M_A)


def l_x_B(x, p: Params):
    """Return survival to age x for hosts carrying microbe B."""
    return l_x_gm(x, p.G_B, p.R_B, p.M_B)


def gm_death_age(G, R, M, ACC=0.0001):
    """Return the age at which survival reaches ``ACC``.

    Analytic solutions are used when one mortality component is absent.
    Otherwise, finite analytically derived brackets are passed to Brent's
    method, avoiding repeated upper-bound doubling into overflow ranges.
    """
    if not 0 < ACC < 1:
        raise ValueError("ACC must be between 0 and 1.")

    if G < 0 or M < 0:
        raise ValueError("G and M must be non-negative.")

    target = float(np.log(1.0 / ACC))
    eps = 1e-12

    # With no mortality, survival never reaches ACC.
    if G == 0 and M == 0:
        return np.inf

    # R = 0 gives a constant total hazard.
    if abs(R) < eps:
        rate = G + M
        return target / rate if rate > 0 else np.inf

    # Pure Makeham mortality.
    if G == 0:
        return target / M if M > 0 else np.inf

    # Pure Gompertz mortality has a closed-form solution.
    if M == 0:
        z = 1.0 + R * target / G

        # For R < 0, cumulative Gompertz hazard approaches G/abs(R). If the
        # target exceeds this limit, survival never reaches ACC.
        if z <= 0:
            return np.inf

        return float(np.log(z) / R)

    def f(x):
        return cumulative_hazard_gm(x, G, R, M) - target

    # The full hazard is at least as large as either component alone, so its
    # root cannot exceed either component-specific root.
    makeham_upper = target / M

    if R > 0:
        gompertz_upper = np.log1p(R * target / G) / R
        upper = min(makeham_upper, gompertz_upper)
    else:
        # With declining Gompertz mortality, M > 0 still guarantees crossing.
        upper = makeham_upper

    f_upper = f(upper)

    # Protect against roundoff at an analytically exact upper bound.
    if f_upper < 0:
        upper *= 1.0 + 1e-10
        f_upper = f(upper)

    if not np.isfinite(f_upper) and not np.isposinf(f_upper):
        raise RuntimeError("Non-finite hazard while bracketing death age.")

    if f_upper < 0:
        raise RuntimeError("Could not bracket death age.")

    return brentq(f, 0.0, upper)


def death_age_A(p: Params, ACC=0.0001):
    """Return A's effective maximum age at survival threshold ACC."""
    return gm_death_age(p.G_A, p.R_A, p.M_A, ACC=ACC)


def death_age_B(p: Params, ACC=0.0001):
    """Return B's effective maximum age at survival threshold ACC."""
    return gm_death_age(p.G_B, p.R_B, p.M_B, ACC=ACC)


def age_grid(p: Params, ACC=0.0001):
    """Construct the common adaptive age grid used by both microbial types.

    B defines the baseline interval width, dx = death_age_B/age_classes.
    Additional classes are added until A's longer death horizon is covered.
    The manuscript simulations use ACC = 0.0001.
    """
    x_d_A = death_age_A(p, ACC=ACC)
    x_d_B = death_age_B(p, ACC=ACC)

    if not np.isfinite(x_d_A) or not np.isfinite(x_d_B):
        raise ValueError("Infinite death age. Check mortality parameters.")

    dx = x_d_B / p.age_classes
    n_classes = int(math.ceil(x_d_A / dx))
    n_classes = max(n_classes, 2)

    if n_classes > p.max_age_classes:
        trait = selected_mortality_parameter(p)
        value_A, value_B = mortality_parameter_values(p)
        raise ValueError(
            f"The adaptive age grid requires {n_classes:,} classes, "
            f"exceeding max_age_classes={p.max_age_classes:,}. The selected "
            f"mortality parameter is {trait} (A={value_A:g}, B={value_B:g}), "
            f"giving death ages A={x_d_A:g} and B={x_d_B:g}. Reduce "
            "mortality_benefit_max, increase the remaining mortality "
            "component, or raise max_age_classes only if the resulting "
            "Leslie matrix is computationally feasible."
        )

    return n_classes, dx, x_d_A, x_d_B


# Reproduction
# ------------
def m_x_A(x, p: Params):
    """Return age-specific reproduction for hosts carrying microbe A."""
    return p.alpha_A * (x**p.beta_A) * np.exp(-p.gamma_A * x)


def m_x_B(x, p: Params):
    """Return age-specific reproduction for hosts carrying microbe B."""
    return p.alpha_B * (x**p.beta_B) * np.exp(-p.gamma_B * x)


def reproduction_age_class_A(age_class, p: Params, dx):
    """Return expected A reproduction during one age interval.

    Reproduction is conditional on the host being alive at the beginning of
    the interval, matching the continuous-to-discrete conversion in Methods.
    """
    a = age_class * dx
    b = a + dx

    base = l_x_A(a, p)

    if base <= 0 or not np.isfinite(base):
        return 0.0

    def integrand(x):
        return m_x_A(x, p) * (l_x_A(x, p) / base)

    value, _ = integrate.quad(integrand, a, b)
    return value


def reproduction_age_class_B(age_class, p: Params, dx):
    """Return expected B reproduction during one age interval."""
    a = age_class * dx
    b = a + dx

    base = l_x_B(a, p)

    if base <= 0 or not np.isfinite(base):
        return 0.0

    def integrand(x):
        return m_x_B(x, p) * (l_x_B(x, p) / base)

    value, _ = integrate.quad(integrand, a, b)
    return value


# Survival between age classes
# ----------------------------
def survival_age_class_A(age_class, p: Params, dx, n_classes):
    """Return conditional A survival into the next discrete age class."""
    if age_class >= n_classes - 1:
        return 0.0

    a = age_class * dx
    b = a + dx

    base = l_x_A(a, p)

    if base <= 0 or not np.isfinite(base):
        return 0.0

    return l_x_A(b, p) / base


def survival_age_class_B(age_class, p: Params, dx, n_classes):
    """Return conditional B survival into the next discrete age class."""
    if age_class >= n_classes - 1:
        return 0.0

    a = age_class * dx
    b = a + dx

    base = l_x_B(a, p)

    if base <= 0 or not np.isfinite(base):
        return 0.0

    return l_x_B(b, p) / base


# Leslie matrix
# -------------
def leslie_matrix(p: Params, n_classes, dx):
    """Build the block-structured Leslie matrix.

    State-vector order is [A0, B0, A1, B1, A2, B2, ...]. Fertility occupies
    the newborn rows, and conditional survival occupies the subdiagonal.
    """
    L = np.zeros((2 * n_classes, 2 * n_classes))

    for i in range(n_classes):
        # Fertility contributions to the newborn A and B classes.
        L[0, 2 * i] = reproduction_age_class_A(i, p, dx)
        L[1, 2 * i + 1] = reproduction_age_class_B(i, p, dx)

        # Survival contributions to the next age class.
        if i < n_classes - 1:
            L[2 * (i + 1), 2 * i] = survival_age_class_A(
                i,
                p,
                dx,
                n_classes,
            )
            L[2 * (i + 1) + 1, 2 * i + 1] = survival_age_class_B(
                i,
                p,
                dx,
                n_classes,
            )

    return L


# Transmission matrix
# -------------------
def transmission_matrix(p: Params, dist):
    """Build the frequency-dependent horizontal-transmission matrix."""
    total = dist.sum()

    if total <= 0 or not np.isfinite(total):
        raise ValueError("Population distribution has invalid total.")

    pi_A = dist[0::2].sum() / total
    pi_B = dist[1::2].sum() / total

    n_classes = len(dist) // 2
    T = np.zeros((2 * n_classes, 2 * n_classes))

    for i in range(n_classes):
        A = 2 * i
        B = 2 * i + 1

        # An A carrier either remains A or is replaced by B.
        T[A, A] = 1.0 - pi_B * p.T_B
        T[B, A] = pi_B * p.T_B

        # A B carrier either remains B or is replaced by A.
        T[B, B] = 1.0 - pi_A * p.T_A
        T[A, B] = pi_A * p.T_A

    return T


# Stable starting distributions
# -----------------------------
def stable_dist_A(p: Params, n_classes, dx):
    """Construct A's survival-weighted initial age distribution."""
    ages = np.arange(n_classes) * dx
    lx = np.array([l_x_A(x, p) for x in ages])

    total = lx.sum()

    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid A survival distribution.")

    return p.N_A * lx / total


def stable_dist_B(p: Params, n_classes, dx):
    """Construct B's survival-weighted initial age distribution."""
    ages = np.arange(n_classes) * dx
    lx = np.array([l_x_B(x, p) for x in ages])

    total = lx.sum()

    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid B survival distribution.")

    return p.N_B * lx / total


def stable_dist_two_strains(p: Params, n_classes, dx):
    """Interleave A and B stable age distributions in model-state order."""
    dist_A = stable_dist_A(p, n_classes, dx)
    dist_B = stable_dist_B(p, n_classes, dx)

    dist = np.zeros(2 * n_classes)
    dist[0::2] = dist_A
    dist[1::2] = dist_B

    return dist


# Simulation
# ----------
def simulation_step(dist, p: Params, L):
    """Advance the population by one time step.

    Event order follows the implemented model: horizontal transmission,
    demography, density regulation, and newborn microbial assignment.
    """
    # 1. Horizontal transmission during one random host interaction.
    dist = transmission_matrix(p, dist) @ dist

    total = dist.sum()
    if total <= 0 or not np.isfinite(total):
        return dist, False

    pi_A = dist[0::2].sum() / total
    pi_B = dist[1::2].sum() / total

    # 2. Reproduction and survival through the Leslie matrix.
    dist = L @ dist

    births = dist[0] + dist[1]
    if births <= 0 or not np.isfinite(births):
        return dist, False

    # 3. Density regulation keeps total abundance at N_A + N_B.
    target_N = p.N_A + p.N_B
    deaths = target_N - dist.sum() + births

    dist[0] *= deaths / births
    dist[1] *= deaths / births

    # 4. Assign newborn microbes by vertical transmission and random
    # acquisition from the post-transmission adult microbial frequencies.
    A_births = dist[0]
    B_births = dist[1]

    A_from_A = p.v * A_births
    B_from_B = p.v * B_births

    random_births = (1.0 - p.v) * (A_births + B_births)

    A_random = random_births * pi_A
    B_random = random_births * pi_B

    dist[0] = A_from_A + A_random
    dist[1] = B_from_B + B_random

    if np.any(~np.isfinite(dist)) or dist.sum() <= 0:
        return dist, False

    return dist, True


def simulation(
    p: Params,
    steps,
    invasion_steps=100,
    invasion_p=0.01,
    ACC=0.0001,
):
    """Classify reciprocal invasion and, when needed, coexistence dynamics.

    Each strain is first introduced at 1% of its baseline abundance and tested
    for 100 time steps. Reciprocal invasion outcomes identify fixation or
    bistability. When both strains invade, the full coexistence simulation is
    continued for ``steps`` time steps and A's final frequency is returned.
    """
    n_classes, dx, _, _ = age_grid(p, ACC=ACC)

    # Rare A invading common B.
    p_low_A = replace(p, N_A=p.N_A * invasion_p, N_B=p.N_B)
    dist_low_A = stable_dist_two_strains(p_low_A, n_classes, dx)
    L_low_A = leslie_matrix(p_low_A, n_classes, dx)

    A_status = "survive"

    for _ in range(invasion_steps):
        dist_low_A, ok = simulation_step(
            dist_low_A,
            p_low_A,
            L_low_A,
        )

        if not ok:
            A_status = "extinct"
            break

    if dist_low_A[0::2].sum() + 1e-8 < p_low_A.N_A:
        A_status = "extinct"

    # Rare B invading common A.
    p_low_B = replace(p, N_A=p.N_A, N_B=p.N_B * invasion_p)
    dist_low_B = stable_dist_two_strains(p_low_B, n_classes, dx)
    L_low_B = leslie_matrix(p_low_B, n_classes, dx)

    B_status = "survive"

    for _ in range(invasion_steps):
        dist_low_B, ok = simulation_step(
            dist_low_B,
            p_low_B,
            L_low_B,
        )

        if not ok:
            B_status = "extinct"
            break

    if dist_low_B[1::2].sum() + 1e-8 < p_low_B.N_B:
        B_status = "extinct"

    # Map reciprocal invasion outcomes to heatmap states.
    if A_status == "extinct" and B_status == "extinct":
        return 0.5, False

    if A_status == "extinct" and B_status == "survive":
        return 0.0, True

    if A_status == "survive" and B_status == "extinct":
        return 1.0, True

    # Both strains invade when rare: estimate the coexistence frequency.
    dist = stable_dist_two_strains(p, n_classes, dx)
    L = leslie_matrix(p, n_classes, dx)

    for _ in range(steps):
        dist, ok = simulation_step(dist, p, L)

        if not ok:
            return 0.5, False

    A_ratio = dist[0::2].sum() / dist.sum()

    return A_ratio, True


# Heatmap data
# ------------
def heatmap_data(res, p: Params, steps):
    """Generate simulation grids for R-, G-, or M-mediated mortality effects.

    The selected trait is ``p.mortality_parameter`` and its relative benefit
    is 1 - theta_A/theta_B. Heatmap mode is selected by ``p.exclude``:

    * ``"m"`` holds alpha_A/alpha_B fixed and varies mortality benefit and
      transmission.
    * ``"mortality"`` holds theta_A/theta_B fixed and varies reproduction and
      transmission.
    * ``"T"`` holds transmission fixed and varies reproduction and mortality
      benefit.
    """
    records = []
    exclude = normalized_exclude(p)
    trait = selected_mortality_parameter(p)
    benefit_col = mortality_benefit_column(p)

    def append_record(p_tmp, A_ratio, stability, benefit=None):
        if benefit is None:
            benefit = mortality_relative_benefit(p_tmp)

        value_A, value_B = mortality_parameter_values(p_tmp)

        record = {
            "T": p_tmp.T_A,
            "T_ratio": (
                p_tmp.T_A / p_tmp.T_B
                if p_tmp.T_B != 0
                else np.nan
            ),
            "mortality_parameter": trait,
            "mortality_A": value_A,
            "mortality_B": value_B,
            "mortality_ratio": mortality_parameter_ratio(p_tmp),
            "mortality_relative_benefit": benefit,
            benefit_col: benefit,
            "m_ratio": p_tmp.alpha_A / p_tmp.alpha_B,
            "A_ratio": A_ratio,
            "stability": stability,
        }

        records.append(record)

    # Hold reproduction fixed; vary mortality benefit and transmission.
    if exclude == "m":
        if not 0.0 <= p.mortality_benefit_max <= 1.0:
            raise ValueError(
                "mortality_benefit_max must be between 0 and 1."
            )

        mortality_benefit_values = np.linspace(
            0.0,
            p.mortality_benefit_max,
            res + 1,
        )

        if p.T_ratio:
            T_values = np.linspace(0.0, p.T_B, res + 1)
        else:
            T_values = np.linspace(0.0, 0.5, res + 1)

        _, mortality_B = mortality_parameter_values(p)

        for T_new in T_values:
            for benefit in mortality_benefit_values:
                mortality_A_new = (1.0 - benefit) * mortality_B
                p_tmp = with_A_mortality_value(p, mortality_A_new)

                if p.T_ratio:
                    p_tmp = replace(p_tmp, T_A=T_new)
                else:
                    p_tmp = replace(p_tmp, T_A=T_new, T_B=T_new)

                A_ratio, stability = simulation(p_tmp, steps)
                append_record(
                    p_tmp,
                    A_ratio,
                    stability,
                    benefit=benefit,
                )

        sim_results = pd.DataFrame(records)
        x_col = "T_ratio" if p.T_ratio else "T"

        heat = sim_results.pivot(
            index="mortality_relative_benefit",
            columns=x_col,
            values="A_ratio",
        )

        stability_mask = ~sim_results.pivot(
            index="mortality_relative_benefit",
            columns=x_col,
            values="stability",
        )

        return heat, stability_mask, sim_results

    # Hold the selected mortality ratio fixed; vary reproduction and T.
    if exclude == "mortality":
        alpha_values = np.linspace(
            0.5 * p.alpha_B,
            p.alpha_B,
            res + 1,
        )

        if p.T_ratio:
            T_values = np.linspace(0.0, p.T_B, res + 1)
        else:
            T_values = np.linspace(0.0, 0.5, res + 1)

        for T_new in T_values:
            for alpha_A_new in alpha_values:
                if p.T_ratio:
                    p_tmp = replace(
                        p,
                        T_A=T_new,
                        alpha_A=alpha_A_new,
                    )
                else:
                    p_tmp = replace(
                        p,
                        T_A=T_new,
                        T_B=T_new,
                        alpha_A=alpha_A_new,
                    )

                A_ratio, stability = simulation(p_tmp, steps)
                append_record(p_tmp, A_ratio, stability)

        sim_results = pd.DataFrame(records)
        x_col = "T_ratio" if p.T_ratio else "T"

        heat = sim_results.pivot(
            index="m_ratio",
            columns=x_col,
            values="A_ratio",
        )

        stability_mask = ~sim_results.pivot(
            index="m_ratio",
            columns=x_col,
            values="stability",
        )

        return heat, stability_mask, sim_results

    # Hold transmission fixed; vary reproduction and mortality benefit.
    if exclude == "T":
        alpha_values = np.linspace(
            0.5 * p.alpha_B,
            p.alpha_B,
            res + 1,
        )

        if not 0.0 <= p.mortality_benefit_max <= 1.0:
            raise ValueError(
                "mortality_benefit_max must be between 0 and 1."
            )

        mortality_benefit_values = np.linspace(
            0.0,
            p.mortality_benefit_max,
            res + 1,
        )
        _, mortality_B = mortality_parameter_values(p)

        for benefit in mortality_benefit_values:
            mortality_A_new = (1.0 - benefit) * mortality_B

            for alpha_A_new in alpha_values:
                p_tmp = with_A_mortality_value(p, mortality_A_new)
                p_tmp = replace(p_tmp, alpha_A=alpha_A_new)

                A_ratio, stability = simulation(p_tmp, steps)
                append_record(
                    p_tmp,
                    A_ratio,
                    stability,
                    benefit=benefit,
                )

        sim_results = pd.DataFrame(records)

        heat = sim_results.pivot(
            index="m_ratio",
            columns="mortality_relative_benefit",
            values="A_ratio",
        )

        stability_mask = ~sim_results.pivot(
            index="m_ratio",
            columns="mortality_relative_benefit",
            values="stability",
        )

        return heat, stability_mask, sim_results

    raise RuntimeError("Unexpected normalized heatmap mode.")


# Plotting helpers
# ----------------
def fmt(s):
    """Format a numeric heatmap tick with two decimal places."""
    try:
        n = "{:.2f}".format(float(s))
    except Exception:
        n = ""
    return n


def fmt1(s):
    """Format a numeric heatmap tick with one decimal place."""
    try:
        n = "{:.1f}".format(float(s))
    except Exception:
        n = ""
    return n


def set_sparse_ticklabels(ax, every=10):
    """Format heatmap ticks and retain every nth tick for readability."""
    ax.set_xticklabels(
        [fmt(label.get_text()) for label in ax.get_xticklabels()],
        fontsize=15,
        rotation=0,
    )

    ax.set_yticklabels(
        [fmt1(label.get_text()) for label in ax.get_yticklabels()],
        fontsize=15,
    )

    xlocs = ax.get_xticks()
    ylocs = ax.get_yticks()

    if len(xlocs) > every:
        ax.set_xticks(xlocs[::every])

    if len(ylocs) > every:
        ax.set_yticks(ylocs[::every])


def plot_heatmap_panel(ax, res, p: Params, steps, cmap):
    """Simulate and draw one equilibrium-frequency heatmap panel.

    Cells masked in grey are reciprocal non-invasion outcomes: both fixation
    states are stable, so the eventual state depends on initial conditions.
    """
    heat, stability_mask, sim_results = heatmap_data(res, p, steps)

    sns.heatmap(
        heat,
        ax=ax,
        mask=stability_mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        cbar=False,
    )

    set_sparse_ticklabels(ax, every=10)

    if p.T_ratio:
        ax.set_xlabel(
            r"Transmission ratio $(T_A/T_B)$",
            fontsize=18,
        )
    else:
        ax.set_xlabel(r"Transmission rate $(T)$", fontsize=18)

    exclude = normalized_exclude(p)

    if exclude == "m":
        ax.set_ylabel(mortality_benefit_axis_label(p), fontsize=17)
    elif exclude in ("mortality", "T"):
        ax.set_ylabel(r"$\alpha_A/\alpha_B$", fontsize=18)

    ax.invert_yaxis()

    return heat, stability_mask, sim_results


def plot_guppy_life_history(ax, life_tables):
    """Fit and plot guppy survival and reproduction schedules.

    The empirical data are read from sheet 42 of the Jones et al. workbook.
    Ages are converted from months to years and reproduction to annual units,
    matching the parameterization used for the guppy simulations.
    """
    gup_df = life_tables[42]

    xx = np.array(gup_df["Guppy"][1:41].array, dtype=float) / 12
    yy = (
        np.array(
            gup_df["Poecilia reticulata"][1:41].array,
            dtype=float,
        )
        / 60
    )

    popt, _ = curve_fit(
        l_x_fit,
        xx,
        yy,
        bounds=(0.00001, [0.2, 0.2, 2.0]),
    )

    M, G, R = popt
    lx = [l_x_fit(x, M, G, R) for x in xx]

    l1 = ax.plot(
        xx,
        yy,
        "o",
        c="lightcoral",
        label=r"guppy population $l_x$",
    )

    l2 = ax.plot(
        xx,
        lx,
        "--",
        c="indianred",
        label=r"estimated $l_x$ function",
    )

    ax.set_ylabel(r"Survival $(l_x)$", fontsize=18)
    ax.set_xlabel(r"Age in years ($x$)", fontsize=18)
    ax.tick_params(labelsize=15)

    ax2 = ax.twinx()

    fertility = np.array(gup_df[2][1:41].array, dtype=float) * 12

    l3 = ax2.plot(
        xx,
        fertility,
        "^",
        c="maroon",
        label=r"guppy population $m_x$",
    )

    popt, _ = curve_fit(
        m_x_fit,
        xx,
        fertility,
        bounds=(0.00001, [20.0, 20.0, 20.0]),
    )

    alpha, beta, gamma = popt
    mx = [m_x_fit(x, alpha, beta, gamma) for x in xx]

    l4 = ax2.plot(
        xx,
        mx,
        c="darkred",
        label=r"estimated $m_x$ function",
    )

    ax2.set_ylabel(r"Reproduction $(m_x)$", fontsize=18)
    ax2.set_xlabel(r"Age in years ($x$)", fontsize=18)
    ax2.tick_params(labelsize=15)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.42, 0),
        fontsize=10,
    )

    ax.set_title("Guppy life-history", fontsize=14)


def plot_baboon_life_history(ax, life_tables):
    """Fit and plot yellow-baboon survival and reproduction schedules.

    The empirical mortality and fertility data are read from sheet 37 of the
    Jones et al. workbook. Annual survival is reconstructed from the empirical
    mortality probabilities before comparison with the fitted survival curve.
    """
    yellow_baboon_df = life_tables[37]

    xx = np.array(
        yellow_baboon_df["Yellow baboon"][1:29].array,
        dtype=float,
    )
    yy = np.array(yellow_baboon_df[5][1:29].array, dtype=float)

    popt, _ = curve_fit(
        mortality_fit,
        xx,
        yy,
        bounds=(0.00001, [1.0, 1.0, 5.0]),
    )

    M, G, R = popt

    zz = 1.0 - yy
    empirical_lx = [math.prod(zz[:i]) for i in range(len(zz))]
    lx = [l_x_fit(x, M, G, R) for x in xx]

    l2 = ax.plot(
        xx,
        lx,
        "--",
        c="gold",
        label=r"estimated $l_x$ function",
    )

    l1 = ax.plot(
        xx,
        empirical_lx,
        "o",
        c="darkkhaki",
        label=r"baboon population $l_x$",
    )

    ax.set_ylabel(r"Survival $(l_x)$", fontsize=18)
    ax.set_xlabel(r"age in years ($x$)", fontsize=18)
    ax.tick_params(labelsize=15)

    ax3 = ax.twinx()

    fertility = np.array(
        yellow_baboon_df["VertMammal"][1:29].array,
        dtype=float,
    )

    l3 = ax3.plot(
        xx,
        fertility,
        "^",
        c="goldenrod",
        label=r"baboon population $m_x$",
    )

    popt, _ = curve_fit(
        m_x_fit,
        xx,
        fertility,
        bounds=(0.00001, [2.0, 2.0, 2.0]),
    )

    alpha, beta, gamma = popt
    mx = [m_x_fit(x, alpha, beta, gamma) for x in xx]

    l4 = ax3.plot(
        xx,
        mx,
        c="darkgoldenrod",
        label=r"estimated $m_x$ function",
    )

    ax3.set_ylabel(r"Reproduction $(m_x)$", fontsize=18)
    ax3.set_xlabel(r"Age in years ($x$)", fontsize=18)
    ax3.tick_params(labelsize=15)

    lines = l1 + l2 + l3 + l4
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="lower center", fontsize=10)

    ax.set_title("Yellow baboon life-history", fontsize=14)


# SVG icon utilities
# ------------------
def svg_color_to_mpl(color):
    """Convert a basic SVG color specification to a Matplotlib color."""
    if color is None:
        return "none"

    color = color.strip()

    # Matplotlib does not interpret SVG gradients or pattern URLs here.
    if color.startswith("url("):
        return None

    if color.startswith("rgb"):
        nums = color[color.find("(") + 1 : color.find(")")]
        r, g, b = [int(x) for x in nums.split(",")]
        return r / 255, g / 255, b / 255

    return color


def load_svg_paths(svg_file):
    """Read SVG path geometry and color attributes from ``svg_file``."""
    tree = ET.parse(svg_file)
    root = tree.getroot()

    ns = {"svg": "http://www.w3.org/2000/svg"}

    paths = []

    for elem in root.findall(".//svg:path", ns):
        d = elem.attrib.get("d")

        if d is None:
            continue

        fill = svg_color_to_mpl(elem.attrib.get("fill", "#cccccc"))
        stroke = svg_color_to_mpl(elem.attrib.get("stroke", "black"))
        stroke_width = float(elem.attrib.get("stroke-width", 1))

        paths.append(
            {
                "path": parse_path(d),
                "fill": fill,
                "stroke": stroke,
                "stroke_width": stroke_width,
            }
        )

    return paths


def add_svg_icon(
    ax,
    svg_file,
    xy=(0.9, 0.9),
    scale=0.04,
    icon_size=120,
):
    """Draw an SVG icon inside a Matplotlib axis.

    ``xy`` is expressed in axis-fraction coordinates. ``scale`` controls the
    imported SVG path geometry, and ``icon_size`` sets the DrawingArea size.
    Stroke attributes are read for compatibility but the original plotting
    behavior renders fill only, with no path outlines.
    """
    svg_paths = load_svg_paths(svg_file)

    da = DrawingArea(icon_size, icon_size, clip=False)

    for item in svg_paths:
        mpl_path = item["path"]

        # SVG y-coordinates increase downward, opposite to Matplotlib.
        transform = (
            Affine2D()
            .scale(scale, -scale)
            .translate(icon_size / 2, icon_size / 2)
        )

        patch = PathPatch(
            transform.transform_path(mpl_path),
            facecolor=(
                item["fill"] if item["fill"] is not None else "none"
            ),
            edgecolor="none",
            lw=0,
            antialiased=True,
        )

        da.add_artist(patch)

    ab = AnnotationBbox(
        da,
        xy,
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=100,
    )

    ax.add_artist(ab)


def add_baboon_icon(
    ax,
    svg_file="baboon_true_vector.svg",
    xy=(0.785, 0.95),
    scale=0.045,
):
    """Add the baboon SVG icon to an axis."""
    add_svg_icon(
        ax=ax,
        svg_file=svg_file,
        xy=xy,
        scale=scale,
        icon_size=140,
    )


def add_guppy_icon(
    ax,
    svg_file="guppy_true_vector.svg",
    xy=(0.65, 1.05),
    scale=0.035,
):
    """Add the guppy SVG icon to an axis."""
    add_svg_icon(
        ax=ax,
        svg_file=svg_file,
        xy=xy,
        scale=scale,
        icon_size=140,
    )


def sep_fast_plot_6_panel(
    res,
    p_guppy: Params,
    p_baboon: Params,
    steps,
    par_ratios,
    classes,
    life_tables,
):
    """Create the separated six-panel host-reproductive-cost figure.

    This is the manuscript-oriented layout used for the first requested plot:
    two and twelve baseline age classes for each species, followed by the
    corresponding empirical life-history fit. With ``T_ratio=False``, the
    x-axis is the common symmetric transmission probability T_A = T_B.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 30))

    cmap = sns.color_palette("rocket", as_cmap=True)
    cmap.set_bad("lightgray")

    # Guppy heatmaps: the manuscript analysis compares two and twelve baseline
    # age classes while retaining alpha_A/alpha_B = 0.65.
    for ax, par_ratio, age_class_value in zip(
        axs.flat[:2],
        par_ratios,
        classes,
    ):
        if normalized_exclude(p_guppy) == "m":
            p_tmp = replace(
                p_guppy,
                age_classes=age_class_value,
                alpha_A=par_ratio * p_guppy.alpha_B,
            )
        elif normalized_exclude(p_guppy) == "mortality":
            p_tmp = replace(
                p_guppy,
                age_classes=age_class_value,
            )
            p_tmp = with_A_mortality_ratio(p_tmp, par_ratio)
        else:
            p_tmp = replace(
                p_guppy,
                age_classes=age_class_value,
            )

        plot_heatmap_panel(ax, res, p_tmp, steps, cmap)
        ax.set_title(f"{age_class_value} age classes", fontsize=14)

    plot_guppy_life_history(axs.flat[2], life_tables)
    
    # Yellow-baboon heatmaps use the same experimental contrasts as the guppy
    # row but the empirically fitted baboon life-history parameters.
    for ax, par_ratio, age_class_value in zip(
        axs.flat[3:5],
        par_ratios,
        classes,
    ):
        if normalized_exclude(p_baboon) == "m":
            p_tmp = replace(
                p_baboon,
                age_classes=age_class_value,
                alpha_A=par_ratio * p_baboon.alpha_B,
            )
        elif normalized_exclude(p_baboon) == "mortality":
            p_tmp = replace(
                p_baboon,
                age_classes=age_class_value,
            )
            p_tmp = with_A_mortality_ratio(p_tmp, par_ratio)
        else:
            p_tmp = replace(
                p_baboon,
                age_classes=age_class_value,
            )

        plot_heatmap_panel(ax, res, p_tmp, steps, cmap)
        ax.set_title(f"{age_class_value} age classes", fontsize=14)

    plot_baboon_life_history(axs.flat[5], life_tables)

    # Remove duplicate heatmap labels and replace them with shared labels.
    for idx in [0, 1, 3, 4]:
        axs.flat[idx].set_xlabel("")
        axs.flat[idx].set_ylabel("")

    if p_guppy.T_ratio:
        shared_x = r"Transmission ratio $(T_A/T_B)$"
    else:
        shared_x = r"Transmission rate $(T)$"

    if normalized_exclude(p_guppy) == "m":
        shared_y = mortality_benefit_axis_label(p_guppy)
    elif normalized_exclude(p_guppy) in ("mortality", "T"):
        shared_y = r"$\alpha_A/\alpha_B$"
    else:
        shared_y = ""

    fig.supxlabel(
        shared_x,
        fontsize=20,
        x=0.334,
        y=0.15,
    )

    fig.supylabel(
        shared_y,
        fontsize=20,
        x=0.01,
        y=0.6,
    )

    axs.flat[0].text(
        -0.05,
        1.1,
        "(a)",
        transform=axs.flat[0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    axs.flat[1].text(
        -0.05,
        1.1,
        "(b)",
        transform=axs.flat[1].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    axs.flat[2].text(
        -0.05,
        1.1,
        "(c)",
        transform=axs.flat[2].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    axs.flat[3].text(
        -0.05,
        1.1,
        "(d)",
        transform=axs.flat[3].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    axs.flat[4].text(
        -0.05,
        1.1,
        "(e)",
        transform=axs.flat[4].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    axs.flat[5].text(
        -0.05,
        1.1,
        "(f)",
        transform=axs.flat[5].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )

    # Draw the original thin separator between simulation/fitting columns.
    x_sep = (
        0.93 * axs[0, 1].get_position().x1
        + axs[0, 2].get_position().x0
    ) / 2

    fig.lines.append(
        plt.Line2D(
            [x_sep, x_sep],
            [0.08, 0.95],
            transform=fig.transFigure,
            color="black",
            linewidth=0.8,
            alpha=0.8,
        )
    )

    fig.tight_layout()

    # Shared continuous A-frequency color bar below the lower-left panel.
    top_left_ax = axs[1, 0]
    cax = inset_axes(
        top_left_ax,
        width="100%",
        height="15%",
        loc="lower left",
        bbox_to_anchor=(0, -0.7, 1, 1),
        bbox_transform=top_left_ax.transAxes,
        borderpad=0.5,
    )

    sm = plt.cm.ScalarMappable(
        cmap="rocket",
        norm=plt.Normalize(vmin=0, vmax=1),
    )

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
    )

    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_title(
        "$A$ frequency",
        fontsize=16,
        pad=1.2,
    )

    plt.subplots_adjust(
        wspace=0.4,
        hspace=0.4,
        bottom=0.25,
    )

    # The icon files are external plot assets. Update these paths when running
    # on another computer; the model results do not depend on the icons.
    for ax in axs[0, :]:
        add_guppy_icon(
            ax,
            svg_file=(
                r"svggenie-1779956596105.svg"
            ),
        )

    for ax in axs[1, :]:
        add_baboon_icon(
            ax,
            svg_file=(
                r"create-a-clean-high-quality-svg-vector-icon-of-a-s(1).svg"
            ),
        )

    return fig


# Six-panel vertical-transmission comparison
# ------------------------------------------
def fast_plot_6_panel_v_compare(
    res,
    p_guppy: Params,
    p_baboon: Params,
    steps,
    par_ratio,
    life_tables=None,
):
    """Create the six-panel vertical-transmission comparison figure.

    Columns compare two age classes with v = 0, two age classes with v = 1,
    and twelve age classes with v = 1. Rows correspond to guppies and yellow
    baboons. ``life_tables`` is retained for call compatibility but is not used
    by this all-heatmap layout.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    cmap = sns.color_palette("rocket", as_cmap=True)
    cmap.set_bad("lightgray")

    def build_params(base_p, age_classes, v_value):
        """Apply panel-specific age structure and vertical transmission."""
        if normalized_exclude(base_p) == "m":
            p_tmp = replace(
                base_p,
                age_classes=age_classes,
                v=v_value,
                alpha_A=par_ratio * base_p.alpha_B,
            )
        elif normalized_exclude(base_p) == "mortality":
            p_tmp = replace(
                base_p,
                age_classes=age_classes,
                v=v_value,
            )
            p_tmp = with_A_mortality_ratio(p_tmp, par_ratio)
        else:
            p_tmp = replace(
                base_p,
                age_classes=age_classes,
                v=v_value,
            )

        return p_tmp

    # Left to right: no vertical transmission, perfect vertical transmission,
    # and perfect vertical transmission with more interaction intervals.
    panel_defs = [
        (2, 0),
        (2, 1),
        (12, 1),
    ]

    for col, (age_classes, v_value) in enumerate(panel_defs):
        ax = axs[0, col]

        p_tmp = build_params(
            p_guppy,
            age_classes,
            v_value,
        )

        plot_heatmap_panel(
            ax,
            res,
            p_tmp,
            steps,
            cmap,
        )

        ax.set_title(
            f"{age_classes} age classes, $v={v_value}$",
            fontsize=16,
        )

    for col, (age_classes, v_value) in enumerate(panel_defs):
        ax = axs[1, col]

        p_tmp = build_params(
            p_baboon,
            age_classes,
            v_value,
        )

        plot_heatmap_panel(
            ax,
            res,
            p_tmp,
            steps,
            cmap,
        )

        ax.set_title(
            f"{age_classes} age classes, $v={v_value}$",
            fontsize=16,
        )

    # Replace repeated panel labels with one shared x- and y-axis label.
    for ax in axs.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")

    if p_guppy.T_ratio:
        shared_x = r"Transmission ratio $(T_A/T_B)$"
    else:
        shared_x = r"Transmission rate $(T)$"

    if normalized_exclude(p_guppy) == "m":
        shared_y = mortality_benefit_axis_label(p_guppy)
    elif normalized_exclude(p_guppy) in ("mortality", "T"):
        shared_y = r"$\alpha_A/\alpha_B$"
    else:
        shared_y = ""

    fig.supxlabel(
        shared_x,
        fontsize=20,
        y=0.08,
    )

    fig.supylabel(
        shared_y,
        fontsize=20,
        x=0.04,
    )

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax, label in zip(axs.flat, labels):
        ax.text(
            -0.08,
            1.2,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

    # The heatmaps remain continuous. The displayed legend is deliberately
    # reduced to three solid color blocks marking A frequencies 0, 0.5, and 1.
    cax = inset_axes(
        axs[1, 0],
        width="100%",
        height="15%",
        loc="lower left",
        bbox_to_anchor=(0, -0.6, 1, 1),
        bbox_transform=axs[1, 0].transAxes,
        borderpad=0.5,
    )

    rocket = plt.colormaps["rocket"]

    cmap3 = ListedColormap(
        [
            rocket(0.0),
            rocket(0.5),
            rocket(1.0),
        ]
    )

    dummy = np.array([[0, 1, 2]])

    im = cax.imshow(
        dummy,
        cmap=cmap3,
        aspect="auto",
        visible=False,
    )

    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )

    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["0", "0.5", "1"])
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_title(
        "$A$ frequency",
        fontsize=16,
        pad=1.2,
    )

    plt.subplots_adjust(
        wspace=0.4,
        hspace=0.3,
        bottom=0.25,
    )

    # External species icons. Update these paths on another workstation.
    for ax in axs[0, :]:
        add_guppy_icon(
            ax,
            svg_file=(
            r"svggenie-1779956596105.svg"
            ),
        )

    for ax in axs[1, :]:
        add_baboon_icon(
            ax,
            svg_file=(
            r"create-a-clean-high-quality-svg-vector-icon-of-a-s(1).svg"
            ),
        )

    return fig


#%% Parameter sets for the figures that are based on Guppies and Baboons life-history


guppy_params = Params(
    age_classes=12,
    v=1,
    T_A=0.2,
    T_B=0.2,
    N_A=100,
    N_B=100,
    alpha_A=20,
    alpha_B=20,
    beta_A=1.43,
    beta_B=1.43,
    gamma_A=1.6,
    gamma_B=1.6,
    G_A=0.02,
    G_B=0.02,
    R_A=1.84,
    R_B=1.84,
    M_A=0.00001,
    M_B=0.00001,
    T_ratio=True,
    mortality_parameter="R",
    mortality_benefit_max=1,
    exclude="m",
)


baboon_params = Params(
    age_classes=12,
    v=1,
    T_A=0.2,
    T_B=0.2,
    N_A=100,
    N_B=100,
    alpha_A=0.014,
    alpha_B=0.014,
    beta_A=2,
    beta_B=2,
    gamma_A=0.17,
    gamma_B=0.17,
    G_A=0.003,
    G_B=0.003,
    R_A=0.2,
    R_B=0.2,
    M_A=0.03,
    M_B=0.03,
    T_ratio=True,
    mortality_parameter="R",
    mortality_benefit_max=1,
    exclude="m",
)


#%% reccreate all three manuscript figures using a single function 


def create_manuscript_figures(
    life_tables,
    p_guppy: Params,
    p_baboon: Params,
    res=20,
    steps=500,
    show=False,
):
    """Create Fig. 1, Fig. 2, and the supplementary G-based Fig. S1.

    Fig. 1 uses symmetric absolute horizontal transmission, an R-mediated
    mortality benefit, alpha_A/alpha_B = 0.65, and two versus twelve baseline
    age classes, with empirical life-history fits in the right column.

    Fig. 2 uses transmission ratios, no reproductive cost, an R-mediated
    mortality benefit up to 1.0, and the vertical-transmission comparison
    (2 classes, v=0; 2 classes, v=1; 12 classes, v=1).

    Fig. S1 repeats Fig. 2 for the Gompertz scale parameter G and limits the
    relative mortality benefit to 0.95 for both species. This keeps G_A above
    zero and avoids the excessive death horizons produced at G_A = 0.

    The supplied baseline parameter objects are copied with ``replace`` and
    are not mutated. A dictionary of Matplotlib Figure objects is returned.
    """
    fig_1_guppy = replace(
        p_guppy,
        v=1,
        T_ratio=False,
        mortality_parameter="R",
        mortality_benefit_max=1.0,
        exclude="m",
    )
    fig_1_baboon = replace(
        p_baboon,
        v=1,
        T_ratio=False,
        mortality_parameter="R",
        mortality_benefit_max=1.0,
        exclude="m",
    )

    fig_1 = sep_fast_plot_6_panel(
        res=res,
        p_guppy=fig_1_guppy,
        p_baboon=fig_1_baboon,
        steps=steps,
        par_ratios=[0.65, 0.65],
        classes=[2, 12],
        life_tables=life_tables,
    )

    fig_2_guppy = replace(
        p_guppy,
        T_ratio=True,
        mortality_parameter="R",
        mortality_benefit_max=1.0,
        exclude="m",
    )
    fig_2_baboon = replace(
        p_baboon,
        T_ratio=True,
        mortality_parameter="R",
        mortality_benefit_max=1.0,
        exclude="m",
    )

    fig_2 = fast_plot_6_panel_v_compare(
        res=res,
        p_guppy=fig_2_guppy,
        p_baboon=fig_2_baboon,
        steps=steps,
        par_ratio=1.0,
    )

    fig_s1_guppy = replace(
        p_guppy,
        T_ratio=True,
        mortality_parameter="G",
        mortality_benefit_max=0.95,
        exclude="m",
    )
    fig_s1_baboon = replace(
        p_baboon,
        T_ratio=True,
        mortality_parameter="G",
        mortality_benefit_max=0.95,
        exclude="m",
    )

    fig_s1 = fast_plot_6_panel_v_compare(
        res=res,
        p_guppy=fig_s1_guppy,
        p_baboon=fig_s1_baboon,
        steps=steps,
        par_ratio=1.0,
    )

    figures = {
        "fig_1": fig_1,
        "fig_2": fig_2,
        "fig_s1": fig_s1,
    }

    if show:
        plt.show()

    return figures


#%% single-call reproduction of all three manuscript figures 
#   estimated time: 10 - 20 min.

    
figures = create_manuscript_figures(
    life_tables=life_tables,
    p_guppy=guppy_params,
    p_baboon=baboon_params,
    res=20,
    steps=500,
    show=True,
)


