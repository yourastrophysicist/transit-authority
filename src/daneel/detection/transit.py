"""Transit light curves of WASP-12 b with batman (Assignment 1, Task F).

transit()  plots one light curve from one parameter set.
compare()  plots the light curves of every WASP-12 b parameter set in the
           NASA Exoplanet Archive that batman can use, with 1 sigma errors
           propagated from each paper's published uncertainties.

Parameters: Kokori et al. 2023, ApJS 265, 4 (NASA Exoplanet Archive).
Limb darkening: ExoCTK, ATLAS9, TESS band, Teff 6265 K, log g 4.11,
[Fe/H] 0.12 (stellar values from Leonardi et al. 2024).
"""
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import setuptools  # noqa: F401  # Provides distutils compatibility for batman.

batman = importlib.import_module("batman")

WASP12B = {
    "t0": 0.0,            # time of mid-transit [days]
    "per": 1.091419179,   # orbital period [days]
    "rp": 0.1178,         # planet radius [stellar radii]
    "a": 3.040,           # semi-major axis [stellar radii]
    "inc": 83.40,         # orbital inclination [deg]
    "ecc": 0.0,
    "w": 90.0,            # argument of periastron [deg]
    "u": [0.272, 0.257],  # quadratic limb darkening, ExoCTK TESS
    "limb_dark": "quadratic",
}

R_EARTH_PER_R_SUN = 109.076   # Earth radii in one solar radius
R_SUN_PER_AU = 215.032        # solar radii in one AU
ARCHIVE_CSV = Path(__file__).resolve().parents[3] / "data" / "wasp12b_nasa_archive.csv"


def transit(params=WASP12B, output="wasp-12b_assignment1_taskF.png",
            title="WASP-12 b (Kokori et al. 2023)"):
    """Plots the batman light curve for one parameter set and saves it."""
    p = batman.TransitParams()
    for key, value in params.items():
        setattr(p, key, value)
    t = np.linspace(-0.1, 0.1, 1000)   # days around mid-transit
    flux = batman.TransitModel(p, t).light_curve(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t * 24, flux)              # days to hours, easier to read
    ax.set_xlabel("Time from mid-transit [hours]")
    ax.set_ylabel("Relative flux")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    return t, flux


def compare(csv=ARCHIVE_CSV, output="wasp-12b_assignment1_taskF_comparison.png",
            n_draws=300, seed=42):
    """Compares the transit depth and duration of every usable archive
    parameter set, with 1 sigma error bars from Monte Carlo draws."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    data = pd.read_csv(csv)
    data["ref"] = data["pl_refname"].str.replace(r"<[^>]+>", "", regex=True).str.strip()
    data["ref"] = data["ref"].str.replace("&amp;", "&").str.replace("&Ouml;", "Ö").str.replace("&uuml;", "ü")

    # Fill Rp/R* and a/R* from each paper's own numbers when not given directly
    rp = data["pl_ratror"].fillna(data["pl_rade"] / R_EARTH_PER_R_SUN / data["st_rad"])
    a = data["pl_ratdor"].fillna(data["pl_orbsmax"] * R_SUN_PER_AU / data["st_rad"])
    usable = data["pl_orbper"].notna() & rp.notna() & a.notna() & data["pl_orbincl"].notna()
    print(f"{usable.sum()} of {len(data)} papers can make a light curve")

    def draw(value, err_up, err_down):
        """Split normal, so asymmetric +/- errors are respected.
        No error given means the value is held fixed."""
        if pd.isna(err_up) or pd.isna(err_down):
            return np.full(n_draws, value)
        z = rng.standard_normal(n_draws)
        return value + np.where(z > 0, z * err_up, z * abs(err_down))

    t = np.linspace(-0.1, 0.1, 2000)
    dt_hours = (t[1] - t[0]) * 24
    curves = {}
    for i in data.index[usable]:
        row = data.loc[i]
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc = 0.0, row["pl_orbper"], rp[i], a[i], row["pl_orbincl"]
        # eccentric orbit only if the paper gives both e (not an upper limit) and w
        if pd.notna(row["pl_orbeccen"]) and row["pl_orbeccenlim"] != 1 and pd.notna(row["pl_orblper"]):
            p.ecc, p.w = row["pl_orbeccen"], row["pl_orblper"]
        else:
            p.ecc, p.w = 0.0, 90.0
        p.u, p.limb_dark = WASP12B["u"], "quadratic"
        model = batman.TransitModel(p, t)
        flux = model.light_curve(p)

        # Only errors the archive gives directly are used: an Rp/R* or a/R*
        # derived from pl_rade or pl_orbsmax is held fixed.
        rp_err = (row["pl_ratrorerr1"], row["pl_ratrorerr2"]) if pd.notna(row["pl_ratror"]) else (np.nan, np.nan)
        a_err = (row["pl_ratdorerr1"], row["pl_ratdorerr2"]) if pd.notna(row["pl_ratdor"]) else (np.nan, np.nan)
        per_s = draw(row["pl_orbper"], row["pl_orbpererr1"], row["pl_orbpererr2"])
        rp_s, a_s = draw(rp[i], *rp_err), draw(a[i], *a_err)
        inc_s = draw(row["pl_orbincl"], row["pl_orbinclerr1"], row["pl_orbinclerr2"])
        inc_s = np.where(inc_s > 90, 180 - inc_s, inc_s)   # i and 180-i give the same transit
        draws = np.empty((n_draws, t.size))
        for k in range(n_draws):
            p.per, p.rp, p.a, p.inc = per_s[k], rp_s[k], a_s[k], inc_s[k]
            draws[k] = model.light_curve(p)
        curves[row["ref"]] = (flux, draws)

    # Two numbers per curve: depth at mid-transit and first-to-last-contact
    # duration, each with the 16-84 percentile range of the draws as 1 sigma.
    measures = (lambda f: 100 * (1 - f.min(axis=-1)), lambda f: (f < 1).sum(axis=-1) * dt_hours)
    summary = {}
    for ref, (flux, draws) in curves.items():
        summary[ref] = []
        for measure in measures:
            best = measure(flux)
            lo, hi = np.percentile(measure(draws), [16, 84])
            summary[ref].append((best, max(best - lo, 0), max(hi - best, 0)))

    refs = list(curves)
    newest = refs[-1]
    INK, GREY, HIGHLIGHT = "#4a4a4a", "#b8b8b8", "#d1495b"
    fig = plt.figure(figsize=(10, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.3, wspace=0.08)

    top = fig.add_subplot(grid[0, :])
    for ref in refs[:-1]:
        top.plot(t * 24, curves[ref][0], color=GREY, lw=1)
    top.plot(t * 24, curves[newest][0], color=HIGHLIGHT, lw=2, label=newest)
    top.plot([], [], color=GREY, lw=1, label=f"the other {len(refs) - 1} papers")
    top.set_xlabel("Time from mid-transit [hours]")
    top.set_ylabel("Relative flux")
    top.set_title(f"By eye, all {len(refs)} published models of WASP-12 b look almost the same", loc="left")
    top.legend(frameon=False, loc="lower right")

    rows = np.arange(len(refs))[::-1]   # oldest paper on top
    titles = ["How deep is the dip? [%]", "How long does the transit last? [hours]"]
    for col, title in enumerate(titles):
        ax = fig.add_subplot(grid[1, col])
        best, lo, hi = summary[newest][col]
        ax.axvspan(best - lo, best + hi, color=HIGHLIGHT, alpha=0.15, lw=0)
        ax.axvline(best, color=HIGHLIGHT, lw=1)
        for y, ref in zip(rows, refs):
            b, l, h = summary[ref][col]
            ax.errorbar(b, y, xerr=[[l], [h]], fmt="o", ms=6, capsize=3, lw=1.5,
                        color=HIGHLIGHT if ref == newest else INK)
        ax.set_yticks(rows)
        ax.set_yticklabels(refs if col == 0 else [])
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.text(0.5, 0.02,
             f"Dots are each paper's best values, bars are 1σ from its published errors. "
             f"Red band: {newest}.\nA bar that reaches the red band agrees with the newest paper.",
             ha="center", fontsize=9, color=INK)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.95, bottom=0.12)
    fig.savefig(output, dpi=150)
    return summary


if __name__ == "__main__":
    transit()
    compare()
