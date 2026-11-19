"""Draw the seven figure files of the paper at their printed size from the data in ../data.

The figure sizes are the physical sizes in the two-column APS layout (column
3.375 in, page 7.0 in), so that the font sizes below are the sizes read on paper.

    python3 figures/replot.py              # all figures, written to paper_figures_redrawn/
    python3 figures/replot.py --only fig03 # one figure

Each data file is produced by the script named in README.md.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COL, PAGE = 3.375, 7.0
BLUE, ORANGE, GREEN, RED, GREY = "tab:blue", "tab:orange", "tab:green", "tab:red", "0.45"
SHELL = "#00e5ff"          # cyan shell: visible on the dark colormap
CMAP = "inferno"


def style():
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "lines.linewidth": 1.2, "lines.markersize": 3.0,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "mathtext.fontset": "cm", "font.family": "serif",
    })


def tag(axis, text, x=0.02, y=0.96, color="black"):
    axis.text(x, y, text, transform=axis.transAxes, ha="left", va="top",
              fontsize=9, fontweight="bold", color=color)


def pi_ticks(axis, which="y"):
    ticks, labels = [-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"]
    if which in ("y", "both"):
        axis.set_yticks(ticks); axis.set_yticklabels(labels)
    if which in ("x", "both"):
        axis.set_xticks(ticks); axis.set_xticklabels(labels)


# --------------------------------------------------------------------------
def fig01(c, out):
    maps, xs = c["maps"], c["coordinates"]
    ext = [xs[0], xs[-1], xs[0], xs[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(COL, 1.25), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    vmax = maps.max()
    for axis, m, t in zip(axes, maps, (r"$H=H_2$", r"$H_1$", r"$H'=H_0$")):
        im = axis.imshow(m.T, extent=ext, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(t, pad=2)
        axis.set_xlabel(r"$x$", labelpad=1)
        axis.set_xticks([-8, 0, 8])
    axes[0].set_ylabel(r"$y$", labelpad=1); axes[0].set_yticks([-8, 0, 8])
    bar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("LDOS", labelpad=2); bar.ax.tick_params(labelsize=6.5)
    fig.savefig(out)


def fig02a(c, out):
    grid, shell = c["grid"], c["shell"]
    lo, hi = grid.min(axis=0), grid.max(axis=0)
    inside = np.all((shell.T >= lo - 1e-9) & (shell.T <= hi + 1e-9), axis=1)
    shell = shell[:, inside]
    fig = plt.figure(figsize=(4.1, 2.3))
    for i, (d, t, sh) in enumerate(((c["density_h"], r"$H$", None),
                                    (c["density_l"], r"$H_1$", shell))):
        axis = fig.add_subplot(1, 2, i + 1, projection="3d")
        mask = d > 0.02
        axis.scatter(grid[mask, 0], grid[mask, 1], grid[mask, 2], c=d[mask], cmap=CMAP,
                     s=10 * d[mask], alpha=0.9, linewidths=0, vmin=0, vmax=1, depthshade=False)
        if sh is not None:
            axis.plot(*sh, color="red", linewidth=1.4)
        axis.set_xlim(lo[0], hi[0]); axis.set_ylim(lo[1], hi[1]); axis.set_zlim(lo[2], hi[2])
        axis.set_box_aspect([1, 1, 1], zoom=0.92)
        axis.set_title(t, pad=-2)
        for lab, f in ((r"$x$", axis.set_xlabel), (r"$y$", axis.set_ylabel), (r"$z$", axis.set_zlabel)):
            f(lab, labelpad=-10)
        axis.tick_params(labelsize=6, pad=-4)
        for a in (axis.xaxis, axis.yaxis, axis.zaxis):
            a.set_ticks([-5, 0, 5])
            a.pane.set_facecolor((0.96, 0.96, 0.96, 1.0)); a.pane.set_edgecolor("0.7")
            a._axinfo["grid"].update(color="0.85", linewidth=0.4)
        axis.text2D(0.02, 0.98, "(a)" if i == 0 else "(b)", transform=axis.transAxes,
                    ha="left", va="top", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0, right=0.97, bottom=0.02, top=0.95, wspace=0.12)
    fig.savefig(out)


def fig02b(clean, dirty, out):
    fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.0), sharex=True,
                             gridspec_kw={"hspace": 0.08, "height_ratios": [1.1, 1]})
    m = clean["m"]; W = float(dirty["disorder"])
    ax = axes[0]
    ax.plot(m, clean["z"] - 0.04, "-", drawstyle="steps-mid", color=BLUE, label=r"$\mathbb{Z}$: $\mathrm{Sig}/2$")
    ax.plot(m, clean["z2"] + 0.04, "-", drawstyle="steps-mid", color=ORANGE, label=r"$\mathbb{Z}_2$: $\mathrm{sgn\,Pf}\,A$")
    ax.plot(dirty["m"], dirty["z"] - 0.12, "--", drawstyle="steps-mid", color=BLUE, lw=1.0)
    ax.plot(dirty["m"], dirty["z2"] + 0.12, "--", drawstyle="steps-mid", color=ORANGE, lw=1.0)
    ax.plot([], [], "k-", label="clean"); ax.plot([], [], "k--", lw=1.0, label=rf"$W={W:g}$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_yticks([-1, 0, 1]); ax.set_ylim(-1.35, 1.35); ax.set_ylabel("index")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, 0.98), columnspacing=0.7, handlelength=1.2, handletextpad=0.4, borderaxespad=0.1, fontsize=6.3)
    ax = axes[1]
    ax.semilogy(m, clean["zgap"], "-", color="black", label="clean")
    ax.semilogy(dirty["m"], dirty["zgap"], "--", color="black", lw=1.0, label=rf"$W={W:g}$")
    ax.set_ylabel("gap"); ax.set_xlabel(r"$M/t$", labelpad=1)
    ax.set_xlim(m.min(), m.max())
    for a, t, xy in zip(axes, ("(c)", "(d)"), ((0.9, 0.3), (0.015, 0.96))):
        a.axvspan(1, 3, color=GREEN, alpha=0.1, lw=0); tag(a, t, x=xy[0], y=xy[1])
    fig.align_ylabels(axes)
    fig.savefig(out)


def fig03_full(cg, cd, ci, out, cm=None, layers=(2, 3), graphene_offset=None):
    """Graphene and four-cone Dirac maps, plus the indices, on the full page width."""
    ML = {1: ("#08306b", "-"), 2: ("#2171b5", "-"), 3: ("#6baed6", "-"), 4: ("#9ecae1", "-")}
    fig = plt.figure(figsize=(PAGE, 2.75))
    outer = fig.add_gridspec(1, 4, width_ratios=[4.0, 0.1, 0.42, 2.2], wspace=0.05,
                             left=0.04, right=0.99, bottom=0.12, top=0.86)
    maps = outer[0].subgridspec(2, 4, hspace=0.42, wspace=0.22)
    im = None
    letters = iter("abcdefgh")
    for col, (c, title) in enumerate(((cg, r"graphene, $H$"), (cg, r"graphene, $H_1$"),
                                       (cd, r"Dirac, $H$"), (cd, r"Dirac, $H_1$"))):
        lx, ly = int(c["lx"]), int(c["ly"])
        d = c["density_h"] if col % 2 == 0 else c["density_l"]
        w = c["weight_h"] if col % 2 == 0 else c["weight_l"]
        top = fig.add_subplot(maps[0, col]); bot = fig.add_subplot(maps[1, col])
        top.imshow((d / d.max()).reshape(lx, ly).T, origin="lower",
                   extent=[-lx / 2, lx / 2, -ly / 2, ly / 2], cmap=CMAP, vmin=0, vmax=1)
        if graphene_offset is not None and col < 2:
            # window [0, 2pi] x [-2pi, 0]: both valleys and the whole shell around K, unbroken
            nk = w.shape[0] - 1
            ww = np.roll(np.roll(w[:-1, :-1], -(nk // 2), axis=0), -(nk // 2), axis=1)
            ww = np.concatenate([ww, ww[:1]], axis=0); ww = np.concatenate([ww, ww[:, :1]], axis=1)
            im = bot.imshow((ww / ww.max()).T, origin="lower", extent=[0, 2 * np.pi, -2 * np.pi, 0],
                            cmap=CMAP, vmin=0, vmax=1)
            kxx, kyy = np.meshgrid(np.linspace(0, 2 * np.pi, 241), np.linspace(-2 * np.pi, 0, 241), indexing="ij")
            K = (2 * np.pi / 3, -2 * np.pi / 3)
            bot.contour(kxx, kyy, graphene_offset - 2 * np.cos(kxx - K[0]) - 2 * np.cos(kyy - K[1]),
                        levels=[0], colors=SHELL, linewidths=0.9)
        else:
            im = bot.imshow((w / w.max()).T, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi],
                            cmap=CMAP, vmin=0, vmax=1)
            bot.contour(c["kx"], c["ky"], c["shell"], levels=[0], colors=SHELL, linewidths=0.9)
        top.set_title(title, pad=3, fontsize=8)
        tk = [-8, 0, 8] if lx < 25 else [-12, 0, 12]
        top.set_xticks(tk); top.set_yticks(tk)
        for a in (top, bot):
            a.tick_params(labelsize=6.5, pad=1.5)
        top.set_xlabel(r"$x$", labelpad=0); bot.set_xlabel(r"$k_x$", labelpad=0)
        pi_ticks(bot, "both")
        shifted = graphene_offset is not None and col < 2
        if shifted:
            bot.set_xticks([0, np.pi, 2 * np.pi]); bot.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
            bot.set_yticks([-2 * np.pi, -np.pi, 0]); bot.set_yticklabels([r"$-2\pi$", r"$-\pi$", r"$0$"])
            bot.set_xlim(0, 2 * np.pi); bot.set_ylim(-2 * np.pi, 0)
        if col == 0:
            top.set_ylabel(r"$y$", labelpad=0); bot.set_ylabel(r"$k_y$", labelpad=-3)
        elif col == 2 and graphene_offset is not None:
            top.tick_params(labelleft=False)
        else:
            top.tick_params(labelleft=False); bot.tick_params(labelleft=False)
        tag(top, "(" + next(letters) + ")", color="black" if col % 2 == 0 else "white", x=0.04, y=0.95)
        tag(bot, "(" + next(letters) + ")", color="white", x=0.04, y=0.95)
    cax = fig.add_subplot(outer[1])
    bar = fig.colorbar(im, cax=cax); bar.set_ticks([0, 0.5, 1]); bar.ax.tick_params(labelsize=6.5)
    right = outer[3].subgridspec(2, 1, hspace=0.08, height_ratios=[1.45, 1])
    ax0 = fig.add_subplot(right[0]); ax1 = fig.add_subplot(right[1], sharex=ax0)
    lam = ci["lambda"]
    ax0.plot(lam, ci["chiral_index"] - 0.05, "-", drawstyle="steps-mid", color=ML[1][0], label=r"graphene, $n=1$: $\mathrm{Sig}/2$")
    if cm is not None:
        for n in layers:
            col, ls = ML[n]
            ax0.plot(cm["lambda"], cm[f"index_{n}"] - 0.05 - 0.05 * (n - 1), linestyle=ls, drawstyle="steps-mid", color=col,
                     label=rf"graphene, $n={n}$")
    ax0.plot(lam, ci["dirac_chiral_index"], "-", drawstyle="steps-mid", color=ORANGE, label=r"Dirac, $C$: $\mathrm{Sig}/2$")
    ax0.plot(lam, ci["det_sign"] + 0.05, "-", drawstyle="steps-mid", color=GREEN, label=r"Dirac, $T$: $\mathrm{sgn}\,\det A$")
    ax0.axhline(0, color="black", lw=0.5)
    nmax = max(layers) if cm is not None else 1
    top = max(1, int(np.max(cm[f"index_{max(layers)}"]))) if cm is not None else 1
    ax0.set_yticks(list(range(-nmax, top + 1))); ax0.set_ylim(-nmax - 0.35, top + 0.3)
    ax0.set_ylabel("index", labelpad=1); ax0.tick_params(labelbottom=False)
    ax0.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.45, 0.98), columnspacing=0.6,
               handlelength=1.1, handletextpad=0.3, borderaxespad=0.1, fontsize=6.2)
    ax1.semilogy(lam, ci["chiral_gap"], "-", color=ML[1][0])
    ax1.semilogy(lam, ci["dirac_chiral_gap"], "-", color=ORANGE)
    ax1.semilogy(lam, ci["tr_gap"], "-", color=GREEN)
    if cm is not None:
        for n in layers:
            col, ls = ML[n]
            ax1.semilogy(cm["lambda"], cm[f"gap_{n}"], linestyle=ls, color=col)
    ax1.set_ylabel("gap", labelpad=1); ax1.set_xlabel(r"$\lambda$", labelpad=0)
    ax1.set_xlim(lam.min(), lam.max())
    tag(ax0, "(i)", x=0.88, y=0.3); tag(ax1, "(j)", x=0.88, y=0.3)
    fig.savefig(out)


def wigner_and_index(c, out, shell_fn, index, ylabel, yticks, ylim, tags=("(a)", "(b)"), tag_xy=(0.015, 0.96), klabel=r"k"):
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.8),
                             gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.5})
    x, k, w = c["x"], c["k"], c["w"]
    ax = axes[0]
    im = ax.imshow(w.T, origin="lower", aspect="auto", extent=[x[0], x[-1], k[0], np.pi],
                   cmap=CMAP, vmin=0, vmax=float(np.quantile(w, 0.999)), interpolation="bicubic")
    xx, kk = np.meshgrid(x, k, indexing="ij")
    ax.contour(xx, kk, shell_fn(xx, kk), levels=[0], colors=SHELL, linewidths=1.1)
    ax.set_xlabel(r"$x$", labelpad=1); ax.set_ylabel(rf"${klabel}$", labelpad=-2); pi_ticks(ax)
    bar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05); bar.set_label(rf"$\mathrm{{Tr}}\,|W(x,{klabel})|$", labelpad=2)
    bar.ax.tick_params(labelsize=6.5)
    tag(ax, tags[0], color="white")
    ax = axes[1]
    ax.plot(c["lambdas"], index, "-", drawstyle="steps-mid", color=BLUE)
    ax.plot(c["lambdas"], index, "o", color=BLUE, ms=2.2)
    ax.set_xlabel(r"$\lambda$", labelpad=1); ax.set_ylabel(ylabel)
    ax.set_yticks(yticks); ax.set_ylim(*ylim); ax.grid(alpha=0.25, lw=0.5)
    # keep the index panel the same width as the image (colorbar steals space above)
    pos0, pos1 = axes[0].get_position(), axes[1].get_position()
    axes[1].set_position([pos0.x0, pos1.y0, pos0.width, pos1.height])
    tag(ax, tags[1], x=tag_xy[0], y=tag_xy[1])
    fig.savefig(out)


def fig04(c, out):
    x0, cc = float(c["x0"]), float(c["c"])
    # shell g_2(x, k_x) = (x/x0)^2 - 2 sin k_x - 1, the symbol of G_2 = (X/x0)^2 + i(T_x^dag - T_x) - 1
    wigner_and_index(c, out, lambda X, K: (X / x0) ** 2 - 2 * np.sin(K) - cc,
                     c["indices"], r"$\mathrm{Sig}(H')/2$", [0, 1], (-0.25, 1.25), klabel="k_x")


def fig05(c, out):
    x0 = float(c["x0"])
    wigner_and_index(c, out, lambda X, K: (X / x0) ** 2 - 2 * np.cos(K),
                     c["signs"], r"$\mathrm{sgn}\,\det A$", [-1, 1], (-1.35, 1.35), tag_xy=(0.9, 0.3))


def fig06(c, out, floor=1e-14):
    m0, v = float(c["mass"]), float(c["pairing"])
    L, E0, E1, Lr, E1r = c["lengths"], c["zero"], c["first"], c["ribbon_lengths"], c["ribbon_first"]
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.45), gridspec_kw={"hspace": 0.55})
    ax = axes[0]
    ax.loglog(L, E1, "o", color=BLUE, ms=2.0, label="Kitaev chain")
    ax.loglog(Lr, E1r, "s", color=RED, ms=2.0, mfc="none", mew=0.6, label="graphene ribbon")
    ls = np.geomspace(L.min(), L.max(), 50)
    ax.loglog(ls, np.sqrt(2 * v * m0 / ls), "k-", lw=0.9, label=r"$\sqrt{2vm_0/L_0}$")
    ax.set_xlabel(r"$L_0$", labelpad=1); ax.set_ylabel(r"$E_1$")
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_yticks([0.1, 0.2, 0.3, 0.5, 0.8])
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.set_ylim(0.1, 1.0); ax.legend(loc="lower left", fontsize=6.3, handlelength=1.2, borderaxespad=0.2)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    tag(ax, "(a)", x=0.9)
    ax = axes[1]
    good = E0 > floor; lmax = L[good].max() + 6; sel = L <= lmax
    slope, intercept = np.polyfit(L[good], np.log(E0[good]), 1)   # |E_0| ~ exp(-L_0 / 1.49)
    ax.semilogy(L[sel & good], E0[sel & good], "o", color=BLUE, ms=2.2, label=r"$|E_0|$, Kitaev chain")
    ax.semilogy(L[sel], np.exp(intercept + slope * L[sel]), "k-", lw=0.9, label="exponential fit")
    ax.axhline(floor, color=GREY, lw=0.8, ls=":")
    ax.text(L[sel][0] + 1, floor * 2.0, "machine precision", ha="left", va="bottom", fontsize=6.5, color=GREY)
    ax.set_ylim(floor / 5, 1e2); ax.set_xlim(L[sel][0] - 1, L[sel][-1] + 1)
    ax.set_xlabel(r"$L_0$", labelpad=1); ax.set_ylabel(r"$|E_0|$")
    ax.legend(loc="upper right", fontsize=6.3, handlelength=1.2, borderaxespad=0.2); ax.grid(alpha=0.25, lw=0.4)
    ax.set_yticks([1e-14, 1e-10, 1e-6, 1e-2])
    tag(ax, "(b)", x=0.02, y=0.42)
    fig.savefig(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "paper_figures_redrawn"))
    p.add_argument("--only", nargs="*", help="fig01 fig02a fig02b fig03 fig04 fig05 fig06")
    a = p.parse_args()
    style()
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    load = lambda name: np.load(data / name)
    jobs = {
        "fig01": lambda: fig01(load("fig01.npz"), out / "qwz_reduction.png"),
        "fig02a": lambda: fig02a(load("fig02a.npz"), out / "LDOS3DHOTI.png"),
        "fig02b": lambda: fig02b(load("fig02b_L20_clean.npz"), load("fig02b_L20_W1.npz"), out / "higherHOTIlocalizer.png"),
        "fig03": lambda: fig03_full(load("fig03a_graphene.npz"), load("fig03a_dirac.npz"), load("fig03_indices.npz"),
                                    out / "semimetal_full.png", cm=load("fig03_multilayer.npz"), graphene_offset=2.0),
        "fig04": lambda: fig04(load("fig04.npz"), out / "valley_hall_zero_mode_signature.png"),
        "fig05": lambda: fig05(load("fig05.npz"), out / "kitaev_valley_trim_wigner_shell.png"),
        "fig06": lambda: fig06(load("fig06.npz"), out / "volkov_pankratov_scaling.png"),
    }
    for name, job in jobs.items():
        if a.only and name not in a.only:
            continue
        try:
            job(); print("saved", name)
        except FileNotFoundError as error:
            print("skip", name, "(missing data file:", error.filename, ")")


if __name__ == "__main__":
    main()
