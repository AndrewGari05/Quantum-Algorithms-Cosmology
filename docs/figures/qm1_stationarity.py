"""Figure: the random-circuit proposal before and after the QM-1 fix.

Target: N(0, 1) in 2 dimensions, step 0.2 sigma, 6 chains x 4000 steps,
seed 7, statevector route (the thesis default). "Before" reproduces the
v0.8 calibration (subtract the empirical mean of 1024 calibration draws,
no random sign); "after" is qablate's RandomCircuitProposal.

    python docs/figures/qm1_stationarity.py   ->  docs/figures/qm1_stationarity.{png,pdf}
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from qablate import MetropolisHastings, RandomCircuitProposal  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"


class LegacyCalibration(RandomCircuitProposal):
    """v0.8 behaviour, for this figure only: empirical mean subtracted, no sign."""

    def _setup(self, ndim, rng):
        super()._setup(ndim, rng)
        raw = self._raw(self.n_calibration, rng)
        self.mu, self.rms = raw.mean(axis=0), raw.std(axis=0)

    def increments(self, n, rng):
        out = []
        while len(out) < n:
            if not self._queue:
                self._queue = list((self._raw(self.block_size, rng) - self.mu) / self.rms)
            out.append(self._queue.pop())
        return np.array(out)


def run(prop):
    mh = MetropolisHastings(lambda x: -0.5 * np.sum(x ** 2, axis=1), 2, 6, proposal=prop, rng=7)
    mh.run_mcmc(np.zeros((6, 2)), 4500)
    return mh.get_chain(discard=500)


def main():
    step = 0.2
    chains = {"before (v0.8 calibration)": run(LegacyCalibration(step, route="statevector")),
              "after (qablate)": run(RandomCircuitProposal(step, route="statevector"))}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    x = np.linspace(-4, 4, 400)
    for ax, (label, ch), color in zip(axes, chains.items(), (ORANGE, BLUE), strict=True):
        s = ch[:, :, 0].ravel()
        p = stats.kstest(ch[::50, :, 0].ravel(), "norm").pvalue
        ax.hist(s, bins=60, range=(-4, 4), density=True, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5)
        ax.plot(x, stats.norm.pdf(x), color=INK, lw=2, label="target N(0, 1)")
        ax.axvline(s.mean(), color=INK, lw=1, ls="--")
        ax.set_title(f"{label}\nmean = {s.mean():+.3f},  KS p = {p:.2g}", fontsize=10, color=INK)
        ax.set_xlabel(r"$x_0$", color=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("density", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"qm1_stationarity.{ext}"), dpi=160)


if __name__ == "__main__":
    main()
