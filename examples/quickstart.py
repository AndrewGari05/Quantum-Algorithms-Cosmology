"""Quick start from the README: a three-rung ablation ladder on a banana target."""
import numpy as np
from qablate import (MetropolisHastings, GaussianProposal, RandomCircuitProposal,
                     MetropolisAcceptance, AmplitudeEncodedMetropolis, Study, diagnostics)

def log_prob(x):                                   # banana-shaped target, vectorized
    return -0.5 * (x[:, 0] ** 2 + (x[:, 1] - x[:, 0] ** 2) ** 2 / 0.25)

def run(parts, rng):
    mh = MetropolisHastings(log_prob, ndim=2, nchains=6, rng=rng, **parts)
    mh.run_mcmc(np.zeros((6, 2)), 20000)
    return {"chain": mh.get_chain(discard=2000)}

study = Study(
    {"proposal":   (lambda: GaussianProposal(0.7), lambda: RandomCircuitProposal(0.7)),
     "acceptance": (MetropolisAcceptance, AmplitudeEncodedMetropolis)},
    run,
    equivalent=[({"proposal": True}, {"proposal": True, "acceptance": True})])
runs = study.run(study.ladder(), seed=42)          # C,C -> Q,C -> Q,Q from the same seed
print(study.check_equivalences(runs, ["chain"]))   # declared-identical pair: True
for r in runs:
    print(r.label, diagnostics.summary(np.swapaxes(r.outputs["chain"], 0, 1)))
