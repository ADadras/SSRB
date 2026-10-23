"""Crossed resampling with linked patient/fundus clusters for new analyses."""
import numpy as np


def cluster_draws(sources, *, replicates=10000, seed=20260814):
    """Share source draws for LIDC/subset and RIGA disc/cup.

    The caller supplies one source ID per independent sampling unit. Model
    seeds are resampled jointly. Visible choices are averaged before calling.
    """
    if "lidc" in sources and "lidc_hard" in sources and not set(sources["lidc_hard"]).issubset(sources["lidc"]):
        raise ValueError("LIDC-hard sources must be a subset of LIDC")
    if "riga_disc" in sources and "riga_cup" in sources and set(sources["riga_disc"]) != set(sources["riga_cup"]):
        raise ValueError("RIGA disc/cup source sets differ")
    rng = np.random.default_rng(seed)
    groups = {}
    for endpoint, ids in sources.items():
        family = "lidc" if endpoint in {"lidc", "lidc_hard"} else "riga" if endpoint in {"riga_disc", "riga_cup"} else endpoint
        groups.setdefault(family, set()).update(ids)
    result = {}
    for family, values in sorted(groups.items()):
        ids = sorted(values)
        weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=replicates)
        for endpoint, local in sources.items():
            parent = "lidc" if endpoint in {"lidc", "lidc_hard"} else "riga" if endpoint in {"riga_disc", "riga_cup"} else endpoint
            if family == parent:
                # Subsets use the corresponding columns of their parent's draw.
                result[endpoint] = {s: weights[:, ids.index(s)] for s in local}
    optimizer = rng.integers(0,3,size=(replicates,3))
    return result, optimizer


def resample_means(sums, counts, source_ids, draws, optimizer):
    """Arrays are source x optimizer x method x metric (counts omit metric)."""
    result = []
    for i, seeds in enumerate(optimizer):
        weights = np.asarray([draws[s][i] for s in source_ids], dtype=float)
        numerator = sum(np.einsum('s,smq->mq', weights, sums[:, seed]) for seed in seeds)
        denominator = sum(np.einsum('s,sm->m', weights, counts[:, seed]) for seed in seeds)
        if (denominator <= 0).any():
            raise ValueError('empty subset in a bootstrap replicate; more source clusters are required')
        result.append(numerator/denominator[:,None])
    return np.asarray(result)
