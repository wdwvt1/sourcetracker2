import pandas as pd
import numpy as np
import scipy.spatial


def loo_mixing(n):
    return np.where(np.eye(n), 0, 1. / (n - 1))

def perfect_dirichlet_mixing(n, dirichlet_alpha):
    return np.random.dirichlet(np.repeat(dirichlet_alpha, n))

def polluted_dirichlet_mixing(n, f_pollution, dirichlet_alpha):
    tmp = (1 - f_pollution) * perfect_dirichlet_mixing(n, dirichlet_alpha)
    return np.append(tmp, f_pollution)

def mixing_proportions_to_sink(probabilities, mps, depth):
    n_sources, n_features = probabilities.shape
    sink = np.zeros(n_features, dtype=np.int32)
    for idx in range(n_sources):
        sink += np.random.multinomial(mps[idx] * depth, probabilities[idx])
    return sink

def default_mixtures(sources_df, depth=100, np_seed=None):
    ''''''
    sources = sources_df.values

    if np_seed is None:
        np_seed = np.random.seed()
    else:
        pass

    n_sources, n_features = sources.shape

    per_mps = loo_mixing(n_sources)

    dir_alphas = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
                           1.0])
    dir_mps = np.vstack([perfect_dirichlet_mixing(n_sources, da) for da in
                         dir_alphas]).T

    # Set the final 10 samples with increasing amounts of noise. All these
    # samples will have contributions based on a dirichlet draw with alpha=0.5
    # from all the sources. The first sample will have 5% mass from a random
    # unseen source, the second will have 10% mass, etc.
    frac_polluted = np.linspace(0.05, 0.50, 10)
    da = 0.5
    pol_mps = np.vstack([polluted_dirichlet_mixing(n_sources, fp, da) for fp in
                         frac_polluted]).T

    tmp = np.hstack((per_mps, dir_mps))
    tmp = np.vstack((tmp, np.zeros(tmp.shape[1])))
    mixing_proportions = np.hstack((tmp, pol_mps))


    # Calculate the probabilities for a multinomial draw from each source. We
    # have an additional source which is our pollution source in the default.
    probabilities = sources / np.expand_dims(sources.sum(1), 1)
    polluting_source = np.random.dirichlet(np.repeat(0.5, n_features))
    probabilities = np.vstack((probabilities, polluting_source))

    n_sinks = mixing_proportions.shape[1]
    sinks = np.zeros((n_sinks, n_features))

    for idx in range(n_sinks):
        sinks[idx] = mixing_proportions_to_sink(probabilities,
                                                mixing_proportions[:, idx],
                                                depth)

    sim_sinks = ['sm_%s' % str(i).zfill(3) for i in range(sinks.shape[0])]
    sinks = pd.DataFrame(sinks, index=sim_sinks, columns=sources_df.columns)

    final_sources = list(sources_df.index) + ['polluted_source']
    mixing_proportions = pd.DataFrame(mixing_proportions, index=final_sources,
                                      columns=sim_sinks)

    return (probabilities, mixing_proportions, sinks, np_seed)


def in_silico_mixing_formatter(exp_mpm, obs_mpms, a1s, a2s):
    ''''''
    exp_mpm.loc['Unknown'] = 0.0
    results = []
    for obs_mpm, a1, a2 in zip(obs_mpms, a1s, a2s):
        obs_mpm['polluted_source'] = 0.0
        for sample in obs_mpm.index:
            d = scipy.spatial.distance.euclidean(exp_mpm[sample],
                                                 obs_mpm.loc[sample])
            results.append([sample, a1, a2, d] + list(exp_mpm[sample]) +
                           list(obs_mpm.loc[sample]))
    columns = ['sample', 'alpha1', 'alpha2', 'd']

    columns += list(map(lambda x: x + '_added', exp_mpm.index))
    columns += list(map(lambda x: x + '_recovered', obs_mpm.columns))
    return pd.DataFrame(results, columns=columns)

# Example:
#  sourcetracker2 simmix -i /Users/wdwvt/biota/sourcetracker2/data/tiny-test/otu_table.biom -o ~/Desktop/simmix_test --burnin 2 --cluster_start_delay 10  --jobs 6 --restarts 1




