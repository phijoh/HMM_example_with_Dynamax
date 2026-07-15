import copy

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dynamax.hidden_markov_model import LogisticRegressionHMM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def set_HMM_params(params_dict, **kwargs):
    new_params_dict = copy.deepcopy(params_dict)
    for key in kwargs:
        if key in params_dict.keys():
            new_params_dict[key] = kwargs[key]
        else:
            raise Exception(f"{key} not in dict")
    return new_params_dict


def estimate_emission_params(session_df, HMM_specs):
    x = session_df[HMM_specs["covariates_emissions"]]
    y = session_df[HMM_specs["outputs"]].values.flatten()
    clf = LogisticRegression().fit(x, y)
    return (clf.coef_, clf.intercept_)


def reshape_priors(emission_weights_prior, num_states, num_classes=None):
    weights_init = np.tile(emission_weights_prior[0][:, :], (num_states, 1))
    biases_init = np.tile(emission_weights_prior[1], (num_states,))
    return weights_init, biases_init


def fit_HMM_dynamax(session_df, HMM_specs, emission_prior_weights, emission_prior_sigma=0.2, key=None):
    input_dim = len(HMM_specs["covariates_emissions"])

    inputs = session_df[HMM_specs["covariates_emissions"]].values
    inp = jnp.array(inputs, dtype=float)
    emissions = np.squeeze(session_df[HMM_specs["outputs"]])
    em = jnp.array(emissions, dtype=int)

    hmm = LogisticRegressionHMM(
        num_states=HMM_specs["n_states"],
        input_dim=input_dim,
        transition_matrix_stickiness=0.8,
    )

    HMM_inits = []
    HMM_inits_lps = []
    HMM_params_inits = []
    for _ in range(HMM_specs["n_inits"]):
        weights_init, biases_init = emission_prior_weights
        weights_noisy = jnp.array(np.random.normal(weights_init, scale=emission_prior_sigma))
        biases_noisy = jnp.array(np.random.normal(biases_init, scale=emission_prior_sigma))

        init_params, props = hmm.initialize(key=key, emission_weights=weights_noisy, emission_biases=biases_noisy)
        estimated_params, log_probs = hmm.fit_em(init_params, props, emissions=em, inputs=inp, num_iters=HMM_specs["n_iters"])

        HMM_inits.append(hmm)
        HMM_inits_lps.append(log_probs)
        HMM_params_inits.append(estimated_params)

    return HMM_inits, HMM_params_inits, HMM_inits_lps


def calculate_cv_bit_trial(ll_model, ll_0, n_trials):
    return ((ll_model - ll_0) / n_trials) / np.log(2)


def calculate_baseline_test_ll(train_y, test_y, C):
    _, train_class_totals = np.unique(train_y, return_counts=True)
    train_class_probs = train_class_totals / train_y.shape[0]
    _, test_class_totals = np.unique(test_y, return_counts=True)

    ll0 = 0
    for c in C:
        ll0 += test_class_totals[c] * np.log(train_class_probs[c])
    return ll0


def crossvalidate_HMM_dynamax(session_df, HMM_specs, emission_prior_weights, emission_prior_sigma=0.2, key=None):
    input_dim = len(HMM_specs["covariates_emissions"])

    hmm = LogisticRegressionHMM(
        num_states=HMM_specs["n_states"],
        input_dim=input_dim,
        transition_matrix_stickiness=0.8,
    )

    gkf = GroupKFold(n_splits=HMM_specs["n_folds"])
    cv_bit_trial_folds = []

    for train_idx, val_idx in gkf.split(session_df, groups=session_df["block"]):
        train_df = session_df.iloc[train_idx]
        val_df = session_df.iloc[val_idx]

        ll0 = calculate_baseline_test_ll(
            train_df[HMM_specs["outputs"]].values,
            val_df[HMM_specs["outputs"]].values,
            C=np.unique(train_df[HMM_specs["outputs"]]),
        )

        if emission_prior_weights is not None:
            fold_prior = estimate_emission_params(train_df, HMM_specs)
            weights_init, biases_init = reshape_priors(fold_prior, HMM_specs["n_states"])
        else:
            weights_init, biases_init = emission_prior_weights

        inputs = train_df[HMM_specs["covariates_emissions"]].values
        inp = jnp.array(inputs, dtype=float)
        emissions = np.squeeze(train_df[HMM_specs["outputs"]])
        em = jnp.array(emissions, dtype=int)

        HMM_inits = []
        HMM_inits_lps = []
        HMM_params_inits = []

        for _ in range(HMM_specs["n_inits"]):
            weights_noisy = jnp.array(np.random.normal(weights_init, scale=emission_prior_sigma))
            biases_noisy = jnp.array(np.random.normal(biases_init, scale=emission_prior_sigma))

            init_params, props = hmm.initialize(key=key, emission_weights=weights_noisy, emission_biases=biases_noisy)
            estimated_params, log_probs = hmm.fit_em(init_params, props, emissions=em, inputs=inp, num_iters=HMM_specs["n_iters"])

            HMM_inits.append(hmm)
            HMM_inits_lps.append(log_probs)
            HMM_params_inits.append(estimated_params)

        HMM_inits_final_ll = [lps[-1] for lps in HMM_inits_lps]
        best_model_ind = np.argmax(HMM_inits_final_ll)

        HMM_test_lps = HMM_inits[best_model_ind].marginal_log_prob(
            HMM_params_inits[best_model_ind],
            emissions=jnp.array(np.squeeze(val_df[HMM_specs["outputs"]].values), dtype=int),
            inputs=jnp.array(val_df[HMM_specs["covariates_emissions"]].values, dtype=float),
        )

        cv_bit_trial_folds.append(calculate_cv_bit_trial(HMM_test_lps, ll0, n_trials=len(val_idx)))

    return cv_bit_trial_folds


def get_state_order_dynamax(params):
    n_states = params.transitions[0].shape[0]
    stim_weight = [params.emissions.weights[i][0] for i in range(n_states)]
    state_order = np.argsort(stim_weight)
    return state_order


def relabel_inferred_states_dynamax(hmm, params, session_df, HMM_specs):
    state_order = get_state_order_dynamax(params)
    n_states = hmm.num_states

    inputs = session_df[HMM_specs["covariates_emissions"]].values
    inp = jnp.array(inputs, dtype=float)
    emissions = np.squeeze(session_df[HMM_specs["outputs"]])
    em = jnp.array(emissions, dtype=int)

    inferred_states = hmm.most_likely_states(params, em, inp)
    inferred_states_relabelled = np.select(
        [inferred_states == i for i in np.array(range(n_states))[state_order]],
        range(n_states),
        inferred_states,
    )
    return inferred_states_relabelled


def get_state_probs_dynamax(hmm, params, session_df, HMM_specs):
    inputs = session_df[HMM_specs["covariates_emissions"]].values
    inp = jnp.array(inputs, dtype=float)
    emissions = np.squeeze(session_df[HMM_specs["outputs"]])
    em = jnp.array(emissions, dtype=int)

    posterior = hmm.filter(params, em, inp)
    return posterior.filtered_probs


def get_state_params_dynamax(params):
    state_order = get_state_order_dynamax(params)

    weights = params.emissions.weights
    biases = params.emissions.biases
    betas = np.concatenate((np.expand_dims(biases, axis=1), weights), axis=1)
    betas = betas[state_order, :]

    transition_matrix = params.transitions.transition_matrix
    transition_matrix = transition_matrix[state_order, :][:, state_order]

    return betas, transition_matrix


def fit_glm_baseline(session_df, HMM_specs):
    x = session_df[HMM_specs["covariates_emissions"]]
    y = session_df[HMM_specs["outputs"]].values.flatten()
    glm = LogisticRegression().fit(x, y)

    p1 = np.clip(glm.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    return glm, ll


def crossvalidate_glm_baseline(session_df, HMM_specs):
    gkf = GroupKFold(n_splits=HMM_specs["n_folds"])
    fold_scores = []

    for train_idx, val_idx in gkf.split(session_df, groups=session_df["block"]):
        train_df = session_df.iloc[train_idx]
        val_df = session_df.iloc[val_idx]

        ll0 = calculate_baseline_test_ll(
            train_df[HMM_specs["outputs"]].values,
            val_df[HMM_specs["outputs"]].values,
            C=np.unique(train_df[HMM_specs["outputs"]]),
        )

        x_train = train_df[HMM_specs["covariates_emissions"]]
        y_train = train_df[HMM_specs["outputs"]].values.flatten()
        x_val = val_df[HMM_specs["covariates_emissions"]]
        y_val = val_df[HMM_specs["outputs"]].values.flatten()

        glm = LogisticRegression().fit(x_train, y_train)
        p1 = np.clip(glm.predict_proba(x_val)[:, 1], 1e-12, 1 - 1e-12)
        ll_model = np.sum(y_val * np.log(p1) + (1 - y_val) * np.log(1 - p1))

        fold_scores.append(calculate_cv_bit_trial(ll_model, ll0, n_trials=len(val_idx)))

    return fold_scores


def plot_state_psychometric_curves(best_params, best_specs, session_df, best_model_name, n_grid=200):
    betas, _ = get_state_params_dynamax(best_params)
    cov_names = best_specs["covariates_emissions"]
    n_states = betas.shape[0]

    x_min = session_df["sensory_evidence"].min()
    x_max = session_df["sensory_evidence"].max()
    x_grid = np.linspace(x_min, x_max, n_grid)

    has_prev = "prev_choice" in cov_names
    ncols = min(n_states, 3)
    nrows = int(np.ceil(n_states / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for s in range(n_states):
        ax = axes[s]
        beta = betas[s]

        if has_prev:
            for prev in [0, 1]:
                x_design = np.zeros((n_grid, len(cov_names)))
                for j, cov in enumerate(cov_names):
                    if cov == "sensory_evidence":
                        x_design[:, j] = x_grid
                    elif cov == "prev_choice":
                        x_design[:, j] = prev
                    else:
                        x_design[:, j] = session_df[cov].mean()

                logits = beta[0] + x_design @ beta[1:]
                p_right = 1 / (1 + np.exp(-logits))
                ax.plot(x_grid, p_right, label=f"prev_choice={prev}")
        else:
            x_design = np.zeros((n_grid, len(cov_names)))
            for j, cov in enumerate(cov_names):
                if cov == "sensory_evidence":
                    x_design[:, j] = x_grid
                else:
                    x_design[:, j] = session_df[cov].mean()

            logits = beta[0] + x_design @ beta[1:]
            p_right = 1 / (1 + np.exp(-logits))
            ax.plot(x_grid, p_right, label="model curve")

        ax.set_title(f"State {s + 1}")
        ax.set_xlabel("Sensory evidence")
        ax.set_ylabel("P(Response = 1)")
        ax.set_ylim(0, 1)
        ax.legend()

    for k in range(n_states, len(axes)):
        axes[k].axis("off")

    plt.suptitle(f"State-specific psychometric curves ({best_model_name})", y=1.02)
    plt.tight_layout()
    plt.show()
