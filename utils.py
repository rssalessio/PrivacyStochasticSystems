#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Alessio Russo [alessior@kth.se]. All rights reserved.
#
# This file is part of PrivacyStochasticSystems.
#
# PrivacyStochasticSystems is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with PrivacyStochasticSystems.
# If not, see <https://opensource.org/licenses/MIT>.
#

import numpy as np
import scipy as sp


def sanity_check_probabilities(P0: np.ndarray,
                               P1: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Checks if the two transition probabilities are ok
    Parameters
    ----------
    P0, P1 : np.ndarray
        Numpy matrices containing the transition probabilities for model M0 and M1
        Each matrix should have dimensions |actions|x|states|x|states|

    Returns
    -------
    P0, P1 : np.ndarray
        Returns the original matrices
    """
    P0, P1 = np.array(P0), np.array(P1)
    if P0.shape != P1.shape:
        raise ValueError('P0 and P1 do not have the same shape.')
    elif len(P0.shape) != 3:
        raise ValueError('P0/P1 should have 3 dimensions.')
    elif P0.shape[1] != P0.shape[2]:
        raise ValueError(
            'The 2nd and 3rd dimension of P0 and P1 should be the same.')
    elif np.any(P0 < 0) or np.any(P0 > 1) or np.any(np.isnan(P0)):
        raise ValueError('P0 contains invalid values')
    elif np.any(P1 < 0) or np.any(P1 > 1) or np.any(np.isnan(P1)):
        raise ValueError('P1 contains invalid values')
    elif not np.all(np.isclose(np.sum(P0, axis=2), 1.)):
        raise ValueError('Probabilities in P0 do not sum to 1')
    elif not np.all(np.isclose(np.sum(P1, axis=2), 1.)):
        raise ValueError('Probabilities in P0 do not sum to 1')
    elif not np.all(np.isclose(P1[np.isclose(P0, 0.)], 0.)):
        print('[WARNING] P1 is not absolutely continuous with respect to P0')
    return P0, P1


def sanity_check_rewards(R0: np.ndarray,
                         R1: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Checks if the two reward matrices are ok
    Parameters
    ----------
    R0, R1 : np.ndarray
        Numpy matrices containing the rewards for model M0 and M1
        Each matrix should have dimensions |states|x|actions|

    Returns
    -------
    R0, R1 : np.ndarray
        Returns the original matrices
    """
    R0, R1 = np.array(R0), np.array(R1)

    if R1.shape != R0.shape:
        raise ValueError('R0 and R1 do not have the same shape')
    elif len(R1.shape) != 2:
        raise ValueError('R0/R1 should have 2 dimensions')

    ns = R0.shape[0]
    na = R0.shape[1]

    if R1.shape != (ns, na):
        raise ValueError('Shape of reward matrix should be ({},{})'.format(
            ns, na))
    return R0, R1


def compute_values(rho: float, xi0: np.ndarray, xi1: np.ndarray,
                   R0: np.ndarray, R1: np.ndarray) -> (float, float, float):
    """ Computes the value functions
    Parameters
    ----------
    rho : float
        Weight given to policy pi_1 (1-rho for policy pi_0)
    xi0, xi1 : np.ndarray
        Numpy matrices of dimensions |states|x|actions| containing the stationary distributions
        over states and actions of the two models (M0 and M1)
    R0, R1 : np.ndarray
        Numpy matrices containing the rewards for model M0 and M1
        Each matrix should have dimensions |states|x|actions|

    Returns
    -------
    V0, V1, V : np.ndarray
        Returns V_0(pi_0), V_1(pi_1) and V(rho, pi_0, pi_1)
    """

    if rho < 0 or rho > 1:
        raise ValueError('rho should be between [0,1]')
    R0, R1 = sanity_check_rewards(R0, R1)
    ns, na = R0.shape
    V0, V1 = np.sum(np.multiply(xi0, R0)), np.sum(np.multiply(xi1, R1))
    V = rho * V1 + (1 - rho) * V0
    return V0, V1, V


def compute_KL_divergence_models(P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    P0, P1 = sanity_check_probabilities(P0, P1)
    ns, na = P0.shape[1], P0.shape[0]

    # Compute KL divergences
    I = np.zeros((ns, na))
    for s in range(ns):
        for a in range(na):
            I[s, a] = np.sum(
                sp.special.kl_div(P1[a, s, :], P0[a, s, :]) + P1[a, s, :] -
                P0[a, s, :])

    return I


def compute_stationary_distribution(
        P: np.ndarray, pi: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Computes stationary distribution given the transition density matrix
        and the policy.
    Parameters
    ----------
    P : np.ndarray
        Numpy matrix containing the transition probabilities for the model
        The matrix should have dimensions |actions|x|states|x|states|
    pi : np.ndarray
        Numpy matrix of dimensions |states|x|actions| containing the
        policy probabilities

    Returns
    -------
    xi : np.ndarray
        Stationary state-action distribution
    mu : np.ndarray
        Stationary state distribution
    """
    P, _ = sanity_check_probabilities(P, P)
    na, ns = P.shape[0], P.shape[1]
    P_pi = build_markov_transition_density(P, pi)

    _, u = np.linalg.eig(P_pi)
    # 0 should be the index of the eigenvalue 1
    mu = np.abs(u[:, 0]) / np.sum(np.abs(u[:, 0]))
    xi = np.zeros((ns, na))

    for s in range(ns):
        xi[s, :] = pi[s, :] * mu[s]
    return xi, mu


def build_markov_transition_density(P: np.ndarray, pi: np.ndarray):
    """ Computes the transition density P^{pi}(x'|x) given a policy pi
    Parameters
    ----------
    P : np.ndarray
        Numpy matrix containing the transition probabilities for the model
        The matrix should have dimensions |actions|x|states|x|states|
    pi : np.ndarray
        Numpy matrix of dimensions |states|x|actions| containing the
        policy probabilities

    Returns
    -------
    P_pi : np.ndarray
        Transition matrix
    """
    P, _ = sanity_check_probabilities(P, P)
    na, ns = P.shape[0], P.shape[1]
    P_pi = np.zeros((ns, ns))
    for s in range(ns):
        for y in range(ns):
            P_pi[y, s] = np.dot(P[:, s, y], pi[s, :])
    return P_pi
