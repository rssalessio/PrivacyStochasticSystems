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
import cvxpy as cp
import scipy as sp
import dccp
from utils import sanity_check_probabilities, sanity_check_rewards, \
    compute_KL_divergence_models

eps = 1e-15


def full_information_privacy(P0: np.ndarray, P1: np.ndarray, xi0: np.ndarray,
                             xi1: np.ndarray) -> float:
    """ Computes 1/I_F(pi_0, pi_1) given xi_0 and xi_1
    Parameters
    ----------
    P0, P1 : np.ndarray
        Numpy matrices containing the transition probabilities for model M0 and M1
        Each matrix should have dimensions |actions|x|states|x|states|
    xi0, xi1 : np.ndarray
        Numpy matrices of dimensions |states|x|actions| containing the stationary distributions
        over states and actions of the two models (M0 and M1)

    Returns
    -------
    1/I_F : float
        Privacy level
    """
    P0, P1 = sanity_check_probabilities(P0, P1)
    xi0, xi1 = np.array(xi0), np.array(xi1)
    na, ns = P0.shape[0], P0.shape[1]

    I = compute_KL_divergence_models(P0, P1)

    privacy = np.sum(np.multiply(xi1, I))
    for s in range(ns):
        mu1_s = np.sum(
            xi1[s, :]) if not np.isclose(np.sum(xi1[s, :]), 0) else 1.
        mu0_s = np.sum(
            xi0[s, :]) if not np.isclose(np.sum(xi0[s, :]), 0) else 1.
        pi1_s = xi1[s, :] / mu1_s
        pi0_s = xi0[s, :] / mu0_s
        privacy += mu1_s * np.sum(sp.special.rel_entr(pi1_s, pi0_s))

    return 1 / privacy if not np.isclose(privacy, 0.) else np.infty


def full_information_privacy_lb(P0: np.ndarray,
                                P1: np.ndarray,
                                solver=cp.ECOS,
                                debug=False):
    """ Computes the policy that achieves the best level of privacy in the
    full information setting
    Parameters
    ----------
    P0, P1 : np.ndarray
        Numpy matrices containing the transition probabilities for model M0 and M1
        Each matrix should have dimensions |actions|x|states|x|states|
    solver : cvxpy.Solver, optional
        Solver used to solve the problem. Default solver is ECOS
    debug : bool, optional
        If true, prints the solver output. Default value is False
    Returns
    -------
    I_F : float
        Inverse of the privacy level
    xi : np.ndarray
        Stationary distribution over states and actions achieving the best
        level of privacy
    """

    # Check the matrices are ok
    P0, P1 = sanity_check_probabilities(P0, P1)
    na, ns = P0.shape[0], P0.shape[1]

    # Compute KL divergences
    I = compute_KL_divergence_models(P0, P1)

    best_res, best_xi = np.inf, None

    # Construct the problem to find minimum privacy
    xi = cp.Variable((ns, na), nonneg=True)

    objective = cp.Minimize(cp.sum(cp.multiply(xi, I)))

    # stationarity_constraint
    stationarity_constraint = 0
    for a in range(na):
        stationarity_constraint += xi[:, a].T @ (P1[a, :, :] - np.eye(ns))

    constraints = [stationarity_constraint == 0, cp.sum(xi) == 1]

    # Solve problem
    problem = cp.Problem(objective, constraints)
    try:
        result = problem.solve(verbose=debug, solver=solver)
    except Exception as err:
        raise Exception('Problem failed!')

    # Check if results are better than previous ones
    best_res, best_xi = result, xi.value

    # Make sure to normalize the results
    best_xi += eps
    best_xi /= np.sum(best_xi) if not np.isclose(np.sum(best_xi), 0) else 1.
    return best_res, best_xi


def full_information_privacy_utility(rho: float,
                                     lmbd: float,
                                     P0: np.ndarray,
                                     P1: np.ndarray,
                                     R0: np.ndarray,
                                     R1: np.ndarray,
                                     initial_points: int = 1,
                                     max_iterations: int = 30,
                                     solver=cp.ECOS,
                                     debug=False):
    """ Optimize the privacy-utility value function over the two policies
    in the full information setting
    Parameters
    ----------
    rho : float
        Weight given to policy pi_1 (1-rho for policy pi_0)
    lmbd : float
        Weight given to the privacy term
    P0, P1 : np.ndarray
        Numpy matrices containing the transition probabilities for model M0 and M1
        Each matrix should have dimensions |actions|x|states|x|states|
    R0, R1 : np.ndarray
        Numpy matrices containing the rewards for model M0 and M1
        Each matrix should have dimensions |states|x|actions|
    initial_points : int, optional
        Number of initial random points to use to solve the concave problem.
        Default value is 1.
    max_iterations : int, optional
        Maximum number of iterations. Should be larger than initial_points.
        Default value is 30.
    solver : cvxpy.Solver, optional
        Solver used to solve the problem. Default solver is ECOS
    debug : bool, optional
        If true, prints the solver output.
    Returns
    -------
    I_F : float
        Inverse of the privacy level
    xi1, xi0 : np.ndarray
        Stationary distributions over states and actions achieving the best
        level of utility-privacy
    """

    # Sanity checks
    P0, P1 = sanity_check_probabilities(P0, P1)
    R0, R1 = sanity_check_rewards(R0, R1)
    initial_points = int(initial_points) if initial_points >= 1 else 1
    max_iterations = initial_points if initial_points > max_iterations else int(
        max_iterations)

    if rho < 0 or rho > 1:
        raise ValueError('Rho should be in [0,1]')

    if lmbd < 0:
        raise ValueError('Lambda should be non-negative')

    na = P0.shape[0]
    ns = P1.shape[1]

    # Compute KL divergences
    I = compute_KL_divergence_models(P0, P1)

    best_res, best_xi1, best_xi0 = np.inf, None, None

    # Loop through initial points and return best result
    i = 0
    n = 0
    while i == 0 or (i < initial_points and n < max_iterations):
        n += 1
        # Construct the problem to find minimum privacy
        gamma = cp.Variable(1)
        xi1 = cp.Variable((ns, na), nonneg=True)
        xi0 = cp.Variable((ns, na), nonneg=True)

        kl_div_statinary_dis = 0
        for s in range(ns):
            kl_div_statinary_dis += cp.kl_div(
                cp.sum(xi1[s, :]), eps + cp.sum(xi0[s, :])) + cp.sum(
                    xi1[s, :]) - cp.sum(xi0[s, :]) - eps
        objective = gamma - lmbd * kl_div_statinary_dis

        # Stationarity constraint
        stationarity_constraint0 = 0
        stationarity_constraint1 = 0
        for a in range(na):
            stationarity_constraint0 += xi0[:, a].T @ (
                P0[a, :, :] - np.eye(ns))
            stationarity_constraint1 += xi1[:, a].T @ (
                P1[a, :, :] - np.eye(ns))

        # Privacy/utility constraint
        privacy_utility_constraint = 0
        for s in range(ns):
            for a in range(na):
                privacy_utility_constraint += xi1[s, a] * (
                    -rho * R1[s, a] + lmbd * I[s, a])
                privacy_utility_constraint += lmbd * (cp.kl_div(
                    xi1[s, a], eps + xi0[s][a]) + xi1[s, a] - xi0[s][a] - eps)
                privacy_utility_constraint += -(1 - rho) * xi0[s, a] * R0[s, a]

        constraints = [cp.sum(xi1) == 1, cp.sum(xi0) == 1]
        constraints += [
            stationarity_constraint0 == 0, stationarity_constraint1 == 0
        ]
        constraints += [privacy_utility_constraint <= gamma]

        # Solve problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not dccp.is_dccp(problem):
            raise Exception('Problem is not Concave with convex constraints!')
        try:
            result = problem.solve(
                method='dccp', ccp_times=1, verbose=debug, solver=solver)
        except Exception as err:
            continue

        # Check if results are better than previous ones
        if result[0] is not None:
            i += 1
            if result[0] < best_res:
                best_res, best_xi1, best_xi0 = result[0], xi1.value, xi0.value

    # Make sure to normalize the results
    best_xi0 += eps
    best_xi1 += eps
    best_xi0 /= np.sum(best_xi0) if not np.isclose(np.sum(best_xi0), 0) else 1.
    best_xi1 /= np.sum(best_xi1) if not np.isclose(np.sum(best_xi1), 0) else 1.
    return best_res, best_xi1, best_xi0
