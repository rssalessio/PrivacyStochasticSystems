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


def limited_information_privacy(P0: np.ndarray, P1: np.ndarray,
                                xi0: np.ndarray, xi1: np.ndarray) -> float:
    """ Computes 1/I_L(pi_0, pi_1) given xi_0 and xi_1
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
    1/I_L : float
        Privacy level
    """

    P0, P1 = sanity_check_probabilities(P0, P1)
    xi0, xi1 = np.array(xi0), np.array(xi1)
    na, ns = P0.shape[0], P0.shape[1]

    privacy = 0
    for s in range(ns):
        z = sp.special.rel_entr(np.sum(xi1[s, :]), np.sum(xi0[s, :]))
        if z == np.infty:
            print(
                'An infinity was computed in a KL-Divergence. Check the first term: {}'
                .format(np.sum(mu1[s, :])))
            z = 0
        privacy -= z

        for y in range(ns):
            z = sp.special.rel_entr(xi1[s, :] @ P1[:, s, y],
                                    xi0[s, :] @ P0[:, s, y])
            if z == np.infty:
                print(
                    'An infinity was computed in a KL-Divergence. Check the first term: {}'
                    .format(xi1[s, :] @ P1[:, s, y]))
                z = 0
            privacy += z

    return 1 / privacy if not np.isclose(privacy, 0.) else np.infty


def limited_information_privacy_lb(P0: np.ndarray,
                                   P1: np.ndarray,
                                   initial_points: int = 1,
                                   max_iterations: int = 30,
                                   solver=cp.ECOS,
                                   debug=False):
    """ Computes the policy that achieves the best level of privacy in the
    limited information setting
    Parameters
    ----------
    P0, P1 : np.ndarray
        Numpy matrices containing the transition probabilities for models M0 and M1
        Each matrix should have dimensions |actions|x|states|x|states|
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
    I_L : float
        Inverse of the privacy level
    xi1, xi0 : np.ndarray
        Stationary distributions over states and actions achieving the best
        level of privacy
    """
    P0, P1 = sanity_check_probabilities(P0, P1)
    initial_points = int(initial_points) if initial_points >= 1 else 1
    max_iterations = initial_points if initial_points > max_iterations else int(
        max_iterations)

    na, ns = P0.shape[0], P0.shape[1]

    best_res, best_xi1, best_xi0 = np.inf, None, None
    # Compute KL divergences
    I = compute_KL_divergence_models(P0, P1)

    # Loop through initial points and return best result
    i = 0
    n = 0
    while i == 0 or (i < initial_points and n < max_iterations):
        n += 1
        gamma = cp.Variable(1)
        xi0 = cp.Variable((ns, na), nonneg=True)
        xi1 = cp.Variable((ns, na), nonneg=True)

        kl_div_statinary_dis = 0
        for s in range(ns):
            kl_div_statinary_dis += cp.entr(cp.sum(xi1[s, :]))

        # stationarity constraints
        stationarity_constraint = 0
        for a in range(na):
            stationarity_constraint += xi1[:, a].T @ (P1[a, :, :] - np.eye(ns))

        constraints = [stationarity_constraint == 0, cp.sum(xi1) == 1]

        # Privacy constraints
        privacy_constraint = 0
        for s in range(ns):
            constraints += [cp.sum(xi0[s, :]) == 1]
            for y in range(ns):
                privacy_constraint += cp.kl_div(
                    xi1[s, :] @ P1[:, s, y], xi0[s, :] @ P0[:, s, y]) + (
                        xi1[s, :] @ P1[:, s, y]) - (xi0[s, :] @ P0[:, s, y])

        constraints += [privacy_constraint <= gamma]
        objective = gamma + kl_div_statinary_dis

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


def limited_information_privacy_utility(rho: float,
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
    in the limited information setting
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
    I_L : float
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

    best_res, best_xi1, best_xi0 = np.inf, None, None

    # Loop through initial points and return best result
    i = 0
    n = 0
    while i == 0 or (i < initial_points and n < max_iterations):
        n += 1

        # Construct the problem to find minimum privacy
        gamma = cp.Variable(1, nonneg=True)
        xi0 = cp.Variable((ns, na), nonneg=True)
        xi1 = cp.Variable((ns, na), nonneg=True)

        kl_div_statinary_dis = 0
        for s in range(ns):
            kl_div_statinary_dis += cp.kl_div(
                cp.sum(xi1[s, :]), cp.sum(xi0[s, :])) + cp.sum(
                    xi1[s, :]) - cp.sum(xi0[s, :])
        objective = gamma - lmbd * kl_div_statinary_dis

        # stationarity constraints
        stationarity_constraint0 = 0
        stationarity_constraint1 = 0
        for a in range(na):
            stationarity_constraint0 += xi0[:, a].T @ (
                P0[a, :, :] - np.eye(ns))
            stationarity_constraint1 += xi1[:, a].T @ (
                P1[a, :, :] - np.eye(ns))

        constraints = [
            stationarity_constraint0 == 0, stationarity_constraint1 == 0,
            cp.sum(xi0) == 1,
            cp.sum(xi1) == 1
        ]

        # Privacy-utility constraints
        privacy_utility_constraint = 0
        for s in range(ns):
            for y in range(ns):
                privacy_utility_constraint += lmbd * (
                    cp.kl_div(xi1[s, :] @ P1[:, s, y], xi0[s, :] @ P0[:, s, y])
                    + (xi1[s, :] @ P1[:, s, y]) - (xi0[s, :] @ P0[:, s, y]))
            for a in range(na):
                privacy_utility_constraint -= (
                    rho * xi1[s, a] * R1[s, a] +
                    (1 - rho) * xi0[s, a] * R0[s, a])

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
