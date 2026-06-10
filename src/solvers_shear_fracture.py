import numpy as np
from src.precomputed_integrals import load_rice_simons_integrals
from src.collocation_utils import (compute_power_law_interpolation_coefficients, build_integral_operator_matrix,
                                   build_profile_integration_matrix)

def solve_shear_prescribed_loading(params_list, array_x, array_x_mid):
    """
    Solve the semi-infinite shear fracture benchmark with prescribed
    uniform shear loading over a finite region.

    Parameters
    ----------
    params_list : list or tuple
        [beta, A], where beta is the normalized undrained-drained Poisson's
        ratio contrast and A is the dimensionless loading length.
    array_x : ndarray
        Grid points in the dimensionless coordinate defining the discretization
        intervals. The derivative of the slip profile is represented at these
        points.
    array_x_mid : ndarray
        Midpoints of the intervals between consecutive grid points in array_x;
        used as collocation points for enforcing the boundary integral equations.

    Returns
    -------
    profiles : ndarray
        Columns: x, slip.
    matrices : list of ndarray
        Discrete integral-operator matrices [S_e].
    dd_dx : ndarray
        Nodal derivative of the dimensionless slip profile.
    K_d : float
        Near-tip slip coefficient, corresponding to the dimensionless Mode-II
        stress intensity factor.
    """

    beta, A = params_list

    a_coef_left_node, a_coef_right_node, b_coef_left_node, b_coef_right_node = (
        compute_power_law_interpolation_coefficients(array_x, -0.5, -1.5))

    # Load precomputed integral matrices used by the collocation method.
    integrals = load_rice_simons_integrals()

    S_e_matrix_0 = integrals["s_e_matrix_0"]
    S_e_matrix_inf = integrals["s_e_matrix_inf"]
    S_e_near_field_integral = integrals["s_e_near_field_integral"]

    cauchy_matrix_0 = integrals["cauchy_matrix_0"]
    cauchy_matrix_inf = integrals["cauchy_matrix_inf"]
    cauchy_near_field_integral = integrals["cauchy_near_field_integral"]

    # The S_e kernels were precomputed for beta = 0.25. Since the beta-dependent
    # part enters linearly after separating the Cauchy contribution, the matrices
    # are reconstructed here for the requested beta value.
    beta_default = 0.25
    S_e_matrix_0_r = (S_e_matrix_0 - cauchy_matrix_0) / beta_default
    S_e_matrix_inf_r = (S_e_matrix_inf - cauchy_matrix_inf) / beta_default
    S_e_near_field_r = (S_e_near_field_integral - cauchy_near_field_integral) / beta_default

    S_e_matrix_0 = cauchy_matrix_0 + beta * S_e_matrix_0_r
    S_e_matrix_inf = cauchy_matrix_inf + beta * S_e_matrix_inf_r
    S_e_near_field_integral = cauchy_near_field_integral + beta * S_e_near_field_r

    S_e = build_integral_operator_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [S_e_matrix_0, S_e_matrix_inf]
    )

    S_e[:, 0] += S_e_near_field_integral * array_x[0] ** 0.5

    A_matrix = 1 / (4 * np.pi * (1 - beta)) * S_e[:, :-1]

    b_vector = np.zeros(array_x_mid.shape[0])
    b_vector[array_x_mid < A] = 1

    dd_dx = np.linalg.solve(A_matrix, b_vector)
    dd_dx = np.concatenate((dd_dx, np.array([0.0])))

    K_d = dd_dx[0] * 2 * array_x[0] ** 0.5

    # Interval integrals of the two power-law basis functions used to
    # reconstruct the slip profile from its derivative.
    basis_integral_0 = 2 * (array_x[1:] ** 0.5 - array_x[:-1] ** 0.5)
    basis_integral_inf = -2 * (array_x[1:] ** -0.5 - array_x[:-1] ** -0.5)

    int_matrix_d = build_profile_integration_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [basis_integral_0, basis_integral_inf]
    )

    int_near_field = K_d * array_x[0] ** 0.5

    slip = np.concatenate((np.array([int_near_field]), int_near_field + int_matrix_d @ dd_dx))

    return np.concatenate((array_x[:, None], slip[:, None]), axis=1), [S_e], dd_dx, K_d