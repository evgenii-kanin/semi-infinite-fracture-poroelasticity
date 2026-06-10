import numpy as np
from src.precomputed_integrals import load_atkinson_craster_integrals
from src.collocation_utils import (compute_power_law_interpolation_coefficients, build_integral_operator_matrix,
                                   build_profile_integration_matrix)

def solve_tensile_prescribed_loading_impermeable_surfaces(params_list, array_x, array_x_mid):
    """
    Solve the semi-infinite tensile fracture benchmark with prescribed
    exponential normal loading and impermeable fracture surfaces.

    Parameters
    ----------
    params_list : list or tuple
        [beta, A], where beta is the normalized undrained-drained Poisson's
        ratio contrast and A is the dimensionless loading length.
    array_x : ndarray
        Grid points in the dimensionless coordinate defining the discretization
        intervals. The derivative of the opening profile is represented at these
        points.
    array_x_mid : ndarray
        Midpoints of the intervals between consecutive grid points in array_x;
        used as collocation points for enforcing the boundary integral equations.

    Returns
    -------
    profiles : ndarray
        Columns: x, opening.
    matrices : list of ndarray
        Discrete integral-operator matrices [S_e, P_e]. The matrix P_e is
        returned for post-processing the induced pore pressure field.
    dw_dx : ndarray
        Nodal derivative of the dimensionless opening profile.
    K_w : float
        Near-tip opening coefficient, corresponding to the dimensionless
        Mode-I stress intensity factor.
    """

    beta, A = params_list

    a_coef_left_node, a_coef_right_node, b_coef_left_node, b_coef_right_node = (
        compute_power_law_interpolation_coefficients(array_x, -0.5, -1.5))

    # Load precomputed integral matrices used by the collocation method.
    integrals = load_atkinson_craster_integrals()

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

    P_e_matrix_0 = integrals["p_e_matrix_0"]
    P_e_matrix_inf = integrals["p_e_matrix_inf"]
    P_e_near_field_integral = integrals["p_e_near_field_integral"]

    P_e = build_integral_operator_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [P_e_matrix_0, P_e_matrix_inf]
    )

    P_e[:, 0] += P_e_near_field_integral * array_x[0] ** 0.5

    A_matrix = 1 / (4 * np.pi * (1 - beta)) * S_e[:, :-1]

    b_vector = np.exp(-array_x_mid / A)

    dw_dx = np.linalg.solve(A_matrix, b_vector)
    dw_dx = np.concatenate((dw_dx, np.array([0.0])))

    K_w = dw_dx[0] * 2 * array_x[0] ** 0.5

    # Interval integrals of the two power-law basis functions used to
    # reconstruct the opening profile from its derivative.
    basis_integral_0 = 2 * (array_x[1:] ** 0.5 - array_x[:-1] ** 0.5)
    basis_integral_inf = -2 * (array_x[1:] ** -0.5 - array_x[:-1] ** -0.5)

    int_matrix_w = build_profile_integration_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [basis_integral_0, basis_integral_inf]
    )

    int_near_field = K_w * array_x[0] ** 0.5

    opening = np.concatenate((np.array([int_near_field]), int_near_field + int_matrix_w @ dw_dx))

    return np.concatenate((array_x[:, None], opening[:, None]), axis=1), [S_e, P_e], dw_dx, K_w

def solve_tensile_prescribed_loading_permeable_surfaces(params_list, array_x, array_x_mid):
    """
    Solve the semi-infinite tensile fracture benchmark with prescribed
    exponential normal loading and permeable fracture surfaces.

    Parameters
    ----------
    params_list : list or tuple
        [eta, beta, A], where eta is the poroelastic stress coefficient,
        beta is the normalized undrained-drained Poisson's ratio contrast,
        and A is the dimensionless loading length.
    array_x : ndarray
        Grid points in the dimensionless coordinate defining the discretization
        intervals. The derivatives of the opening profile and fluid displacement
        function are represented at these points.
    array_x_mid : ndarray
        Midpoints of the intervals between consecutive grid points in array_x;
        used as collocation points for enforcing the boundary integral equations.

    Returns
    -------
    profiles : ndarray
        Columns: x, opening, fluid_displacement.
    matrices : list of ndarray
        Discrete integral-operator matrices [S_e, S_s, P_e, P_s].
    sol : ndarray
        Concatenated vector of nodal derivatives of the dimensionless opening and
        fluid displacement function profiles [dw_dx, dv_dx].
    coefficients : list of float
        [K_w, K_v], where K_w is the near-tip opening coefficient,
        corresponding to the dimensionless Mode-I stress intensity factor,
        and K_v is the near-tip fluid displacement coefficient.
    """

    eta, beta, A = params_list

    a_coef_left_node, a_coef_right_node, b_coef_left_node, b_coef_right_node = (
        compute_power_law_interpolation_coefficients(array_x, -0.5, -1.5))

    # Load precomputed integral matrices used by the collocation method.
    integrals = load_atkinson_craster_integrals()

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

    P_e_matrix_0 = integrals["p_e_matrix_0"]
    P_e_matrix_inf = integrals["p_e_matrix_inf"]
    P_e_near_field_integral = integrals["p_e_near_field_integral"]

    P_e = build_integral_operator_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [P_e_matrix_0, P_e_matrix_inf]
    )

    P_e[:, 0] += P_e_near_field_integral * array_x[0] ** 0.5

    P_s_matrix_0 = integrals["p_s_matrix_0"]
    P_s_matrix_inf = integrals["p_s_matrix_inf"]
    P_s_near_field_integral = integrals["p_s_near_field_integral"]

    P_s = build_integral_operator_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [P_s_matrix_0, P_s_matrix_inf]
    )

    P_s[:, 0] += P_s_near_field_integral * array_x[0] ** 0.5

    # The same discrete coupling matrix is used for S_s and P_e;
    # separate names are kept to mirror the block-operator notation.
    S_s = np.copy(P_e)

    if eta == 0 or beta == 0:
        eta = beta = 0

    storage_coefficient = 2 * (1 - beta) * eta ** 2 / beta if eta > 0 else 1

    A_matrix = np.vstack((
        np.hstack((1 / (4 * np.pi * (1 - beta)) * S_e[:, :-1], -eta / (2 * np.pi * storage_coefficient) * S_s[:, :-1])),
        np.hstack((-eta * P_e[:, :-1], P_s[:, :-1]))
    ))

    b_vector = np.hstack((
        np.exp(-array_x_mid / A),
        np.zeros(array_x_mid.shape[0])
    ))

    sol = np.linalg.solve(A_matrix, b_vector)

    dw_dx, dv_dx = sol[:array_x.shape[0] - 1], sol[array_x.shape[0] - 1:]

    dw_dx = np.concatenate((dw_dx, np.array([0.0])))
    dv_dx = np.concatenate((dv_dx, np.array([0.0])))

    K_w, K_v = dw_dx[0] * 2 * array_x[0] ** 0.5, dv_dx[0] * 2 * array_x[0] ** 0.5

    # Interval integrals of the two power-law basis functions used to
    # reconstruct the opening and fluid displacement function profiles from their derivatives.
    basis_integral_0 = 2 * (array_x[1:] ** 0.5 - array_x[:-1] ** 0.5)
    basis_integral_inf = -2 * (array_x[1:] ** -0.5 - array_x[:-1] ** -0.5)

    int_matrix_w_v = build_profile_integration_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [basis_integral_0, basis_integral_inf]
    )

    int_opening_near_field = K_w * array_x[0] ** 0.5
    opening = np.concatenate((np.array([int_opening_near_field]), int_opening_near_field + int_matrix_w_v @ dw_dx))

    int_fluid_near_field = K_v * array_x[0] ** 0.5
    fluid_displacement = np.concatenate((np.array([int_fluid_near_field]), int_fluid_near_field + int_matrix_w_v @ dv_dx))

    return np.concatenate((array_x[:, None], opening[:, None], fluid_displacement[:, None]), axis=1), \
        [S_e, S_s, P_e, P_s], sol, [K_w, K_v]

def solve_tensile_prescribed_pore_pressure(params_list, array_x, array_x_mid):
    """
    Solve the semi-infinite tensile fracture benchmark with prescribed
    exponential pore fluid pressure on the fracture surfaces.

    Parameters
    ----------
    params_list : list or tuple
        [eta, beta, A], where eta is the poroelastic stress coefficient,
        beta is the normalized undrained-drained Poisson's ratio contrast,
        and A is the dimensionless pressure-decay length.
    array_x : ndarray
        Grid points in the dimensionless coordinate defining the discretization
        intervals. The derivatives of the opening profile and fluid displacement
        function are represented at these points.
    array_x_mid : ndarray
        Midpoints of the intervals between consecutive grid points in array_x;
        used as collocation points for enforcing the boundary integral equations.

    Returns
    -------
    profiles : ndarray
        Columns: x, opening, fluid_displacement.
    matrices : list of ndarray
        Discrete integral-operator matrices [S_e, S_s, P_e, P_s].
    sol : ndarray
        Concatenated vector of nodal derivatives of the dimensionless opening and
        fluid displacement function profiles [dw_dx, dv_dx].
    coefficients : list of float
        [K_w, K_v], where K_w is the near-tip opening coefficient,
        corresponding to the dimensionless Mode-I stress intensity factor,
        and K_v is the near-tip fluid displacement coefficient.
    """

    eta, beta, A = params_list

    a_coef_left_node, a_coef_right_node, b_coef_left_node, b_coef_right_node = (
        compute_power_law_interpolation_coefficients(array_x, -0.5, -1.5))

    # Load precomputed integral matrices used by the collocation method.
    integrals = load_atkinson_craster_integrals()

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

    P_e_matrix_0 = integrals["p_e_matrix_0"]
    P_e_matrix_inf = integrals["p_e_matrix_inf"]
    P_e_near_field_integral = integrals["p_e_near_field_integral"]

    P_e = build_integral_operator_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [P_e_matrix_0, P_e_matrix_inf]
    )

    P_e[:, 0] += P_e_near_field_integral * array_x[0] ** 0.5

    P_s_matrix_0 = integrals["p_s_matrix_0"]
    P_s_matrix_inf = integrals["p_s_matrix_inf"]
    P_s_near_field_integral = integrals["p_s_near_field_integral"]

    P_s = build_integral_operator_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [P_s_matrix_0, P_s_matrix_inf]
    )

    P_s[:, 0] += P_s_near_field_integral * array_x[0] ** 0.5

    # The same discrete coupling matrix is used for S_s and P_e;
    # separate names are kept to mirror the block-operator notation.
    S_s = np.copy(P_e)

    if eta == 0 or beta == 0:
        eta = beta = 0

    storage_coefficient = 2 * (1 - beta) * eta ** 2 / beta if eta > 0 else 1

    A_matrix = np.vstack((
        np.hstack((1 / (4 * np.pi * (1 - beta)) * S_e[:, :-1], -eta / (2 * np.pi * storage_coefficient) * S_s[:, :-1])),
        np.hstack((-eta / (2 * np.pi * storage_coefficient) * P_e[:, :-1], 1 / (2 * np.pi * storage_coefficient) * P_s[:, :-1]))
    ))

    b_vector = np.hstack((
        np.zeros(array_x_mid.shape[0]),
        np.exp(-array_x_mid / A)
    ))

    sol = np.linalg.solve(A_matrix, b_vector)

    dw_dx, dv_dx = sol[:array_x.shape[0] - 1], sol[array_x.shape[0] - 1:]

    dw_dx = np.concatenate((dw_dx, np.array([0.0])))
    dv_dx = np.concatenate((dv_dx, np.array([0.0])))

    K_w, K_v = dw_dx[0] * 2 * array_x[0] ** 0.5, dv_dx[0] * 2 * array_x[0] ** 0.5

    # Interval integrals of the two power-law basis functions used to
    # reconstruct the opening profile and fluid displacement function
    # from their derivatives.
    basis_integral_0 = 2 * (array_x[1:] ** 0.5 - array_x[:-1] ** 0.5)
    basis_integral_inf = -2 * (array_x[1:] ** -0.5 - array_x[:-1] ** -0.5)

    int_matrix_w_v = build_profile_integration_matrix(
        [a_coef_left_node, a_coef_right_node],
        [b_coef_left_node, b_coef_right_node],
        [basis_integral_0, basis_integral_inf]
    )

    int_opening_near_field = K_w * array_x[0] ** 0.5
    opening = np.concatenate((np.array([int_opening_near_field]), int_opening_near_field + int_matrix_w_v @ dw_dx))

    int_fluid_near_field = K_v * array_x[0] ** 0.5
    fluid_displacement = np.concatenate((np.array([int_fluid_near_field]), int_fluid_near_field + int_matrix_w_v @ dv_dx))

    return np.concatenate((array_x[:, None], opening[:, None], fluid_displacement[:, None]), axis=1), \
        [S_e, S_s, P_e, P_s], sol, [K_w, K_v]