


def fast_remove_redundant_facets(lb, ub, S, c, opt_percentage=100):
    ""Find and remove the redundant facets and identify facets with small offsets.

   
    S -- the mxn stoichiometric matrix, s.t. Sv = 0
    c -- the objective function to maximize
    opt_percentage -- consider solutions that give you at least a certain
                      percentage of the optimal solution (default is to consider
                      optimal solutions only)
    ""

    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )

    
    redundant_facet_tol = 1e-07
    tol = 1e-06

    m = S.shape[0]
    n = S.shape[1]
    beq = np.zeros(m)
    Aeq_res = S

    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")

    b = np.concatenate((ub, -lb), axis=0)
    b = np.asarray(b, dtype="float")
    b = np.ascontiguousarray(b, dtype="float")

    # call fba to obtain an optimal solution
    max_biomass_flux_vector, max_biomass_objective = fast_fba(lb, ub, S, c)
    val = -np.floor(max_biomass_objective / tol) * tol * opt_percentage / 100

    b_res = []
    A_res = np.empty((0, n), float)
    beq_res = np.array(beq)

    try:

       
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            d = A.shape[1]

            with gp.Model(env=env) as model:

                # Create variables
                x = model.addMVar(
                    shape=d,
                    vtype=GRB.CONTINUOUS,
                    name="x",
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                )
                model.update()

                # Make A_full_dim sparse
                A_expand = np.c_[A, np.linalg.norm(A, axis=1)]
                A_expand_sparse = sp.csr_matrix(A_expand.astype("float"))

                # Add constraints
                model.addMConstr(A_expand_sparse, x, "<", b, name="c")
                model.update()

                model_iter = model.copy()

                # Initialize
                indices_iter = range(n)
                removed = 1
                offset = 1
                facet_left_removed = np.zeros((1, n), dtype=bool)
                facet_right_removed = np.zeros((1, n), dtype=bool)

                # Loop until no redundant facets are found
                while removed > 0 or offset > 0:

                    removed = 0
                    offset = 0
                    indices = indices_iter.copy()
                    indices_iter = []

                    Aeq_sparse = sp.csr_matrix(Aeq_res)
                    beq = np.array(beq_res)

                    b_res = []
                    A_res = np.empty((0, n), float)
                    for i in indices:

                        # Set the ith row of the A matrix as the objective function
                        objective_function = A[i, :]

                        redundant_facet_right = True
                        redundant_facet_left = True

                        # For the maximum
                        objective_function_max = np.asarray(
                            [-x for x in objective_function]
                        )
                        model_iter.setMObjective(
                            None, objective_function_max, 0.0, None, None, x, GRB.MINIMIZE
                        )
                        model_iter.update()
                        model_iter.optimize()

                        # Again if optimized
                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            # Get the max objective value
                            max_objective = -model_iter.getObjective().getValue()
                        else:
                            max_objective = ub[i]

                        # If this facet was not removed in a previous iteration
                        if not facet_right_removed[0, i]:
                            ub_iter = ub.copy()
                            ub_iter[i] = ub_iter[i] + 1
                            model_iter.setMObjective(
                                None, objective_function_max, 0.0, None, None, x, GRB.MINIMIZE
                            )
                            model_iter.update()
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                # Get the max objective value with relaxed inequality
                                max_objective2 = -model_iter.getObjective().getValue()
                                if (
                                    np.abs(max_objective2 - max_objective)
                                    > redundant_facet_tol
                                ):
                                    redundant_facet_right = False
                                else:
                                    removed += 1
                                    facet_right_removed[0, i] = True

                        model_iter.setMObjective(
                            None, objective_function, 0.0, None, None, x, GRB.MINIMIZE
                        )
                        model_iter.update()
                        model_iter.optimize()

                        # If optimized
                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            # Get the min objective value
                            min_objective = model_iter.getObjective().getValue()
                        else:
                            min_objective = lb[i]

                        # If this facet was not removed in a previous iteration
                        if not facet_left_removed[0, i]:
                            lb_iter = lb.copy()
                            lb_iter[i] = lb_iter[i] - 1
                            model_iter.setMObjective(
                                None, objective_function, 0.0, None, None, x, GRB.MINIMIZE
                            )
                            model_iter.update()
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                # Get the min objective value with relaxed inequality
                                min_objective2 = model_iter.getObjective().getValue()
                                if (
                                    np.abs(min_objective2 - min_objective)
                                    > redundant_facet_tol
                                ):
                                    redundant_facet_left = False
                                else:
                                    removed += 1
                                    facet_left_removed[0, i] = True

                        if (not redundant_facet_left) or (not redundant_facet_right):
                            width = abs(max_objective - min_objective)

                            if width > redundant_facet_tol:
                                # offset small facet
                                if width < tol:
                                    offset += 1
                                    indices_iter.append(i)
                                    Aeq_sparse = sp.vstack(
                                        [
                                            Aeq_sparse,
                                            np.expand_dims(objective_function, 0),
                                        ],
                                        format="csr",
                                    )
                                    beq = np.append(
                                        beq, [np.dot(objective_function, lb)]
                                    )

                                else:
                                    Aeq_sparse = sp.vstack(
                                        [Aeq_sparse, np.expand_dims(objective_function, 0)],
                                        format="csr",
                                    )
                                    beq = np.append(
                                        beq, [np.dot(objective_function, lb)]
                                    )

                        # Update the index sets to optimize
                        if (not redundant_facet_left) and (
                            not facet_left_removed[0, i]
                        ):
                            lb[i] = lb[i] - 1

                        if (not redundant_facet_right) and (
                            not facet_right_removed[0, i]
                        ):
                            ub[i] = ub[i] + 1

                    if removed + offset > 0:
                        indices_iter = list(set(indices_iter))
                        for i in indices_iter:
                            indices.remove(i)

                        indices_iter = list(set(indices_iter))
                        beq_res = beq
                        Aeq_res = sp.vstack(
                            [
                                Aeq_sparse[i, :].toarray()
                                for i in range(Aeq_sparse.shape[0])
                            ],
                            format="csr",
                        )
                        if Aeq_res.shape[0] > n + 1:
                            b_res = np.append(
                                beq_res,
                                np.dot(
                                    Aeq_res[n + 1 :, :],
                                    np.linalg.inv(Aeq_res[0 : n + 1, :]),
                                ),
                            )

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        sys.exit(1)

    return lb, ub, Aeq_res, beq_res

# Usage example:
# lb, ub, S, c = load_your_data_here()  # Replace with your data loading code
# lb, ub, S, c = preprocess_data(lb, ub, S, c)  # Optional data preprocessing
# lb, ub, Aeq_res, beq_res = fast_remove_redundant_facets(lb, ub, S, c)
