def fast_remove_redundant_facets(lb, ub, S, c, opt_percentage=100):
    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )

    # Constants
    redundant_facet_tol = 1e-07
    tol = 1e-06
    m = S.shape[0]
    n = S.shape[1]
    beq = np.zeros(m)
    Aeq_res = S

    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")

    b = np.concatenate((ub, -lb), axis=0).astype(float)
    
    # Calculate val without unnecessary conversion
    max_biomass_flux_vector, max_biomass_objective = fast_fba(lb, ub, S, c)
    val = -np.floor(max_biomass_objective / tol) * tol * opt_percentage / 100

    b_res = []
    A_res = np.empty((0, n), dtype=float)
    beq_res = np.array(beq, dtype=float)

    try:
        model = Model()

        # Create variables
        x = model.addVars(n, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")

        # Make sparse Aeq
        Aeq_sparse = sp.csr_matrix(S)

        # Make A sparse
        A_sparse = sp.csr_matrix(np.array(-c))
        b_sparse = np.array(val)

        # Set the b and beq vectors as numpy vectors
        b = np.array(b, dtype=float)
        beq = np.array(beq, dtype=float)

        # Add constraints
        model.addMConstr(Aeq_sparse, x.values(), "=", beq, name="c")

        # Add constraints for the inequalities of A
        model.addMConstr(A_sparse, x.values(), "<", [val], name="d")

        model.update()

        # Initialize variables
        indices_iter = range(n)
        removed = 1
        offset = 1
        facet_left_removed = np.zeros((1, n), dtype=bool)
        facet_right_removed = np.zeros((1, n), dtype=bool)

        # Loop until no redundant facets are found
        while removed > 0 or offset > 0:
            removed = 0
            offset = 0
            indices = indices_iter
            indices_iter = []

            Aeq_sparse = sp.csr_matrix(Aeq_res)
            beq = np.array(beq_res)

            b_res = []
            A_res = np.empty((0, n), dtype=float)
            for i in indices:
                objective_function = A[i, :]

                redundant_facet_right = True
                redundant_facet_left = True

                # For the maximum
                objective_function_max = np.asarray([-x[i] for i in x])
                x_new = model.copy()
                x_new.update()

                x_new.addMConstr(Aeq_sparse, x.values(), "=", beq, name="c")
                x_new.setObjective(objective_function_max @ x.values(), GRB.MAXIMIZE)
                x_new.update()
                x_new.optimize()

                status = x_new.status
                if status == GRB.OPTIMAL:
                    max_objective = -x_new.objval
                else:
                    max_objective = ub[i]

                if not facet_right_removed[0, i]:
                    ub_iter = ub.copy()
                    ub_iter[i] = ub_iter[i] + 1

                    x_new = model.copy()
                    x_new.update()
                    x_new.addMConstr(Aeq_sparse, x.values(), "=", beq, name="c")
                    x_new.setObjective(objective_function_max @ x.values(), GRB.MAXIMIZE)
                    x_new.update()
                    x_new.optimize()

                    status = x_new.status
                    if status == GRB.OPTIMAL:
                        max_objective2 = -x_new.objval
                        if np.abs(max_objective2 - max_objective) > redundant_facet_tol:
                            redundant_facet_right = False
                        else:
                            removed += 1
                            facet_right_removed[0, i] = True

                x_new = model.copy()
                x_new.update()

                x_new.addMConstr(Aeq_sparse, x.values(), "=", beq, name="c")
                x_new.setObjective(objective_function @ x.values(), GRB.MINIMIZE)
                x_new.update()
                x_new.optimize()

                status = x_new.status
                if status == GRB.OPTIMAL:
                    min_objective = x_new.objval
                else:
                    min_objective = lb[i]

                if not facet_left_removed[0, i]:
                    lb_iter = lb.copy()
                    lb_iter[i] = lb_iter[i] - 1

                    x_new = model.copy()
                    x_new.update()
                    x_new.addMConstr(Aeq_sparse, x.values(), "=", beq, name="c")
                    x_new.setObjective(objective_function @ x.values(), GRB.MINIMIZE)
                    x_new.update()
                    x_new.optimize()

                    status = x_new.status
                    if status == GRB.OPTIMAL:
                        min_objective2 = x_new.objval
                        if np.abs(min_objective2 - min_objective) > redundant_facet_tol:
                            redundant_facet_left = False
                        else:
                            removed += 1
                            facet_left_removed[0, i] = True

                if (not redundant_facet_left) or (not redundant_facet_right):
                    width = abs(max_objective - min_objective)

                    if width < redundant_facet_tol:
                        offset += 1
                        Aeq_res = np.vstack((Aeq_res, A[i, :]))
                        beq_res = np.append(beq_res, min(max_objective, min_objective))
                        ub[i] = np.inf
                        lb[i] = -np.inf
                    else:
                        indices_iter.append(i)

                        if not redundant_facet_left:
                            A_res = np.append(A_res, np.array([A[n + i, :]]), axis=0)
                            b_res.append(b[n + i])
                        else:
                            lb[i] = -np.inf

                        if not redundant_facet_right:
                            A_res = np.append(A_res, np.array([A[i, :]]), axis=0)
                            b_res.append(b[i])
                        else:
                            ub[i] = np.inf
                else:
                    ub[i] = np.inf
                    lb[i] = -np.inf

            b_res = np.asarray(b_res, dtype=float)
            A_res = np.asarray(A_res, dtype=float)

        return A_res, b_res, Aeq_res, beq_res

    except Exception as e:
        print("Error:", e)

# Sample usage:
# lb, ub, S, c = ...  # Define your input data
# A_res, b_res, Aeq_res, beq_res = fast_remov
