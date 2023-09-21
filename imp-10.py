

def fast_remove_redundant_facets(lb, ub, S, c, opt_percentage=100):
    

    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception(
            "The number of reactions must be equal to the number of given flux bounds."
        )

    # Declare the tolerance that Gurobi works properly (we found it experimentally)
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
        # To avoid printing the output of the optimize() function of Gurobi, we need to set an environment like this
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model(env=env) as model:
                # Create variables
                x = model.addMVar(
                    shape=n,
                    vtype=GRB.CONTINUOUS,
                    name="x",
                    lb=lb,
                    ub=ub,
                )

                # Make sparse Aeq
                Aeq_sparse = sp.csr_matrix(S)

                # Make A sparse
                A_sparse = sp.csr_matrix(np.array(-c))
                b_sparse = np.array(val)

                # Set the b and beq vectors as numpy vectors
                b = np.array(b)
                beq = np.array(beq)

                # Add constraints
                model.addMConstr(Aeq_sparse, x, "=", beq, name="c")
                model.addMConstr(A_sparse, x, "<", [val], name="d")

                # Initialize the LP model
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
                    indices = indices_iter
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
                        objective_function_max = [-x for x in objective_function]
                        # Update the model with the new objective function
                        model_iter.setMObjective(None, objective_function_max, 0.0, None, None, x, GRB.MINIMIZE)
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
                            # Update the model with relaxed inequality
                            model_iter.setMObjective(None, objective_function_max, 0.0, None, None, x, GRB.MINIMIZE)
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                # Get the max objective value with relaxed inequality
                                max_objective2 = -model_iter.getObjective().getValue()
                                if np.abs(max_objective2 - max_objective) > redundant_facet_tol:
                                    redundant_facet_right = False
                                else:
                                    removed += 1
                                    facet_right_removed[0, i] = True

                        # Update the model for the minimum objective
                        model_iter.setMObjective(None, objective_function, 0.0, None, None, x, GRB.MINIMIZE)
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
                            # Update the model with relaxed inequality
                            model_iter.setMObjective(None, objective_function, 0.0, None, None, x, GRB.MINIMIZE)
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                # Get the min objective value with relaxed inequality
                                min_objective2 = model_iter.getObjective().getValue()
                                if np.abs(min_objective2 - min_objective) > redundant_facet_tol:
                                    redundant_facet_left = False
                                else:
                                    removed += 1
                                    facet_left_removed[0, i] = True

                        if (not redundant_facet_left) or (not redundant_facet_right):
                            width = abs(max_objective - min_objective)

                            # Check whether the offset in this dimension is small (and set an equality)
                            if width < redundant_facet_tol:
                                offset += 1
                                Aeq_res = np.vstack((Aeq_res, A[i, :]))
                                beq_res = np.append(beq_res, min(max_objective, min_objective))
                                # Remove the bounds on this dimension
                                ub[i] = sys.float_info.max
                                lb[i] = -sys.float_info.max
                            else:
                                # Store this dimension
                                indices_iter.append(i)

                                if not redundant_facet_left:
                                    # Not a redundant inequality
                                    A_res = np.append(A_res, np.array([A[n + i, :]]), axis=0)
                                    b_res.append(b[n + i])
                                else:
                                    lb[i] = -sys.float_info.max

                                if not redundant_facet_right:
                                    # Not a redundant inequality
                                    A_res = np.append(A_res, np.array([A[i, :]]), axis=0)
                                    b_res.append(b[i])
                                else:
                                    ub[i] = sys.float_info.max
                        else:
                            # Remove the bounds on this dimension
                            ub[i] = sys.float_info.max
                            lb[i] = -sys.float_info.max

                    b_res = np.asarray(b_res)
                    A_res = np.asarray(A_res, dtype="float")
                    A_res = np.ascontiguousarray(A_res, dtype="float")

                return A_res, b_res, Aeq_res, beq_res

    # Print error messages
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Gurobi solver failed.")
