

def fast_remove_redundant_facets(lb, ub, S, c, opt_percentage=100):
    if lb.size != S.shape[1] or ub.size != S.shape[1]:
        raise Exception("The number of reactions must be equal to the number of given flux bounds.")

    redundant_facet_tol = 1e-07
    tol = 1e-06

    m, n = S.shape
    beq = np.zeros(m)

   
    Aeq_sparse = sp.csc_matrix(S)

    A = sp.lil_matrix((2 * n, n), dtype="float")
    A[:n, :] = sp.eye(n)
    A[n:, :] -= sp.eye(n)

    b_upper = sp.csc_matrix((ub, (range(n), [0] * n)), shape=(n, 1), dtype="float")
    b_lower = sp.csc_matrix((lb, (range(n), [0] * n)), shape=(n, 1), dtype="float")

    b = sp.vstack([b_upper, -b_lower], format="csr")

    # Call FBA to obtain an optimal solution
    max_biomass_flux_vector, max_biomass_objective = fast_fba(lb, ub, S, c)
    val = -np.floor(max_biomass_objective / tol) * tol * opt_percentage / 100

    b_res = []
    A_res = sp.lil_matrix((0, n), dtype="float")
    beq_res = np.array(beq)

    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            with gp.Model(env=env) as model:
                x = model.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="x", lb=lb, ub=ub)

                # Make sparse A_sparse
                A_sparse = sp.csr_matrix(np.array(-c))
                b_sparse = np.array(val)

                # Add constraints
                model.addMConstr(Aeq_sparse, x, "=", beq, name="c")

                # Update the model to include the constraints added
                model.update()

                # Add constraints for the inequalities of A
                model.addMConstr(A_sparse, x, "<", [val], name="d")

                # Update the model with the extra constraints and then print it
                model.update()

                model_iter = model.copy()

                indices_iter = range(n)
                removed = 1
                offset = 1
                facet_left_removed = np.zeros((1, n), dtype=bool)
                facet_right_removed = np.zeros((1, n), dtype=bool)

                while removed > 0 or offset > 0:
                    removed = 0
                    offset = 0
                    indices = indices_iter.copy()
                    indices_iter = []

                    Aeq_sparse = sp.csr_matrix(Aeq_res)
                    beq = np.array(beq_res)

                    b_res = []
                    A_res = sp.lil_matrix((0, n), dtype="float")
                    for i in indices:
                        objective_function = A[i, :]

                        redundant_facet_right = True
                        redundant_facet_left = True

                        objective_function_max = -objective_function
                        model_iter.setObjective(objective_function_max, GRB.MAXIMIZE)
                        model_iter.optimize()

                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            max_objective = -model_iter.getObjective().getValue()
                        else:
                            max_objective = ub[i]

                        if not facet_right_removed[0, i]:
                            ub_iter = ub.copy()
                            ub_iter[i] = ub_iter[i] + 1
                            model_iter.setObjective(objective_function_max, GRB.MAXIMIZE)
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                max_objective2 = -model_iter.getObjective().getValue()
                                if np.abs(max_objective2 - max_objective) > redundant_facet_tol:
                                    redundant_facet_right = False
                                else:
                                    removed += 1
                                    facet_right_removed[0, i] = True

                        model_iter.setObjective(objective_function, GRB.MINIMIZE)
                        model_iter.optimize()

                        status = model_iter.status
                        if status == GRB.OPTIMAL:
                            min_objective = model_iter.getObjective().getValue()
                        else:
                            min_objective = lb[i]

                        if not facet_left_removed[0, i]:
                            lb_iter = lb.copy()
                            lb_iter[i] = lb_iter[i] - 1
                            model_iter.setObjective(objective_function, GRB.MINIMIZE)
                            model_iter.optimize()

                            status = model_iter.status
                            if status == GRB.OPTIMAL:
                                min_objective2 = model_iter.getObjective().getValue()
                                if np.abs(min_objective2 - min_objective) > redundant_facet_tol:
                                    redundant_facet_left = False
                                else:
                                    removed += 1
                                    facet_left_removed[0, i] = True

                        if (not redundant_facet_left) or (not redundant_facet_right):
                            width = abs(max_objective - min_objective)

                            if width < redundant_facet_tol:
                                offset += 1
                                Aeq_res = sp.vstack([Aeq_res, A[i, :]], format="csr")
                                beq_res = np.append(beq_res, min(max_objective, min_objective))
                                ub[i] = sys.float_info.max
                                lb[i] = -sys.float_info.max
                            else:
                                indices_iter.append(i)

                                if not redundant_facet_left:
                                    A_res = sp.vstack([A_res, A[n + i, :]], format="lil")
                                    b_res.append(b[n + i, 0])
                                else:
                                    lb[i] = -sys.float_info.max

                                if not redundant_facet_right:
                                    A_res = sp.vstack([A_res, A[i, :]], format="lil")
                                    b_res.append(b[i, 0])
                                else:
                                    ub[i] = sys.float_info.max
                        else:
                            ub[i] = sys.float_info.max
                            lb[i] = -sys.float_info.max

                b_res = np.asarray(b_res)
                A_res = A_res.tocsr()

                return A_res, b_res, Aeq_res, beq_res
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
    except AttributeError:
        print("Gurobi solver failed.")
