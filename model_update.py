ef update_model(model, n, Aeq_sparse, beq, lb, ub, A_sparse, b, objective_function):
    """A function to update a gurobi model that solves a linear program
    Keyword arguments:
    model -- gurobi model
    n -- the dimension
    Aeq_sparse -- a sparse matrix s.t. Aeq_sparse x = beq
    beq -- a vector s.t. Aeq_sparse x = beq
    lb -- lower bounds for the variables, i.e., a n-dimensional vector
    ub -- upper bounds for the variables, i.e., a n-dimensional vector
    A_sparse -- a sparse matrix s.t. A_sparse x <= b
    b -- a vector matrix s.t. A_sparse x <= b
    objective_function -- the objective function, i.e., a n-dimensional vector
    """
    model.remove(model.getVars())
    model.update()
    model.remove(model.getConstrs())
    model.update()
    x = model.addMVar(
        shape=n,
        vtype=GRB.CONTINUOUS,
        name="x",
        lb=lb,
        ub=ub,
    )
    model.update()
    model.addMConstr(Aeq_sparse, x, "=", beq, name="c")
    model.update()
    model.addMConstr(A_sparse, x, "<", b, name="d")
    model.update()
    model.setMObjective(None, objective_function, 0.0, None, None, x, GRB.MINIMIZE)
    model.update()

    return model
