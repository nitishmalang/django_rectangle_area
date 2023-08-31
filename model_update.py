def update_model(model, n, Aeq_sparse, beq, lb, ub, A_sparse, b, objective_function):
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

    # Get the variables
    x = model.getVars()
    # Update the bounds of the variables
    for i in range(n):
        x[i].lb = lb[i]
        x[i].ub = ub[i]

   
    # Update the constraints
    model.removeConstr(model.getConstrs())
    model.addMConstr(Aeq_sparse, x, "=", beq, name="c")
    model.addMConstr(A_sparse, x, "<", b, name="d")

   # Update the objective function
    model.setMObjective(None, objective_function, 0.0, None, None, x, GRB.MINIMIZE)

    model.update()

    return model
