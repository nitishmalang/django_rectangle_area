# Inside the loop
for i in indices:
    # Update the objective function
    objective_function = A[i, :]
    model.setMObjective(
        None, objective_function, 0.0, None, None, x, GRB.MINIMIZE
    )
    model.update()

    # Remove the old constraint and add the updated constraint
    model.remove(model.getConstrByName("c"))
    model.addMConstr(Aeq_sparse, x, "=", beq, name="c")
    model.update()
