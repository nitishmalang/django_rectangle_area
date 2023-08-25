with gp.Model(env=env) as model:
    x = model.addMVar(
        shape=n,
        vtype=GRB.CONTINUOUS,
        name="x",
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
    )
    model.update()
