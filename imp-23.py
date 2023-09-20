import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import math

# Define a class to handle LP model initialization and reuse
class LPModel:
    def __init__(self, n, Aeq_sparse, beq, lb, ub, A_sparse, b):
        self.model = gp.Model()
        self.x = self.model.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="x", lb=lb, ub=ub)
        self.Aeq_sparse = Aeq_sparse
        self.beq = beq
        self.lb = lb
        self.ub = ub
        self.A_sparse = A_sparse
        self.b = b

        # Add constraints
        self.model.addMConstr(self.Aeq_sparse, self.x, "=", self.beq, name="c")
        self.model.addMConstr(self.A_sparse, self.x, "<", self.b, name="d")

    def update_objective(self, objective_function):
        self.model.setMObjective(None, objective_function, 0.0, None, None, self.x, GRB.MINIMIZE)
        self.model.update()

    def optimize(self):
        self.model.optimize()

    def get_optimal_solution(self):
        return self.x.X, -self.model.getObjective().getValue()

# Define a function to perform FBA
def fast_fba(lb, ub, S, c):
    m, n = S.shape
    Aeq_sparse = sp.csr_matrix(S)
    beq = np.zeros(m)
    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")
    b = np.concatenate((ub, -lb), axis=0)
    b = np.asarray(b, dtype="float")
    b = np.ascontiguousarray(b, dtype="float")
    
    lp_model = LPModel(n, Aeq_sparse, beq, lb, ub, sp.csr_matrix(A), b)
    lp_model.update_objective(c)
    return fast_fba_lp_model(lp_model)

def fast_fba_lp_model(lp_model):
    lp_model.optimize()
    return lp_model.get_optimal_solution()

# Define a function to perform FVA
def fast_fva(lb, ub, S, c, opt_percentage=100):
    m, n = S.shape
    Aeq_sparse = sp.csr_matrix(S)
    beq = np.zeros(m)
    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")
    b = np.concatenate((ub, -lb), axis=0)
    b = np.asarray(b, dtype="float")
    b = np.ascontiguousarray(b, dtype="float")
    
    lp_model = LPModel(n, Aeq_sparse, beq, lb, ub, sp.csr_matrix(A), b)
    lp_model.update_objective(c)
    
    return fast_fva_lp_model(lp_model, opt_percentage)

def fast_fva_lp_model(lp_model, opt_percentage=100):
    min_fluxes = []
    max_fluxes = []
    lp_model.optimize()
    # Add your FVA implementation here using lp_model
    return min_fluxes, max_fluxes

# Define a function to update an LP model
def update_model(lp_model, n, Aeq_sparse, beq, lb, ub, A_sparse, b, objective_function):
    lp_model.model.remove(lp_model.x)
    lp_model.model.update()
    lp_model.model.remove(lp_model.model.getConstrs())
    lp_model.model.update()
    
    lp_model.x = lp_model.model.addMVar(
        shape=n,
        vtype=GRB.CONTINUOUS,
        name="x",
        lb=lb,
        ub=ub,
    )
    lp_model.model.update()
    lp_model.model.addMConstr(Aeq_sparse, lp_model.x, "=", beq, name="c")
    lp_model.model.update()
    lp_model.model.addMConstr(A_sparse, lp_model.x, "<", b, name="d")
    lp_model.model.update()
    lp_model.update_objective(objective_function)
    return lp_model

import numpy as np
import scipy.sparse as sp
from gurobipy import GRB

# ... (previous code)

def fast_remove_redundant_facets(lb, ub, S, c, opt_percentage=1e-3):
    m, n = S.shape
    Aeq_sparse = sp.csr_matrix(S)
    beq = np.zeros(m)
    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")
    b = np.concatenate((ub, -lb), axis=0)
    b = np.asarray(b, dtype="float")
    b = np.ascontiguousarray(b, dtype="float")

    # Initialize Aeq_res and beq_res
    Aeq_res = np.empty((0, n), float)
    beq_res = np.array(beq)

    lp_model = LPModel(n, Aeq_sparse, beq, lb, ub, sp.csr_matrix(A), b)
    lp_model.update_objective(np.array([-x for x in c]))

    # Initialize other variables
    lbx = lb.copy()
    ubx = ub.copy()
    indices_iter = list(range(n))
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

        for i in indices:
            objective_function = A[i, :]
            redundant_facet_right = True
            redundant_facet_left = True

            # For the maximum
            objective_function_max = np.array([-x for x in objective_function])

            # Make a copy of the LP model
            lp_model_copy = LPModel(n, Aeq_sparse, beq, lb, ub, sp.csr_matrix(A), b)
            lp_model_copy.update_objective(objective_function_max)
            lp_model_copy.optimize()

            # If optimized
            status = lp_model_copy.model.status
            if status == GRB.OPTIMAL:
                max_objective = -lp_model_copy.model.getObjective().getValue()
            else:
                max_objective = ub[i]

            if not facet_right_removed[0, i]:
                ub_iter = ub.copy()
                ub_iter[i] = ub_iter[i] + 1
                lp_model_copy = LPModel(n, Aeq_sparse, beq, lb, ub_iter, sp.csr_matrix(A), b)
                lp_model_copy.update_objective(objective_function_max)
                lp_model_copy.optimize()

                status = lp_model_copy.model.status
                if status == GRB.OPTIMAL:
                    max_objective2 = -lp_model_copy.model.getObjective().getValue()
                    if np.abs(max_objective2 - max_objective) > 1e-07:
                        redundant_facet_right = False
                    else:
                        removed += 1
                        facet_right_removed[0, i] = True

            lp_model_copy = LPModel(n, Aeq_sparse, beq, lb, ub, sp.csr_matrix(A), b)
            lp_model_copy.update_objective(objective_function)
            lp_model_copy.optimize()

            # If optimized
            status = lp_model_copy.model.status
            if status == GRB.OPTIMAL:
                min_objective = lp_model_copy.model.getObjective().getValue()
            else:
                min_objective = lb[i]

            if not facet_left_removed[0, i]:
                lb_iter = lb.copy()
                lb_iter[i] = lb_iter[i] - 1
                lp_model_copy = LPModel(n, Aeq_sparse, beq, lb_iter, ub, sp.csr_matrix(A), b)
                lp_model_copy.update_objective(objective_function)
                lp_model_copy.optimize()

                status = lp_model_copy.model.status
                if status == GRB.OPTIMAL:
                    min_objective2 = lp_model_copy.model.getObjective().getValue()
                    if np.abs(min_objective2 - min_objective) > 1e-07:
                        redundant_facet_left = False
                    else:
                        removed += 1
                        facet_left_removed[0, i] = True

            if (not redundant_facet_left) or (not redundant_facet_right):
                width = abs(max_objective - min_objective)
                if width < 1e-07:
                    offset += 1
                    Aeq_res = np.vstack((Aeq_res, A[i, :]))
                    beq_res = np.append(beq_res, min(max_objective, min_objective))
                    ub[i] = -sys.float_info.max
                    lb[i] = sys.float_info.max
                else:
                    indices_iter.append(i)
                    if not redundant_facet_left:
                        Aeq_res = np.vstack((Aeq_res, A[n + i, :]))
                        beq_res = np.append(beq_res, b[n + i])
                    else:
                        lb[i] = -sys.float_info.max

                    if not redundant_facet_right:
                        Aeq_res = np.vstack((Aeq_res, A[i, :]))
                        beq_res = np.append(beq_res, b[i])
                    else:
                        ub[i] = sys.float_info.max

    indices = np.setdiff1d(list(range(n)), indices_iter)
    lb_iter = lb.copy()
    ub_iter = ub.copy()
    for i in indices:
        # For the maximum
        objective_function_max = np.array([-x for x in A[i, :]])
        ub_iter[i] = ub_iter[i] + 1
        lp_model_copy = LPModel(n, Aeq_sparse, beq, lb, ub_iter, sp.csr_matrix(A), b)
        lp_model_copy.update_objective(objective_function_max)
        lp_model_copy.optimize()
        status = lp_model_copy.model.status
        if status == GRB.OPTIMAL:
            max_objective = -lp_model_copy.model.getObjective().getValue()
        else:
            max_objective = ub[i]

        lb_iter[i] = lb_iter[i] - 1
        lp_model_copy = LPModel(n, Aeq_sparse, beq, lb_iter, ub, sp.csr_matrix(A), b)
        lp_model_copy.update_objective(objective_function_max)
        lp_model_copy.optimize()
        status = lp_model_copy.model.status
        if status == GRB.OPTIMAL:
            min_objective = lp_model_copy.model.getObjective().getValue()
        else:
            min_objective = lb[i]

        if np.abs(max_objective - min_objective) < 1e-07:
            Aeq_res = np.vstack((Aeq_res, A[i, :]))
            beq_res = np.append(beq_res, max_objective)
            lb[i] = -sys.float_info.max
            ub[i] = sys.float_info.max

    return (
                    A_res,
                    b_res,
                    Aeq_res,
                    beq_res,
                )


# Define a function to compute the fast inner ball
def fast_inner_ball(lb, ub, S, c):
    m, n = S.shape
    Aeq_sparse = sp.csr_matrix(S)
    beq = np.zeros(m)
    A = np.zeros((2 * n, n), dtype="float")
    A[0:n] = np.eye(n)
    A[n:] -= np.eye(n, n, dtype="float")
    b = np.concatenate((ub, -lb), axis=0)
    b = np.asarray(b, dtype="float")
    b = np.ascontiguousarray(b, dtype="float")
    
    lp_model = LPModel(n, Aeq_sparse, beq, lb, ub, sp.csr_matrix(A), b)
    lp_model.update_objective(c)

    lbx = lb.copy()
    ubx = ub.copy()
    x = (lbx + ubx) / 2
    tol = 1e-8

    while max(ubx - lbx) >= tol:
        A_eq = Aeq_sparse.toarray()
        A_ub = A.toarray()
        res_eq = np.dot(A_eq, x) - beq
        res_ub = np.dot(A_ub, x) - b
        indices_eq = np.where(abs(res_eq) > tol)
        indices_ub = np.where(abs(res_ub) > tol)
        indices = np.unique(np.concatenate((indices_eq[0], indices_ub[0]), axis=0))
        
        indices_lb = np.where(x - lbx <= tol)
        indices_ub = np.where(ubx - x <= tol)
        
        indices = np.unique(np.concatenate((indices, indices_lb[0], indices_ub[0]), axis=0))
        
        if len(indices) == 0:
            break
        d = np.zeros(n, dtype="float")
        for i in indices:
            if (x[i] - lbx[i] <= tol) and (ubx[i] - x[i] <= tol):
                continue
            if abs(res_eq[i]) >= abs(res_ub[i]):
                d[i] = -res_eq[i] / np.linalg.norm(A_eq[i])
            else:
                d[i] = -res_ub[i] / np.linalg.norm(A_ub[i])
            if d[i] > 0:
                d[i] = -d[i]
        t = 1
        x += t * d
        lbx = np.maximum(lbx, x)
        ubx = np.minimum(ubx, x)
    
    return x, -lp_model.model.getObjective().getValue()

# Test the functions
if __name__ == "__main__":
    # Define example inputs
    lb = np.array([0, 0, 0, 0], dtype="float")
    ub = np.array([1, 1, 1, 1], dtype="float")
    S = np.array([
        [-1, 1, 0, 0],
        [0, -1, 1, 0],
        [0, 0, -1, 1]
    ], dtype="float")
    c = np.array([1, 1, 1, 1], dtype="float")

    # Test fast_fba
    result_fba = fast_fba(lb, ub, S, c)
    print("Fast FBA Result:")
    print("Optimal Solution:", result_fba[0])
    print("Optimal Objective Value:", result_fba[1])

    # Test fast_fva
    min_fluxes, max_fluxes = fast_fva(lb, ub, S, c)
    print("Fast FVA Result:")
    print("Minimum Fluxes:", min_fluxes)
    print("Maximum Fluxes:", max_fluxes)

    # Test fast_remove_redundant_facets
    removed_facets_lp = fast_remove_redundant_facets(lb, ub, S, c)
    print("Fast Remove Redundant Facets Result:")
    print("Optimal Solution:", removed_facets_lp.get_optimal_solution()[0])
    print("Optimal Objective Value:", removed_facets_lp.get_optimal_solution()[1])

    # Test fast_inner_ball
    result_inner_ball = fast_inner_ball(lb, ub, S, c)
    print("Fast Inner Ball Result:")
    print("Inner Ball Solution:", result_inner_ball[0])
    print("Inner Ball Objective Value:", result_inner_ball[1])
