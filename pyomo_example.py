import pyomo.environ as pyo

model = pyo.ConcreteModel()

# Decision variables
model.x = pyo.Var(within=pyo.NonNegativeReals)
model.y = pyo.Var(within=pyo.NonNegativeReals)

# Objective function
model.profit = pyo.Objective(expr=40*model.x + 30*model.y, sense=pyo.maximize)

# Constraints
model.con1 = pyo.Constraint(expr=2*model.x + model.y <= 40)
model.con2 = pyo.Constraint(expr=model.x + 2*model.y <= 50)

# Solve
solver = pyo.SolverFactory('glpk')
solver.solve(model)

# Results
print(f"x = {model.x():.2f}, y = {model.y():.2f}")
print(f"Max Profit = {model.profit():.2f}")
