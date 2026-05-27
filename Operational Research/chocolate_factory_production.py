import pyomo.environ as pyo
 
model=pyo.ConcreteModel()

model.x=pyo.Var(within=pyo.NonNegativeReals) #for milk chocolate
model.y=pyo.Var(within=pyo.NonNegativeReals) # for dark chocolate 

model.profit=pyo.Objective(expr=50*model.x+70*model.y,sense=pyo.maximize)

model.const1=pyo.Constraint(expr=2*model.x+3*model.y<=100)
model.const2=pyo.Constraint(expr=model.x+2*model.y<=70)

solver=pyo.SolverFactory('glpk')
solver.solve(model)

print(f"x= {model.x():.2f} , y= {model.y():.2f}")
print(f"Maximum Profit: {model.profit():.2f}")