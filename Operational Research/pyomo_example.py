#maximize 40x+30y subject t0 : 2x+y<=40 , x+2y<=50 , x,y>=0
import pyomo.environ as pyo
model=pyo.ConcreteModel()

model.x=pyo.Var(within=pyo.NonNegativeReals)
model.y=pyo.Var(within=pyo.NonNegativeReals)

model.profit=pyo.Objective(expr=40*model.x+30*model.y,sense=pyo.maximize)

model.const1=pyo.Constraint(expr=2*model.x+model.y<=40)
model.const2=pyo.Constraint(expr=model.x+2*model.y<=50)

solver=pyo.SolverFactory('glpk')
solver.solve(model)

print(f"x= {model.x():.2f} , y= {model.y():.2f}")
print(f"Maximum Profit: {model.profit():.2f}")