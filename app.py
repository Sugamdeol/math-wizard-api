# app.py
# MathWiz API v1.2.0 - Final Consolidated Version

import os
import uuid
import re
import io

# Core Frameworks
from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Math & Plotting Libraries
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- App Setup & Configuration ---
app = FastAPI(
    title="MathWiz API",
    description="An advanced REST API to solve equations, perform calculus, plot 2D/3D graphs, handle matrices, and more.",
    version="1.2.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Professional Error Handling ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": f"An unexpected server error occurred: {type(exc).__name__}: {exc}"},
    )

# --- Directory and API Key Setup ---
os.makedirs("static/plots", exist_ok=True)
os.makedirs("static/latex", exist_ok=True)
API_KEY = os.environ.get("API_KEY", "your-secret-key-for-local-dev")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- Pydantic Models ---
class ExpressionInput(BaseModel):
    expression: str

class EquationInput(BaseModel):
    equation: str

class DifferentiateInput(BaseModel):
    expression: str
    variable: str = "x"

class WordProblemInput(BaseModel):
    question: str

class MCQInput(BaseModel):
    question: str
    options: list[str]

class LatexInput(BaseModel):
    latex: str

class InequalityInput(BaseModel):
    inequality: str

class DefiniteIntegralInput(BaseModel):
    expression: str
    variable: str = "x"
    lower_limit: float
    upper_limit: float

class MatrixInput(BaseModel):
    matrix: list[list[float]]
    operation: str

# --- Helper Functions ---
def parse_safe_expr(expr_str: str):
    try:
        transformations = (standard_transformations + (implicit_multiplication_application,))
        return parse_expr(expr_str, transformations=transformations, evaluate=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid mathematical expression: '{expr_str}'. Error: {e}")

def get_base_url(request: Request):
    return str(request.base_url)

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to MathWiz API v1.2", "documentation": "/docs"}

@app.post("/evaluate", tags=["Core Math"], dependencies=[Depends(get_api_key)])
async def evaluate_expression(data: ExpressionInput):
    parsed_expr = parse_safe_expr(data.expression)
    result = parsed_expr.evalf()
    return {"result": float(result)}

@app.post("/solve-equation", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def solve_equation_endpoint(data: EquationInput):
    if "=" not in data.equation:
        raise HTTPException(status_code=400, detail="Invalid equation. Must include an '=' sign.")
    
    x = sympy.symbols('x')
    lhs_str, rhs_str = data.equation.split('=', 1)
    lhs = parse_safe_expr(lhs_str)
    rhs = parse_safe_expr(rhs_str)
    
    equation = sympy.Eq(lhs - rhs, 0)
    solutions = sympy.solve(equation, x)
    formatted_solutions = [str(s.evalf(4)) for s in solutions]
    
    steps = [f"Original Equation: {sympy.pretty(sympy.Eq(lhs, rhs))}"]
    steps.append(f"Rearranged Form: {sympy.pretty(equation)}")

    poly = sympy.Poly(equation.lhs, x)
    if poly.degree() == 2:
        a, b, c = poly.all_coeffs()
        steps.append("The equation is a quadratic in the form ax^2 + bx + c = 0.")
        steps.append(f"Coefficients: a={a}, b={b}, c={c}")
        discriminant = b**2 - 4*a*c
        steps.append(f"Calculate discriminant (Δ = b^2 - 4ac): {b**2} - 4*({a})*({c}) = {discriminant}")
        if discriminant >= 0:
            steps.append("Using the quadratic formula: x = (-b ± sqrt(Δ)) / 2a")
    steps.append(f"Final Solutions: {', '.join(formatted_solutions)}")

    return {"solutions": formatted_solutions, "steps": steps}

@app.post("/plot-graph", tags=["Graphing"], dependencies=[Depends(get_api_key)])
async def plot_graph(data: ExpressionInput, request: Request):
    x = sympy.symbols('x')
    expr = parse_safe_expr(data.expression)
    func = sympy.lambdify(x, expr, 'numpy')
    x_vals = np.linspace(-10, 10, 400)
    y_vals = func(x_vals)
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label=f"y = {data.expression}")
    plt.title(f"Graph of y = {data.expression}"); plt.xlabel("x-axis"); plt.ylabel("y-axis"); plt.grid(True); plt.legend()
    filename = f"{uuid.uuid4()}.png"; filepath = os.path.join("static/plots", filename)
    plt.savefig(filepath); plt.close()
    return {"image_url": f"{get_base_url(request)}static/plots/{filename}"}

@app.post("/plot-graph-3d", tags=["Graphing"], dependencies=[Depends(get_api_key)])
async def plot_graph_3d(data: ExpressionInput, request: Request):
    x, y = sympy.symbols('x y')
    expr = parse_safe_expr(data.expression)
    func = sympy.lambdify((x, y), expr, 'numpy')
    x_vals = np.linspace(-10, 10, 50); y_vals = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x_vals, y_vals); Z = func(X, Y)
    fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x-axis'); ax.set_ylabel('y-axis'); ax.set_zlabel('z-axis')
    ax.set_title(f"3D Plot of z = {data.expression}", pad=20)
    filename = f"{uuid.uuid4()}.png"; filepath = os.path.join("static/plots", filename)
    plt.savefig(filepath); plt.close()
    return {"image_url": f"{get_base_url(request)}static/plots/{filename}"}

@app.post("/differentiate", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def differentiate_expression(data: DifferentiateInput):
    expr = parse_safe_expr(data.expression)
    variable = sympy.symbols(data.variable)
    derivative = sympy.diff(expr, variable)
    return {"result": str(derivative)}

@app.post("/integrate", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def integrate_expression(data: ExpressionInput):
    expr = parse_safe_expr(data.expression)
    variable = sympy.symbols('x') 
    integral = sympy.integrate(expr, variable)
    return {"indefinite_integral": f"{str(integral)} + C"}

@app.post("/definite-integral", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def definite_integral(data: DefiniteIntegralInput):
    expr = parse_safe_expr(data.expression)
    variable = sympy.symbols(data.variable)
    result = sympy.integrate(expr, (variable, data.lower_limit, data.upper_limit))
    return {"result": str(result.evalf(6))}

@app.post("/mcq-solver", tags=["Problem Solving"], dependencies=[Depends(get_api_key)])
async def mcq_solver(data: MCQInput):
    match = re.search(r'(?:what is|value of|evaluate|solve|is)\s*(.*)\?', data.question, re.IGNORECASE)
    if not match:
        raise HTTPException(status_code=400, detail="Could not extract a mathematical expression from the question.")
    expression_str = match.group(1).strip()
    var_assignments = re.findall(r'(\w+)\s*=\s*(-?\d+\.?\d*)', data.question)
    subs_dict = {sympy.symbols(var): float(val) for var, val in var_assignments}
    expr = parse_safe_expr(expression_str)
    result = expr.subs(subs_dict).evalf() if subs_dict else expr.evalf()
    for i, option_str in enumerate(data.options):
        try:
            if abs(float(option_str) - float(result)) < 1e-9:
                return {"answer": option_str, "option": chr(ord('A') + i), "calculated_result": float(result)}
        except ValueError: continue
    raise HTTPException(status_code=404, detail=f"Could not find a matching answer. Calculated result was {result}, but it's not in the options.")

@app.post("/inequality-solver", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def solve_inequality(data: InequalityInput):
    x = sympy.symbols('x')
    inequality_expr = parse_safe_expr(data.inequality)
    solution = sympy.solve_univariate_inequality(inequality_expr, x, relational=False)
    try:
        equality = sympy.Eq(inequality_expr.lhs, inequality_expr.rhs)
        critical_points = sympy.solve(equality, x)
        points_str = ", ".join([str(p.evalf(3)) for p in critical_points])
        steps = [f"Given inequality: {inequality_expr}", f"Critical points found at: x = {points_str}", f"The final solution set is: {solution}"]
    except Exception:
        steps = [f"Given inequality: {inequality_expr}", f"The final solution set is: {solution}"]
    return {"solution": str(solution), "steps": steps}

@app.post("/matrix-operations", tags=["Linear Algebra"], dependencies=[Depends(get_api_key)])
async def matrix_operations(data: MatrixInput):
    matrix = sympy.Matrix(data.matrix)
    operation = data.operation.lower()
    if not matrix.is_square: raise HTTPException(status_code=400, detail="Operation requires a square matrix.")
    if operation == "determinant":
        return {"operation": "determinant", "result": str(matrix.det().evalf())}
    elif operation == "inverse":
        if matrix.det() == 0: raise HTTPException(status_code=400, detail="Matrix is singular and cannot be inverted.")
        return {"operation": "inverse", "result": np.array(matrix.inv().evalf()).tolist()}
    elif operation == "eigenvalues":
        eigenvals = matrix.eigenvals()
        result = {str(k.evalf(4)): v for k, v in eigenvals.items()}
        return {"operation": "eigenvalues", "result": result}
    else:
        raise HTTPException(status_code=400, detail="Invalid operation. Choose from: determinant, inverse, eigenvalues.")

@app.post("/latex-to-image", tags=["Utility"], dependencies=[Depends(get_api_key)])
async def latex_to_image(data: LatexInput, request: Request):
    fig, ax = plt.subplots(figsize=(6, 1), dpi=300)
    ax.text(0.5, 0.5, f"${data.latex}$", size=15, ha='center', va='center')
    ax.axis('off')
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join("static/latex", filename)
    plt.savefig(filepath, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)
    return {"image_url": f"{get_base_url(request)}static/latex/{filename}"}

# --- Static File Serving ---
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
