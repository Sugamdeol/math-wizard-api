# app.py
# MathWiz API v2.0.0 - Enhanced Version with More Features

import os
import uuid
import re
import io
import base64
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

# Core Frameworks
from fastapi import FastAPI, Request, HTTPException, Security, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware

# Math & Science Libraries
import sympy
from sympy import symbols, solve, diff, integrate, limit, series, factorial, binomial
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

# --- App Setup & Configuration ---
app = FastAPI(
    title="MathWiz API",
    description="Advanced Mathematical Computation API with enhanced features for algebra, calculus, statistics, and visualization.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Error Handling ---
class MathError(Exception):
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}

@app.exception_handler(MathError)
async def math_error_handler(request: Request, exc: MathError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
            "type": type(exc).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# --- API Key Setup ---
API_KEY = os.environ.get("API_KEY", "test-api-key-for-development")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(status_code=403, detail="API Key required")
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Invalid API Key")

# --- Enhanced Pydantic Models ---
class ExpressionInput(BaseModel):
    expression: str
    variables: Optional[Dict[str, float]] = None
    
    @validator('expression')
    def validate_expression(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Expression cannot be empty")
        return v.strip()

class EquationInput(BaseModel):
    equation: str
    variable: str = "x"
    domain: Optional[str] = "real"  # real, complex, integer, positive

class SystemOfEquationsInput(BaseModel):
    equations: List[str]
    variables: List[str]

class PlotInput(BaseModel):
    expressions: List[str]
    x_range: List[float] = [-10, 10]
    y_range: Optional[List[float]] = None
    title: Optional[str] = "Mathematical Plot"
    labels: Optional[List[str]] = None
    grid: bool = True
    points: Optional[int] = 400

class Plot3DInput(BaseModel):
    expression: str
    x_range: List[float] = [-10, 10]
    y_range: List[float] = [-10, 10]
    title: Optional[str] = "3D Surface Plot"
    colormap: str = "viridis"

class CalculusInput(BaseModel):
    expression: str
    variable: str = "x"
    order: int = 1

class IntegralInput(BaseModel):
    expression: str
    variable: str = "x"
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    method: str = "auto"  # auto, numerical, symbolic

class LimitInput(BaseModel):
    expression: str
    variable: str = "x"
    value: float
    direction: str = "both"  # both, left, right

class SeriesInput(BaseModel):
    expression: str
    variable: str = "x"
    point: float = 0
    order: int = 5

class StatisticsInput(BaseModel):
    data: List[float]
    operation: str  # mean, median, mode, std, variance, quartiles, correlation

class ProbabilityInput(BaseModel):
    distribution: str  # normal, binomial, poisson, exponential
    parameters: Dict[str, float]
    operation: str  # pdf, cdf, sample, quantile
    value: Optional[float] = None

class MatrixInput(BaseModel):
    matrix: List[List[float]]
    operation: str  # determinant, inverse, eigenvalues, rank, trace, transpose

class MatrixOperationInput(BaseModel):
    matrix1: List[List[float]]
    matrix2: List[List[float]]
    operation: str  # add, subtract, multiply, dot

class FactorizationInput(BaseModel):
    expression: str
    method: str = "auto"  # auto, complete, partial

class SimplifyInput(BaseModel):
    expression: str
    expand: bool = False
    factor: bool = False
    collect: bool = False

class TrigonometryInput(BaseModel):
    expression: str
    simplify: bool = True
    to_degrees: bool = False

# --- Helper Functions ---
def parse_safe_expr(expr_str: str, variables: Dict[str, float] = None):
    """Safely parse mathematical expression with variable substitution"""
    try:
        # Clean the expression
        expr_str = expr_str.strip()
        
        # Add implicit multiplication support
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        # Parse the expression
        expr = parse_expr(expr_str, transformations=transformations, evaluate=False)
        
        # Substitute variables if provided
        if variables:
            subs_dict = {symbols(k): v for k, v in variables.items()}
            expr = expr.subs(subs_dict)
        
        return expr
    except Exception as e:
        raise MathError(
            f"Invalid mathematical expression",
            {"expression": expr_str, "error": str(e)}
        )

def create_plot_buffer(fig):
    """Create a BytesIO buffer from matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

def encode_plot_base64(fig):
    """Encode matplotlib figure as base64 string"""
    buf = create_plot_buffer(fig)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

# --- Core Endpoints ---
@app.get("/", tags=["General"])
async def root():
    return {
        "name": "MathWiz API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "algebra": ["/evaluate", "/solve-equation", "/system-equations", "/factorize", "/simplify"],
            "calculus": ["/differentiate", "/integrate", "/limit", "/series"],
            "plotting": ["/plot", "/plot-3d", "/plot-parametric", "/plot-polar"],
            "statistics": ["/statistics", "/probability", "/regression"],
            "linear_algebra": ["/matrix", "/matrix-operation"],
            "utilities": ["/latex", "/validate-expression", "/math-constants"]
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# --- Algebra Endpoints ---
@app.post("/evaluate", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def evaluate_expression(data: ExpressionInput):
    """Evaluate a mathematical expression with optional variable substitution"""
    expr = parse_safe_expr(data.expression, data.variables)
    result = expr.evalf()
    
    return {
        "success": True,
        "expression": data.expression,
        "variables": data.variables,
        "result": float(result) if result.is_real else complex(result),
        "latex": sympy.latex(expr),
        "simplified": str(sympy.simplify(expr))
    }

@app.post("/solve-equation", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def solve_equation(data: EquationInput):
    """Solve single equation with step-by-step solution"""
    if "=" not in data.equation:
        raise MathError("Invalid equation format. Must include '=' sign.")
    
    var = symbols(data.variable)
    lhs_str, rhs_str = data.equation.split('=', 1)
    lhs = parse_safe_expr(lhs_str)
    rhs = parse_safe_expr(rhs_str)
    equation = sympy.Eq(lhs, rhs)
    
    # Solve based on domain
    if data.domain == "complex":
        solutions = solve(equation, var, complex=True)
    elif data.domain == "integer":
        solutions = solve(equation, var, integer=True)
    elif data.domain == "positive":
        solutions = solve(equation, var, positive=True)
    else:
        solutions = solve(equation, var)
    
    # Generate steps
    steps = []
    steps.append(f"Original equation: {sympy.latex(equation)}")
    steps.append(f"Rearranging: {sympy.latex(lhs - rhs)} = 0")
    
    # Try to identify equation type
    poly = sympy.Poly(lhs - rhs, var, domain='ZZ')
    if poly.degree() == 1:
        steps.append("This is a linear equation")
    elif poly.degree() == 2:
        steps.append("This is a quadratic equation")
        a, b, c = poly.all_coeffs()
        disc = b**2 - 4*a*c
        steps.append(f"Discriminant: Δ = {disc}")
    elif poly.degree() == 3:
        steps.append("This is a cubic equation")
    
    formatted_solutions = []
    for sol in solutions:
        if sol.is_real:
            formatted_solutions.append(float(sol.evalf()))
        else:
            formatted_solutions.append(str(sol))
    
    return {
        "success": True,
        "equation": data.equation,
        "solutions": formatted_solutions,
        "steps": steps,
        "latex": sympy.latex(equation),
        "solution_count": len(solutions)
    }

@app.post("/system-equations", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def solve_system_equations(data: SystemOfEquationsInput):
    """Solve system of linear equations"""
    if len(data.equations) != len(data.variables):
        raise MathError("Number of equations must match number of variables")
    
    vars_symbols = symbols(data.variables)
    equations = []
    
    for eq_str in data.equations:
        if "=" not in eq_str:
            raise MathError(f"Invalid equation format: {eq_str}")
        lhs_str, rhs_str = eq_str.split('=', 1)
        lhs = parse_safe_expr(lhs_str)
        rhs = parse_safe_expr(rhs_str)
        equations.append(sympy.Eq(lhs, rhs))
    
    solutions = solve(equations, vars_symbols)
    
    formatted_solutions = {}
    for var, val in solutions.items():
        if val.is_real:
            formatted_solutions[str(var)] = float(val.evalf())
        else:
            formatted_solutions[str(var)] = str(val)
    
    return {
        "success": True,
        "equations": data.equations,
        "variables": data.variables,
        "solutions": formatted_solutions,
        "verification": {
            eq: str(eq.subs(solutions)) for eq in equations
        }
    }

@app.post("/factorize", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def factorize_expression(data: FactorizationInput):
    """Factorize algebraic expression"""
    expr = parse_safe_expr(data.expression)
    
    if data.method == "complete":
        factored = sympy.factor(expr, deep=True)
    elif data.method == "partial":
        factored = sympy.factor(expr, deep=False)
    else:
        factored = sympy.factor(expr)
    
    # Get prime factorization if it's a number
    prime_factors = None
    if expr.is_number and expr.is_integer:
        prime_factors = sympy.factorint(int(expr))
    
    return {
        "success": True,
        "original": str(expr),
        "factored": str(factored),
        "latex_original": sympy.latex(expr),
        "latex_factored": sympy.latex(factored),
        "prime_factors": prime_factors,
        "is_prime": sympy.isprime(expr) if expr.is_integer else None
    }

@app.post("/simplify", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def simplify_expression(data: SimplifyInput):
    """Simplify mathematical expression with various options"""
    expr = parse_safe_expr(data.expression)
    
    result = expr
    steps = []
    
    if data.expand:
        result = sympy.expand(result)
        steps.append(f"Expanded: {result}")
    
    if data.factor:
        result = sympy.factor(result)
        steps.append(f"Factored: {result}")
    
    if data.collect:
        result = sympy.collect(result, symbols('x'))
        steps.append(f"Collected: {result}")
    
    if not (data.expand or data.factor or data.collect):
        result = sympy.simplify(result)
        steps.append(f"Simplified: {result}")
    
    return {
        "success": True,
        "original": str(expr),
        "simplified": str(result),
        "latex_original": sympy.latex(expr),
        "latex_simplified": sympy.latex(result),
        "steps": steps
    }

# --- Calculus Endpoints ---
@app.post("/differentiate", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def differentiate(data: CalculusInput):
    """Calculate derivative of an expression"""
    expr = parse_safe_expr(data.expression)
    var = symbols(data.variable)
    
    derivatives = []
    current = expr
    
    for i in range(1, data.order + 1):
        current = diff(current, var)
        derivatives.append({
            "order": i,
            "derivative": str(current),
            "latex": sympy.latex(current),
            "simplified": str(sympy.simplify(current))
        })
    
    # Calculate critical points (where first derivative = 0)
    critical_points = []
    if data.order >= 1:
        first_deriv = diff(expr, var)
        critical = solve(first_deriv, var)
        critical_points = [float(p.evalf()) for p in critical if p.is_real]
    
    return {
        "success": True,
        "expression": data.expression,
        "variable": data.variable,
        "derivatives": derivatives,
        "critical_points": critical_points
    }

@app.post("/integrate", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def integrate_expression(data: IntegralInput):
    """Calculate integral (definite or indefinite)"""
    expr = parse_safe_expr(data.expression)
    var = symbols(data.variable)
    
    result = {}
    
    # Indefinite integral
    if data.lower_limit is None and data.upper_limit is None:
        integral = integrate(expr, var)
        result["indefinite_integral"] = f"{str(integral)} + C"
        result["latex"] = sympy.latex(integral) + " + C"
    
    # Definite integral
    else:
        if data.lower_limit is None or data.upper_limit is None:
            raise MathError("Both limits required for definite integral")
        
        if data.method == "numerical":
            # Use numerical integration for complex expressions
            from scipy import integrate as scipy_integrate
            func = sympy.lambdify(var, expr, 'numpy')
            value, error = scipy_integrate.quad(func, data.lower_limit, data.upper_limit)
            result["value"] = value
            result["error_estimate"] = error
            result["method"] = "numerical"
        else:
            integral = integrate(expr, (var, data.lower_limit, data.upper_limit))
            result["value"] = float(integral.evalf())
            result["exact_form"] = str(integral)
            result["method"] = "symbolic"
        
        result["lower_limit"] = data.lower_limit
        result["upper_limit"] = data.upper_limit
    
    result["success"] = True
    result["expression"] = data.expression
    
    return result

@app.post("/limit", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def calculate_limit(data: LimitInput):
    """Calculate limit of expression"""
    expr = parse_safe_expr(data.expression)
    var = symbols(data.variable)
    
    if data.direction == "left":
        result = limit(expr, var, data.value, '-')
    elif data.direction == "right":
        result = limit(expr, var, data.value, '+')
    else:
        result = limit(expr, var, data.value)
    
    # Check if limit exists
    if data.direction == "both":
        left_limit = limit(expr, var, data.value, '-')
        right_limit = limit(expr, var, data.value, '+')
        exists = left_limit == right_limit
    else:
        exists = result != sympy.oo and result != -sympy.oo
    
    return {
        "success": True,
        "expression": data.expression,
        "variable": data.variable,
        "approaching": data.value,
        "direction": data.direction,
        "limit": str(result),
        "exists": exists,
        "is_infinity": result == sympy.oo or result == -sympy.oo,
        "latex": sympy.latex(result)
    }

@app.post("/series", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def taylor_series(data: SeriesInput):
    """Calculate Taylor/Maclaurin series expansion"""
    expr = parse_safe_expr(data.expression)
    var = symbols(data.variable)
    
    expansion = series(expr, var, data.point, data.order + 1)
    
    # Get individual terms
    terms = []
    for i in range(data.order + 1):
        coeff = expansion.coeff(var - data.point, i)
        if coeff != 0:
            terms.append({
                "degree": i,
                "coefficient": str(coeff),
                "term": str(coeff * (var - data.point)**i)
            })
    
    return {
        "success": True,
        "expression": data.expression,
        "expansion": str(expansion.removeO()),
        "center": data.point,
        "order": data.order,
        "terms": terms,
        "latex": sympy.latex(expansion.removeO()),
        "remainder": f"O(({data.variable}-{data.point})^{data.order + 1})"
    }

# --- Plotting Endpoints ---
@app.post("/plot", tags=["Plotting"], dependencies=[Depends(get_api_key)])
async def plot_functions(data: PlotInput):
    """Plot multiple mathematical functions"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = symbols('x')
    x_vals = np.linspace(data.x_range[0], data.x_range[1], data.points)
    
    for i, expr_str in enumerate(data.expressions):
        expr = parse_safe_expr(expr_str)
        func = sympy.lambdify(x, expr, 'numpy')
        y_vals = func(x_vals)
        
        label = data.labels[i] if data.labels and i < len(data.labels) else f"y = {expr_str}"
        ax.plot(x_vals, y_vals, linewidth=2, label=label)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(data.title, fontsize=14, fontweight='bold')
    
    if data.y_range:
        ax.set_ylim(data.y_range)
    
    if data.grid:
        ax.grid(True, alpha=0.3)
    
    ax.legend(loc='best')
    
    # Add zero lines
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    img_base64 = encode_plot_base64(fig)
    
    return {
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "format": "base64",
        "expressions": data.expressions
    }

@app.post("/plot-3d", tags=["Plotting"], dependencies=[Depends(get_api_key)])
async def plot_3d_surface(data: Plot3DInput):
    """Generate 3D surface plot"""
    x, y = symbols('x y')
    expr = parse_safe_expr(data.expression)
    func = sympy.lambdify((x, y), expr, 'numpy')
    
    x_vals = np.linspace(data.x_range[0], data.x_range[1], 50)
    y_vals = np.linspace(data.y_range[0], data.y_range[1], 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    try:
        Z = func(X, Y)
    except Exception as e:
        raise MathError(f"Error evaluating expression: {str(e)}")
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap=data.colormap, alpha=0.9, edgecolor='none')
    
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    ax.set_title(data.title, fontsize=14, fontweight='bold', pad=20)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    img_base64 = encode_plot_base64(fig)
    
    return {
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "format": "base64",
        "expression": data.expression
    }

@app.post("/plot-parametric", tags=["Plotting"], dependencies=[Depends(get_api_key)])
async def plot_parametric(x_expr: str, y_expr: str, t_range: List[float] = Query([-10, 10]), 
                         api_key: str = Depends(get_api_key)):
    """Plot parametric equations"""
    t = symbols('t')
    x_func = sympy.lambdify(t, parse_safe_expr(x_expr), 'numpy')
    y_func = sympy.lambdify(t, parse_safe_expr(y_expr), 'numpy')
    
    t_vals = np.linspace(t_range[0], t_range[1], 1000)
    x_vals = x_func(t_vals)
    y_vals = y_func(t_vals)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_vals, y_vals, linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Parametric Plot: x={x_expr}, y={y_expr}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    img_base64 = encode_plot_base64(fig)
    
    return {
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "x_equation": x_expr,
        "y_equation": y_expr,
        "parameter_range": t_range
    }

@app.post("/plot-polar", tags=["Plotting"], dependencies=[Depends(get_api_key)])
async def plot_polar(r_expr: str, theta_range: List[float] = Query([0, 2*np.pi]),
                     api_key: str = Depends(get_api_key)):
    """Plot polar equations"""
    theta = symbols('theta')
    r_func = sympy.lambdify(theta, parse_safe_expr(r_expr), 'numpy')
    
    theta_vals = np.linspace(theta_range[0], theta_range[1], 1000)
    r_vals = r_func(theta_vals)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(theta_vals, r_vals, linewidth=2)
    ax.set_title(f'Polar Plot: r = {r_expr}', fontsize=14, pad=20)
    ax.grid(True)
    
    img_base64 = encode_plot_base64(fig)
    
    return {
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "equation": r_expr,
        "theta_range": theta_range
    }

# --- Statistics Endpoints ---
@app.post("/statistics", tags=["Statistics"], dependencies=[Depends(get_api_key)])
async def calculate_statistics(data: StatisticsInput):
    """Calculate various statistical measures"""
    arr = np.array(data.data)
    
    if data.operation == "mean":
        result = {"mean": float(np.mean(arr))}
    elif data.operation == "median":
        result = {"median": float(np.median(arr))}
    elif data.operation == "mode":
        mode_result = stats.mode(arr)
        result = {"mode": float(mode_result.mode[0]), "count": int(mode_result.count[0])}
    elif data.operation == "std":
        result = {"std_dev": float(np.std(arr)), "variance": float(np.var(arr))}
    elif data.operation == "quartiles":
        result = {
            "Q1": float(np.percentile(arr, 25)),
            "Q2": float(np.percentile(arr, 50)),
            "Q3": float(np.percentile(arr, 75)),
            "IQR": float(np.percentile(arr, 75) - np.percentile(arr, 25))
        }
    elif data.operation == "summary":
        result = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "variance": float(np.var(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "Q1": float(np.percentile(arr, 25)),
            "Q3": float(np.percentile(arr, 75)),
            "skewness": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr))
        }
    else:
        raise MathError(f"Unknown operation: {data.operation}")
    
    return {
        "success": True,
        "data_points": len(data.data),
        "operation": data.operation,
        "result": result
    }

@app.post("/probability", tags=["Statistics"], dependencies=[Depends(get_api_key)])
async def probability_distributions(data: ProbabilityInput):
    """Work with probability distributions"""
    dist = data.distribution.lower()
    params = data.parameters
    
    result = {}
    
    if dist == "normal":
        mu = params.get("mean", 0)
        sigma = params.get("std", 1)
        
        if data.operation == "pdf" and data.value is not None:
            result["pdf"] = float(stats.norm.pdf(data.value, mu, sigma))
        elif data.operation == "cdf" and data.value is not None:
            result["cdf"] = float(stats.norm.cdf(data.value, mu, sigma))
        elif data.operation == "quantile" and data.value is not None:
            result["quantile"] = float(stats.norm.ppf(data.value, mu, sigma))
        elif data.operation == "sample":
            size = int(params.get("size", 10))
            result["samples"] = stats.norm.rvs(mu, sigma, size=size).tolist()
    
    elif dist == "binomial":
        n = int(params.get("n", 10))
        p = params.get("p", 0.5)
        
        if data.operation == "pmf" and data.value is not None:
            result["pmf"] = float(stats.binom.pmf(int(data.value), n, p))
        elif data.operation == "cdf" and data.value is not None:
            result["cdf"] = float(stats.binom.cdf(int(data.value), n, p))
        elif data.operation == "sample":
            size = int(params.get("size", 10))
            result["samples"] = stats.binom.rvs(n, p, size=size).tolist()
    
    elif dist == "poisson":
        lam = params.get("lambda", 1)
        
        if data.operation == "pmf" and data.value is not None:
            result["pmf"] = float(stats.poisson.pmf(int(data.value), lam))
        elif data.operation == "cdf" and data.value is not None:
            result["cdf"] = float(stats.poisson.cdf(int(data.value), lam))
    
    else:
        raise MathError(f"Unknown distribution: {dist}")
    
    result["success"] = True
    result["distribution"] = data.distribution
    result["parameters"] = params
    
    return result

@app.post("/regression", tags=["Statistics"], dependencies=[Depends(get_api_key)])
async def linear_regression(x_data: List[float], y_data: List[float], 
                           predict_x: Optional[float] = None,
                           api_key: str = Depends(get_api_key)):
    """Perform linear regression analysis"""
    if len(x_data) != len(y_data):
        raise MathError("X and Y data must have same length")
    
    x = np.array(x_data)
    y = np.array(y_data)
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    result = {
        "success": True,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_error": float(std_err),
        "equation": f"y = {slope:.4f}x + {intercept:.4f}"
    }
    
    if predict_x is not None:
        result["prediction"] = float(slope * predict_x + intercept)
        result["predicted_x"] = predict_x
    
    # Create regression plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.5, label='Data points')
    ax.plot(x, slope * x + intercept, 'r-', label=f'Regression line: {result["equation"]}')
    if predict_x is not None:
        ax.scatter([predict_x], [result["prediction"]], color='green', s=100, 
                  label=f'Prediction at x={predict_x}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    img_base64 = encode_plot_base64(fig)
    result["plot"] = f"data:image/png;base64,{img_base64}"
    
    return result

# --- Linear Algebra Endpoints ---
@app.post("/matrix", tags=["Linear Algebra"], dependencies=[Depends(get_api_key)])
async def matrix_operations(data: MatrixInput):
    """Perform various matrix operations"""
    matrix = sympy.Matrix(data.matrix)
    operation = data.operation.lower()
    
    result = {"success": True, "operation": operation}
    
    if operation == "determinant":
        if not matrix.is_square:
            raise MathError("Determinant requires a square matrix")
        result["determinant"] = float(matrix.det())
    
    elif operation == "inverse":
        if not matrix.is_square:
            raise MathError("Inverse requires a square matrix")
        if matrix.det() == 0:
            raise MathError("Matrix is singular (determinant = 0)")
        result["inverse"] = np.array(matrix.inv()).tolist()
    
    elif operation == "eigenvalues":
        if not matrix.is_square:
            raise MathError("Eigenvalues require a square matrix")
        eigenvals = matrix.eigenvals()
        result["eigenvalues"] = {str(k): v for k, v in eigenvals.items()}
        
        # Also get eigenvectors
        eigenvects = matrix.eigenvects()
        result["eigenvectors"] = []
        for eigenval, multiplicity, vectors in eigenvects:
            for v in vectors:
                result["eigenvectors"].append({
                    "eigenvalue": str(eigenval),
                    "vector": np.array(v).tolist()
                })
    
    elif operation == "rank":
        result["rank"] = matrix.rank()
    
    elif operation == "trace":
        if not matrix.is_square:
            raise MathError("Trace requires a square matrix")
        result["trace"] = float(matrix.trace())
    
    elif operation == "transpose":
        result["transpose"] = np.array(matrix.T).tolist()
    
    elif operation == "rref":
        rref_matrix, pivot_cols = matrix.rref()
        result["rref"] = np.array(rref_matrix).tolist()
        result["pivot_columns"] = list(pivot_cols)
    
    elif operation == "nullspace":
        nullspace = matrix.nullspace()
        result["nullspace"] = [np.array(v).tolist() for v in nullspace]
        result["nullity"] = len(nullspace)
    
    else:
        raise MathError(f"Unknown operation: {operation}")
    
    result["original_matrix"] = data.matrix
    result["shape"] = matrix.shape
    
    return result

@app.post("/matrix-operation", tags=["Linear Algebra"], dependencies=[Depends(get_api_key)])
async def matrix_arithmetic(data: MatrixOperationInput):
    """Perform operations between two matrices"""
    m1 = sympy.Matrix(data.matrix1)
    m2 = sympy.Matrix(data.matrix2)
    operation = data.operation.lower()
    
    result = {"success": True, "operation": operation}
    
    if operation == "add":
        if m1.shape != m2.shape:
            raise MathError("Matrices must have same dimensions for addition")
        result["result"] = np.array(m1 + m2).tolist()
    
    elif operation == "subtract":
        if m1.shape != m2.shape:
            raise MathError("Matrices must have same dimensions for subtraction")
        result["result"] = np.array(m1 - m2).tolist()
    
    elif operation == "multiply":
        if m1.shape[1] != m2.shape[0]:
            raise MathError(f"Cannot multiply: {m1.shape} x {m2.shape}")
        result["result"] = np.array(m1 * m2).tolist()
    
    elif operation == "dot":
        # Element-wise multiplication (Hadamard product)
        if m1.shape != m2.shape:
            raise MathError("Matrices must have same dimensions for element-wise multiplication")
        result["result"] = np.array(m1.multiply_elementwise(m2)).tolist()
    
    else:
        raise MathError(f"Unknown operation: {operation}")
    
    result["matrix1_shape"] = m1.shape
    result["matrix2_shape"] = m2.shape
    result["result_shape"] = sympy.Matrix(result["result"]).shape
    
    return result

# --- Utility Endpoints ---
@app.post("/latex", tags=["Utilities"], dependencies=[Depends(get_api_key)])
async def expression_to_latex(expression: str):
    """Convert expression to LaTeX format"""
    expr = parse_safe_expr(expression)
    
    return {
        "success": True,
        "expression": expression,
        "latex": sympy.latex(expr),
        "pretty": sympy.pretty(expr),
        "mathml": sympy.mathml(expr, printer='presentation')
    }

@app.post("/validate-expression", tags=["Utilities"], dependencies=[Depends(get_api_key)])
async def validate_expression(expression: str):
    """Validate if an expression is mathematically valid"""
    try:
        expr = parse_safe_expr(expression)
        variables = list(expr.free_symbols)
        
        return {
            "success": True,
            "valid": True,
            "expression": expression,
            "simplified": str(sympy.simplify(expr)),
            "variables": [str(v) for v in variables],
            "is_polynomial": expr.is_polynomial(),
            "is_rational": expr.is_rational_function(),
            "is_algebraic": expr.is_algebraic_expr()
        }
    except Exception as e:
        return {
            "success": False,
            "valid": False,
            "expression": expression,
            "error": str(e)
        }

@app.get("/math-constants", tags=["Utilities"])
async def get_math_constants():
    """Get common mathematical constants"""
    return {
        "success": True,
        "constants": {
            "pi": {
                "value": float(sympy.pi.evalf()),
                "latex": "\\pi",
                "description": "Ratio of circle's circumference to diameter"
            },
            "e": {
                "value": float(sympy.E.evalf()),
                "latex": "e",
                "description": "Euler's number, base of natural logarithm"
            },
            "golden_ratio": {
                "value": float(sympy.GoldenRatio.evalf()),
                "latex": "\\phi",
                "description": "Golden ratio: (1 + sqrt(5))/2"
            },
            "euler_gamma": {
                "value": float(sympy.EulerGamma.evalf()),
                "latex": "\\gamma",
                "description": "Euler-Mascheroni constant"
            },
            "catalan": {
                "value": float(sympy.Catalan.evalf()),
                "latex": "G",
                "description": "Catalan's constant"
            }
        }
    }

@app.post("/combinatorics", tags=["Utilities"], dependencies=[Depends(get_api_key)])
async def combinatorics(n: int, k: int, operation: str = "combination"):
    """Calculate combinatorial values"""
    if n < 0 or k < 0:
        raise MathError("n and k must be non-negative")
    
    if operation == "combination":
        result = binomial(n, k)
        formula = f"C({n},{k}) = {n}! / ({k}! * ({n}-{k})!)"
    elif operation == "permutation":
        result = factorial(n) / factorial(n - k)
        formula = f"P({n},{k}) = {n}! / ({n}-{k})!"
    elif operation == "factorial":
        result = factorial(n)
        formula = f"{n}!"
    else:
        raise MathError("Operation must be: combination, permutation, or factorial")
    
    return {
        "success": True,
        "operation": operation,
        "n": n,
        "k": k,
        "result": int(result),
        "formula": formula
    }

# Run the app (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
