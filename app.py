# app.py
# MathWiz API v2.0.1 - Enhanced Version with Robust Validation and Utilities

import os
import io
import uuid
import re
import base64
from typing import Optional, List, Dict, Any
from datetime import datetime

# Core Frameworks
from fastapi import FastAPI, Request, HTTPException, Security, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Pydantic v1/v2 compatibility for validators
try:
    from pydantic import field_validator  # v2
except Exception:  # v1 fallback
    from pydantic import validator as field_validator  # type: ignore

from fastapi.middleware.cors import CORSMiddleware

# Math & Science Libraries
import sympy as sp
from sympy import symbols, Eq, solveset, S, diff, integrate, limit, series, factorial, binomial, Poly
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction, BooleanTrue, BooleanFalse, BooleanAtom, And, Or, Not
import numpy as np

# Optional SciPy (for statistics, probability, regression)
try:
    from scipy import stats
except Exception:
    stats = None  # Endpoints that need SciPy will error clearly if missing

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    pass

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

# --- App Setup & Configuration ---
app = FastAPI(
    title="MathWiz API",
    description="Advanced Mathematical Computation API with enhanced features for algebra, calculus, statistics, and visualization.",
    version="2.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
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

    @field_validator('expression')
    @classmethod
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
    points: int = 400

class Plot3DInput(BaseModel):
    expression: str
    x_range: List[float] = [-10, 10]
    y_range: List[float] = [-10, 10]
    title: Optional[str] = "3D Surface Plot"
    colormap: str = "viridis"

class ParametricPlotInput(BaseModel):
    x_expr: str
    y_expr: str
    t_range: List[float] = [-10, 10]
    points: int = 1000
    title: Optional[str] = None

class PolarPlotInput(BaseModel):
    r_expr: str
    theta_range: List[float] = [0, float(2*np.pi)]
    points: int = 1000
    title: Optional[str] = None

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
    operation: str  # mean, median, mode, std, variance, quartiles, summary, correlation

class ProbabilityInput(BaseModel):
    distribution: str  # normal, binomial, poisson, exponential
    parameters: Dict[str, float]
    operation: str  # pdf, cdf, pmf, sample, quantile
    value: Optional[float] = None

class RegressionInput(BaseModel):
    x_data: List[float]
    y_data: List[float]
    predict_x: Optional[float] = None

class MatrixInput(BaseModel):
    matrix: List[List[float]]
    operation: str  # determinant, inverse, eigenvalues, rank, trace, transpose, rref, nullspace

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

class LatexImageInput(BaseModel):
    latex: str
    dpi: int = 200
    fontsize: int = 16

class InequalityInput(BaseModel):
    inequality: str
    variable: str = "x"
    domain: str = "real"  # real only

# --- Helper Functions ---
def parse_safe_expr(expr_str: str, variables: Dict[str, float] = None):
    """Safely parse mathematical expression with optional variable substitution."""
    try:
        expr_str = expr_str.strip()
        transformations = (standard_transformations + (implicit_multiplication_application,))
        expr = parse_expr(expr_str, transformations=transformations, evaluate=False)
        if variables:
            subs_dict = {symbols(k): v for k, v in variables.items()}
            expr = expr.subs(subs_dict)
        return expr
    except Exception as e:
        raise MathError("Invalid mathematical expression", {"expression": expr_str, "error": str(e)})

def create_plot_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

def encode_plot_base64(fig):
    buf = create_plot_buffer(fig)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

def assert_pure_math(expr, ctx="expression"):
    """Reject boolean/relational expressions where pure math is required."""
    if isinstance(expr, (Relational, BooleanFunction, BooleanTrue, BooleanFalse, BooleanAtom, And, Or, Not)):
        raise MathError(
            f"{ctx}: boolean/relational operators are not allowed here.",
            {"hint": "Use /inequality-solver for inequalities or only arithmetic/trig/exponential forms here."}
        )
    return expr

def to_jsonable_number(val: sp.Expr) -> Any:
    """Convert a SymPy value to JSON-safe type."""
    try:
        if getattr(val, "is_number", False):
            if getattr(val, "is_real", None) is True:
                return float(val.evalf())
            # Complex numbers as string
            return str(val.evalf())
        return float(val)  # might work for basic types
    except Exception:
        return str(val)

# --- Core Endpoints ---
@app.get("/", tags=["General"])
async def root():
    return {
        "name": "MathWiz API",
        "version": "2.0.1",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "algebra": ["/evaluate", "/solve-equation", "/system-equations", "/factorize", "/simplify", "/inequality-solver"],
            "calculus": ["/differentiate", "/integrate", "/limit", "/series"],
            "plotting": ["/plot", "/plot-3d", "/plot-parametric", "/plot-polar"],
            "statistics": ["/statistics", "/probability", "/regression"],
            "linear_algebra": ["/matrix", "/matrix-operation"],
            "utilities": ["/latex", "/latex-to-image", "/validate-expression", "/math-constants", "/combinatorics"]
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# --- Algebra Endpoints ---
@app.post("/evaluate", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def evaluate_expression(data: ExpressionInput):
    expr = parse_safe_expr(data.expression, data.variables)
    assert_pure_math(expr, "evaluate")
    result = expr.evalf()
    return {
        "success": True,
        "expression": data.expression,
        "variables": data.variables,
        "result": to_jsonable_number(result),
        "latex": sp.latex(expr),
        "simplified": str(sp.simplify(expr))
    }

@app.post("/solve-equation", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def solve_equation(data: EquationInput):
    if "=" not in data.equation:
        raise MathError("Invalid equation format. Must include '=' sign.")
    # Proactively block inequalities and '!=' here
    bad_ops = ["!=", "<=", ">=", "<", ">"]
    if any(op in data.equation for op in bad_ops):
        raise MathError(
            "Unsupported operator in /solve-equation.",
            {"received": data.equation, "detail": "Only a single '=' is allowed. Use /inequality-solver for <, >, <=, >= or '!='."}
        )

    var = symbols(data.variable)
    lhs_str, rhs_str = data.equation.split('=', 1)
    lhs = parse_safe_expr(lhs_str)
    rhs = parse_safe_expr(rhs_str)

    # Also block boolean/relational expressions
    assert_pure_math(lhs, "solve-equation.lhs")
    assert_pure_math(rhs, "solve-equation.rhs")

    equation = Eq(lhs, rhs)

    # Domain handling via solveset
    domain_map = {
        "real": S.Reals,
        "complex": S.Complexes
    }
    domain = domain_map.get(data.domain, S.Reals)
    solset = solveset(lhs - rhs, var, domain=domain)

    # Post-filtering for 'integer' and 'positive'
    solutions = []
    try:
        iterable = list(solset) if hasattr(solset, "__iter__") else [solset]
    except Exception:
        iterable = [solset]

    for sol in iterable:
        try:
            if data.domain == "integer":
                if sol.is_real and sol.is_integer:
                    solutions.append(sol)
            elif data.domain == "positive":
                if sol.is_real and sol.evalf() > 0:
                    solutions.append(sol)
            else:
                solutions.append(sol)
        except Exception:
            solutions.append(sol)

    # Steps and type hints
    steps = []
    steps.append(f"Original equation: {sp.latex(equation)}")
    steps.append(f"Rearranging: {sp.latex(lhs - rhs)} = 0")

    try:
        poly = Poly(lhs - rhs, var)
        deg = poly.degree()
        if deg == 1:
            steps.append("This is a linear equation.")
        elif deg == 2:
            steps.append("This is a quadratic equation.")
            a, b, c = poly.all_coeffs()
            disc = b**2 - 4*a*c
            steps.append(f"Discriminant: Δ = {sp.simplify(disc)}")
        elif deg == 3:
            steps.append("This is a cubic equation.")
        else:
            steps.append(f"Polynomial degree detected: {deg}")
    except Exception:
        steps.append("Non-polynomial or symbolic equation type detected.")

    formatted_solutions = [to_jsonable_number(sp.nsimplify(sol)) for sol in solutions]

    return {
        "success": True,
        "equation": data.equation,
        "solutions": formatted_solutions,
        "steps": steps,
        "latex": sp.latex(equation),
        "solution_count": len(formatted_solutions)
    }

@app.post("/system-equations", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def solve_system_equations(data: SystemOfEquationsInput):
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
        assert_pure_math(lhs, "system-equations.lhs")
        assert_pure_math(rhs, "system-equations.rhs")
        equations.append(sp.Eq(lhs, rhs))

    solutions = sp.solve(equations, vars_symbols, dict=True)

    formatted_solutions = []
    for sol in solutions:
        formatted = {}
        for var, val in sol.items():
            formatted[str(var)] = to_jsonable_number(val)
        formatted_solutions.append(formatted)

    return {
        "success": True,
        "equations": data.equations,
        "variables": data.variables,
        "solutions": formatted_solutions,
    }

@app.post("/factorize", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def factorize_expression(data: FactorizationInput):
    expr = parse_safe_expr(data.expression)
    assert_pure_math(expr, "factorize")
    if data.method == "complete":
        factored = sp.factor(expr, deep=True)
    elif data.method == "partial":
        factored = sp.factor(expr, deep=False)
    else:
        factored = sp.factor(expr)

    prime_factors = None
    if expr.is_number and expr.is_integer:
        prime_factors = sp.factorint(int(expr))

    return {
        "success": True,
        "original": str(expr),
        "factored": str(factored),
        "latex_original": sp.latex(expr),
        "latex_factored": sp.latex(factored),
        "prime_factors": prime_factors,
        "is_prime": sp.isprime(expr) if expr.is_integer else None
    }

@app.post("/simplify", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def simplify_expression(data: SimplifyInput):
    expr = parse_safe_expr(data.expression)
    assert_pure_math(expr, "simplify")
    result = expr
    steps = []
    if data.expand:
        result = sp.expand(result)
        steps.append(f"Expanded: {result}")
    if data.factor:
        result = sp.factor(result)
        steps.append(f"Factored: {result}")
    if data.collect:
        # Collect on x if present; otherwise generic simplify
        x = symbols('x')
        result = sp.collect(result, x) if x in result.free_symbols else sp.simplify(result)
        steps.append(f"Collected: {result}")
    if not (data.expand or data.factor or data.collect):
        result = sp.simplify(result)
        steps.append(f"Simplified: {result}")
    return {
        "success": True,
        "original": str(expr),
        "simplified": str(result),
        "latex_original": sp.latex(expr),
        "latex_simplified": sp.latex(result),
        "steps": steps
    }

# --- Calculus Endpoints ---
@app.post("/differentiate", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def differentiate(data: CalculusInput):
    expr = parse_safe_expr(data.expression)
    assert_pure_math(expr, "differentiate")
    var = symbols(data.variable)
    derivatives = []
    current = expr
    for i in range(1, data.order + 1):
        current = diff(current, var)
        derivatives.append({
            "order": i,
            "derivative": str(current),
            "latex": sp.latex(current),
            "simplified": str(sp.simplify(current))
        })
    critical_points = []
    if data.order >= 1:
        first_deriv = diff(expr, var)
        try:
            crit = sp.solve(first_deriv, var)
            critical_points = [to_jsonable_number(c) for c in crit if getattr(c, "is_real", None)]
        except Exception:
            critical_points = []
    return {
        "success": True,
        "expression": data.expression,
        "variable": data.variable,
        "derivatives": derivatives,
        "critical_points": critical_points
    }

@app.post("/integrate", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def integrate_expression(data: IntegralInput):
    expr = parse_safe_expr(data.expression)
    assert_pure_math(expr, "integrate")
    var = symbols(data.variable)
    result = {}
    if data.lower_limit is None and data.upper_limit is None:
        integ = integrate(expr, var)
        result["indefinite_integral"] = f"{str(integ)} + C"
        result["latex"] = sp.latex(integ) + " + C"
    else:
        if data.lower_limit is None or data.upper_limit is None:
            raise MathError("Both limits required for definite integral")
        if data.method == "numerical":
            if stats is None:
                raise MathError("SciPy required for numerical integration (stats missing).")
            from scipy import integrate as scipy_integrate  # local import
            f = sp.lambdify(var, expr, 'numpy')
            value, error = scipy_integrate.quad(f, data.lower_limit, data.upper_limit)
            result["value"] = float(value)
            result["error_estimate"] = float(error)
            result["method"] = "numerical"
        else:
            integ = integrate(expr, (var, data.lower_limit, data.upper_limit))
            result["value"] = float(integ.evalf())
            result["exact_form"] = str(integ)
            result["method"] = "symbolic"
        result["lower_limit"] = data.lower_limit
        result["upper_limit"] = data.upper_limit
    result["success"] = True
    result["expression"] = data.expression
    return result

@app.post("/limit", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def calculate_limit(data: LimitInput):
    expr = parse_safe_expr(data.expression)
    assert_pure_math(expr, "limit")
    var = symbols(data.variable)
    if data.direction == "left":
        res = limit(expr, var, data.value, '-')
    elif data.direction == "right":
        res = limit(expr, var, data.value, '+')
    else:
        res = limit(expr, var, data.value)
    if data.direction == "both":
        left = limit(expr, var, data.value, '-')
        right = limit(expr, var, data.value, '+')
        exists = left == right
    else:
        exists = res != sp.oo and res != -sp.oo
    return {
        "success": True,
        "expression": data.expression,
        "variable": data.variable,
        "approaching": data.value,
        "direction": data.direction,
        "limit": str(res),
        "exists": bool(exists),
        "is_infinity": bool(res == sp.oo or res == -sp.oo),
        "latex": sp.latex(res)
    }

@app.post("/series", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def taylor_series(data: SeriesInput):
    expr = parse_safe_expr(data.expression)
    assert_pure_math(expr, "series")
    var = symbols(data.variable)
    expansion = series(expr, var, data.point, data.order + 1).removeO()
    terms = []
    # Extract terms up to given order (simple approach)
    poly_like = sp.expand(expansion)
    for i in range(0, data.order + 1):
        term = sp.expand(sp.series(expr, var, data.point, i + 1).removeO())
        # Capture exact i-th degree coefficient around the point
        try:
            coeff = sp.expand((poly_like.subs(var, var + data.point))).series(var, 0, i + 1).removeO().coeff(var, i)
        except Exception:
            coeff = sp.S(0)
        terms.append({
            "degree": i,
            "coefficient": str(coeff),
        })
    return {
        "success": True,
        "expression": data.expression,
        "expansion": str(expansion),
        "center": data.point,
        "order": data.order,
        "terms": terms,
        "latex": sp.latex(expansion),
        "remainder": f"O(({data.variable}-{data.point})^{data.order + 1})"
    }

# --- Plotting Endpoints ---
@app.post("/plot", tags=["Plotting"], dependencies=[Depends(get_api_key)])
async def plot_functions(data: PlotInput):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = symbols('x')
    x_vals = np.linspace(data.x_range[0], data.x_range[1], data.points)
    for i, expr_str in enumerate(data.expressions):
        expr = parse_safe_expr(expr_str)
        assert_pure_math(expr, "plot")
        func = sp.lambdify(x, expr, 'numpy')
        y_vals = func(x_vals)
        label = data.labels[i] if data.labels and i < len(data.labels) else f"y = {expr_str}"
        ax.plot(x_vals, y_vals, linewidth=2, label=label)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(data.title or "Plot", fontsize=14, fontweight='bold')
    if data.y_range:
        ax.set_ylim(data.y_range)
    if data.grid:
        ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
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
    x, y = symbols('x y')
    expr = parse_safe_expr(data.expression)
    assert_pure_math(expr, "plot-3d")
    func = sp.lambdify((x, y), expr, 'numpy')
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
async def plot_parametric(data: ParametricPlotInput):
    t = symbols('t')
    x_func = sp.lambdify(t, parse_safe_expr(data.x_expr), 'numpy')
    y_func = sp.lambdify(t, parse_safe_expr(data.y_expr), 'numpy')
    t_vals = np.linspace(data.t_range[0], data.t_range[1], data.points)
    x_vals = x_func(t_vals)
    y_vals = y_func(t_vals)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_vals, y_vals, linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(data.title or f'Parametric: x={data.x_expr}, y={data.y_expr}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    img_base64 = encode_plot_base64(fig)
    return {
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "x_equation": data.x_expr,
        "y_equation": data.y_expr,
        "parameter_range": data.t_range
    }

@app.post("/plot-polar", tags=["Plotting"], dependencies=[Depends(get_api_key)])
async def plot_polar(data: PolarPlotInput):
    theta = symbols('theta')
    r_func = sp.lambdify(theta, parse_safe_expr(data.r_expr), 'numpy')
    theta_vals = np.linspace(data.theta_range[0], data.theta_range[1], data.points)
    r_vals = r_func(theta_vals)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(theta_vals, r_vals, linewidth=2)
    ax.set_title(data.title or f'Polar: r = {data.r_expr}', fontsize=14, pad=20)
    ax.grid(True)
    img_base64 = encode_plot_base64(fig)
    return {
        "success": True,
        "image": f"data:image/png;base64,{img_base64}",
        "equation": data.r_expr,
        "theta_range": data.theta_range
    }

# --- Inequality Solver ---
@app.post("/inequality-solver", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def inequality_solver(data: InequalityInput):
    x = symbols(data.variable)
    ineq = parse_safe_expr(data.inequality)
    if not isinstance(ineq, Relational):
        raise MathError("Provide a valid inequality (e.g., x < 2, x >= 3, x != 1)")
    steps = [f"Given inequality: {str(ineq)}"]
    # Handle '!=' separately
    if getattr(ineq, "rel_op", None) == "!=":
        # x != a -> Reals \ {solutions of x = a}
        eq_set = solveset(Eq(ineq.lhs, ineq.rhs), x, domain=S.Reals)
        solution = sp.Complement(S.Reals, eq_set)
        steps.append(f"Critical points (excluded): {str(eq_set)}")
    else:
        # Use solve_univariate_inequality for others
        try:
            solution = sp.solve_univariate_inequality(ineq, x, relational=False)
        except Exception as e:
            raise MathError("Unable to solve inequality", {"error": str(e)})
    steps.append(f"The final solution set is: {str(solution)}")
    return {"success": True, "solution": str(solution), "steps": steps}

# --- Statistics Endpoints ---
@app.post("/statistics", tags=["Statistics"], dependencies=[Depends(get_api_key)])
async def calculate_statistics(data: StatisticsInput):
    if stats is None:
        raise MathError("SciPy is required for statistics endpoints (scipy not installed).")
    arr = np.array(data.data)
    op = data.operation.lower()
    if op == "mean":
        result = {"mean": float(np.mean(arr))}
    elif op == "median":
        result = {"median": float(np.median(arr))}
    elif op == "mode":
        mode_result = stats.mode(arr, keepdims=True)
        result = {"mode": float(mode_result.mode[0]), "count": int(mode_result.count[0])}
    elif op == "std":
        result = {"std_dev": float(np.std(arr)), "variance": float(np.var(arr))}
    elif op == "quartiles":
        result = {
            "Q1": float(np.percentile(arr, 25)),
            "Q2": float(np.percentile(arr, 50)),
            "Q3": float(np.percentile(arr, 75)),
            "IQR": float(np.percentile(arr, 75) - np.percentile(arr, 25))
        }
    elif op == "summary":
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
    if stats is None:
        raise MathError("SciPy is required for probability endpoints (scipy not installed).")
    dist = data.distribution.lower()
    params = data.parameters
    op = data.operation.lower()
    result = {}
    if dist == "normal":
        mu = params.get("mean", 0.0)
        sigma = params.get("std", 1.0)
        if op == "pdf" and data.value is not None:
            result["pdf"] = float(stats.norm.pdf(data.value, mu, sigma))
        elif op == "cdf" and data.value is not None:
            result["cdf"] = float(stats.norm.cdf(data.value, mu, sigma))
        elif op == "quantile" and data.value is not None:
            result["quantile"] = float(stats.norm.ppf(data.value, mu, sigma))
        elif op == "sample":
            size = int(params.get("size", 10))
            result["samples"] = stats.norm.rvs(mu, sigma, size=size).tolist()
        else:
            raise MathError(f"Missing or invalid operation/value for normal: {op}")
    elif dist == "binomial":
        n = int(params.get("n", 10))
        p = float(params.get("p", 0.5))
        if op == "pmf" and data.value is not None:
            result["pmf"] = float(stats.binom.pmf(int(data.value), n, p))
        elif op == "cdf" and data.value is not None:
            result["cdf"] = float(stats.binom.cdf(int(data.value), n, p))
        elif op == "sample":
            size = int(params.get("size", 10))
            result["samples"] = stats.binom.rvs(n, p, size=size).tolist()
        else:
            raise MathError(f"Missing or invalid operation/value for binomial: {op}")
    elif dist == "poisson":
        lam = float(params.get("lambda", 1.0))
        if op == "pmf" and data.value is not None:
            result["pmf"] = float(stats.poisson.pmf(int(data.value), lam))
        elif op == "cdf" and data.value is not None:
            result["cdf"] = float(stats.poisson.cdf(int(data.value), lam))
        elif op == "sample":
            size = int(params.get("size", 10))
            result["samples"] = stats.poisson.rvs(lam, size=size).tolist()
        else:
            raise MathError(f"Missing or invalid operation/value for poisson: {op}")
    else:
        raise MathError(f"Unknown distribution: {dist}")
    result["success"] = True
    result["distribution"] = data.distribution
    result["parameters"] = params
    return result

@app.post("/regression", tags=["Statistics"], dependencies=[Depends(get_api_key)])
async def linear_regression(data: RegressionInput):
    if stats is None:
        raise MathError("SciPy is required for regression endpoints (scipy not installed).")
    if len(data.x_data) != len(data.y_data):
        raise MathError("X and Y data must have same length")
    x = np.array(data.x_data)
    y = np.array(data.y_data)
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
    if data.predict_x is not None:
        result["prediction"] = float(slope * data.predict_x + intercept)
        result["predicted_x"] = data.predict_x
    # Plot regression
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.5, label='Data')
    ax.plot(x, slope * x + intercept, 'r-', label=f'Line: {result["equation"]}')
    if data.predict_x is not None:
        ax.scatter([data.predict_x], [result["prediction"]], color='green', s=100, label=f'Prediction at x={data.predict_x}')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Linear Regression'); ax.legend(); ax.grid(True, alpha=0.3)
    img_base64 = encode_plot_base64(fig)
    result["plot"] = f"data:image/png;base64,{img_base64}"
    return result

# --- Linear Algebra Endpoints ---
@app.post("/matrix", tags=["Linear Algebra"], dependencies=[Depends(get_api_key)])
async def matrix_operations(data: MatrixInput):
    matrix = sp.Matrix(data.matrix)
    operation = data.operation.lower()
    result = {"success": True, "operation": operation, "shape": matrix.shape, "original_matrix": data.matrix}
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
        eigenvects = matrix.eigenvects()
        result["eigenvectors"] = []
        for eigenval, multiplicity, vectors in eigenvects:
            for v in vectors:
                result["eigenvectors"].append({"eigenvalue": str(eigenval), "vector": np.array(v).tolist()})
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
    return result

@app.post("/matrix-operation", tags=["Linear Algebra"], dependencies=[Depends(get_api_key)])
async def matrix_arithmetic(data: MatrixOperationInput):
    m1 = sp.Matrix(data.matrix1)
    m2 = sp.Matrix(data.matrix2)
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
        if m1.shape != m2.shape:
            raise MathError("Matrices must have same dimensions for element-wise multiplication")
        result["result"] = np.array(m1.multiply_elementwise(m2)).tolist()
    else:
        raise MathError(f"Unknown operation: {operation}")
    result["matrix1_shape"] = m1.shape
    result["matrix2_shape"] = m2.shape
    result["result_shape"] = sp.Matrix(result["result"]).shape
    return result

# --- Utility Endpoints ---
@app.post("/latex", tags=["Utilities"], dependencies=[Depends(get_api_key)])
async def expression_to_latex(expression: str):
    expr = parse_safe_expr(expression)
    return {
        "success": True,
        "expression": expression,
        "latex": sp.latex(expr),
        "pretty": sp.pretty(expr),
        "mathml": sp.mathml(expr, printer='presentation')
    }

@app.post("/latex-to-image", tags=["Utilities"], dependencies=[Depends(get_api_key)])
async def latex_to_image_base64(data: LatexImageInput):
    fig, ax = plt.subplots(figsize=(6, 1), dpi=data.dpi)
    ax.axis('off')
    ax.text(0.5, 0.5, f"${data.latex}$", size=data.fontsize, ha='center', va='center')
    img_b64 = encode_plot_base64(fig)
    return {
        "success": True,
        "format": "base64",
        "image": f"data:image/png;base64,{img_b64}"
    }

@app.post("/validate-expression", tags=["Utilities"], dependencies=[Depends(get_api_key)])
async def validate_expression(expression: str):
    try:
        expr = parse_safe_expr(expression)
        variables = list(expr.free_symbols)
        return {
            "success": True,
            "valid": True,
            "expression": expression,
            "simplified": str(sp.simplify(expr)),
            "variables": [str(v) for v in variables],
            "is_polynomial": bool(getattr(expr, "is_polynomial", lambda: None)()),
            "is_rational": bool(getattr(expr, "is_rational_function", lambda: None)()),
            "is_algebraic": bool(getattr(expr, "is_algebraic_expr", lambda: None)())
        }
    except Exception as e:
        return {"success": False, "valid": False, "expression": expression, "error": str(e)}

@app.get("/math-constants", tags=["Utilities"])
async def get_math_constants():
    return {
        "success": True,
        "constants": {
            "pi": {"value": float(sp.pi.evalf()), "latex": "\\pi", "description": "Ratio of circle's circumference to diameter"},
            "e": {"value": float(sp.E.evalf()), "latex": "e", "description": "Euler's number"},
            "golden_ratio": {"value": float(sp.GoldenRatio.evalf()), "latex": "\\phi", "description": "Golden ratio: (1 + sqrt(5))/2"},
            "euler_gamma": {"value": float(sp.EulerGamma.evalf()), "latex": "\\gamma", "description": "Euler-Mascheroni constant"},
            "catalan": {"value": float(sp.Catalan.evalf()), "latex": "G", "description": "Catalan's constant"}
        }
    }

@app.post("/combinatorics", tags=["Utilities"], dependencies=[Depends(get_api_key)])
async def combinatorics(n: int, k: int = 0, operation: str = "combination"):
    if n < 0 or k < 0:
        raise MathError("n and k must be non-negative")
    if operation == "combination":
        result = binomial(n, k)
        formula = f"C({n},{k}) = {n}! / ({k}! * ({n}-{k})!)"
    elif operation == "permutation":
        if n < k:
            raise MathError("For permutations, n must be >= k")
        result = factorial(n) / factorial(n - k)
        formula = f"P({n},{k}) = {n}! / ({n}-{k})!"
    elif operation == "factorial":
        result = factorial(n)
        formula = f"{n}!"
    else:
        raise MathError("Operation must be: combination, permutation, or factorial")
    return {"success": True, "operation": operation, "n": n, "k": k, "result": int(result), "formula": formula}

# Run (local dev)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
