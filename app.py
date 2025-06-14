# app.py
# Main file for the MathWiz API

import os
import uuid
import re
import io

# Core Frameworks
from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Math & Plotting Libraries
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt

# --- App Setup & Configuration ---

app = FastAPI(
    title="MathWiz API",
    description="A REST API to solve mathematical expressions, explain steps, plot graphs, and much more. Like WolframAlpha, but in an API.",
    version="1.0.0",
)

# Create directories for static files if they don't exist
os.makedirs("static/plots", exist_ok=True)
os.makedirs("static/latex", exist_ok=True)

# --- API Key Authentication ---

API_KEY = os.environ.get("API_KEY", "your-secret-key-for-local-dev")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Dependency to check for API key in the header."""
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- Pydantic Models for Input Validation ---

class ExpressionInput(BaseModel):
    expression: str = Field(..., example="3 * (x + 2)", title="A mathematical expression string.")

class EquationInput(BaseModel):
    equation: str = Field(..., example="x**2 - 4 = 0", title="An algebraic equation with an equals sign.")

class DifferentiateInput(BaseModel):
    expression: str = Field(..., example="x**2 + sin(x)", title="The function to differentiate.")
    variable: str = Field("x", example="x", title="The variable to differentiate with respect to.")

class WordProblemInput(BaseModel):
    question: str = Field(..., example="A car travels 250 km in 5 hours. What is its speed?", title="A math word problem in English.")

class MCQInput(BaseModel):
    question: str = Field(..., example="If x = 5, what is the value of x^2 + 3*x - 10?", title="A multiple-choice question.")
    options: list[str] = Field(..., example=["20", "25", "30", "35"], title="A list of possible answers as strings.")

class LatexInput(BaseModel):
    latex: str = Field(..., example=r"\frac{d}{dx} (x^2 + \sin(x)) = 2x + \cos(x)", title="A LaTeX mathematical string.")

class InequalityInput(BaseModel):
    inequality: str = Field(..., example="x**2 - 9 > 0", title="A mathematical inequality.")

# --- Helper Functions ---

def parse_safe_expr(expr_str: str):
    """Safely parse a mathematical expression string using sympy."""
    try:
        # These transformations allow for things like "2x" to be parsed as "2*x"
        transformations = (standard_transformations + (implicit_multiplication_application,))
        return parse_expr(expr_str, transformations=transformations, evaluate=False)
    except (SyntaxError, TypeError, sympy.SympifyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid mathematical expression: {str(e)}")


def get_base_url(request: Request):
    """Constructs the base URL for generating static file links."""
    return str(request.base_url)

# --- API Endpoints ---

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with a link to the documentation."""
    return {
        "message": "Welcome to the MathWiz API!",
        "documentation": "/docs"
    }

@app.post("/evaluate", tags=["Core Math"], dependencies=[Depends(get_api_key)])
async def evaluate_expression(data: ExpressionInput):
    """Solves a simple or complex mathematical expression."""
    try:
        parsed_expr = parse_safe_expr(data.expression)
        result = parsed_expr.evalf()
        return {"result": float(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/solve-equation", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def solve_equation_endpoint(data: EquationInput):
    """Solves algebraic equations for a variable (usually 'x')."""
    try:
        if "=" not in data.equation:
            raise HTTPException(status_code=400, detail="Invalid equation. Please include an equals sign '='.")

        lhs, rhs = map(parse_safe_expr, data.equation.split('='))
        equation = sympy.Eq(lhs, rhs)
        solutions = sympy.solve(equation)
        
        # Format solutions nicely
        formatted_solutions = [str(s.evalf()) for s in solutions]

        # Basic steps (can be enhanced)
        steps = [
            f"Given equation: {sympy.pretty(equation)}",
            f"Rearranging to solve for the variable.",
            f"The solutions are: {', '.join(formatted_solutions)}"
        ]

        return {"solutions": formatted_solutions, "steps": steps}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not solve equation: {str(e)}")

@app.post("/plot-graph", tags=["Graphing"], dependencies=[Depends(get_api_key)])
async def plot_graph(data: ExpressionInput, request: Request):
    """Generates and returns a URL to a plot of a 2D function (e.g., y = x**2)."""
    try:
        # Assume the expression is in terms of 'x' for a 'y = f(x)' plot
        x = sympy.symbols('x')
        expr = parse_safe_expr(data.expression)
        
        # Create a numerical function from the symbolic expression
        func = sympy.lambdify(x, expr, 'numpy')

        x_vals = np.linspace(-10, 10, 400)
        y_vals = func(x_vals)

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f"y = {data.expression}")
        plt.title(f"Graph of y = {data.expression}")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.grid(True)
        plt.legend()
        
        # Save plot to a unique file
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("static/plots", filename)
        plt.savefig(filepath)
        plt.close() # Important to free up memory

        image_url = f"{get_base_url(request)}static/plots/{filename}"
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not plot graph: {str(e)}")

@app.post("/differentiate", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def differentiate_expression(data: DifferentiateInput):
    """Calculates the derivative of an expression with respect to a variable."""
    try:
        expr = parse_safe_expr(data.expression)
        variable = sympy.symbols(data.variable)
        derivative = sympy.diff(expr, variable)
        return {"result": str(derivative)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not differentiate: {str(e)}")

@app.post("/integrate", tags=["Calculus"], dependencies=[Depends(get_api_key)])
async def integrate_expression(data: ExpressionInput):
    """Calculates the indefinite integral of an expression."""
    try:
        expr = parse_safe_expr(data.expression)
        # Assume integration with respect to 'x' if not specified
        variable = sympy.symbols('x') 
        integral = sympy.integrate(expr, variable)
        return {"indefinite_integral": f"{str(integral)} + C"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not integrate: {str(e)}")

@app.post("/word-to-equation", tags=["Problem Solving"], dependencies=[Depends(get_api_key)])
async def word_to_equation_endpoint(data: WordProblemInput):
    """
    (Simplified) Converts a simple word problem into a math equation and solves it.
    This is a rule-based example. A full implementation would require a powerful NLP model.
    """
    question = data.question.lower()
    # Example: "A car travels 100 km in 2 hours. Find speed."
    if "speed" in question and "km" in question and "hour" in question:
        numbers = re.findall(r'\d+\.?\d*', data.question)
        if len(numbers) >= 2:
            distance, time = float(numbers[0]), float(numbers[1])
            equation = f"speed = {distance} / {time}"
            answer = distance / time
            return {"equation": equation, "answer": f"{answer} km/hr"}

    # Example: "What is 5 plus 10?"
    if "what is" in question:
        expression_part = question.replace("what is", "").replace("?", "").strip()
        expression_part = expression_part.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("divided by", "/")
        try:
            result = sympy.sympify(expression_part).evalf()
            return {"equation": expression_part, "answer": str(result)}
        except Exception:
            pass

    return JSONResponse(
        status_code=400,
        content={"error": "Could not understand the word problem. This endpoint has limited capabilities."}
    )

@app.post("/mcq-solver", tags=["Problem Solving"], dependencies=[Depends(get_api_key)])
async def mcq_solver(data: MCQInput):
    """Solves a mathematical MCQ by evaluating the question and matching the answer."""
    # Find the mathematical expression in the question, e.g., after "value of" or "what is"
    match = re.search(r'(?:what is|value of|evaluate|solve)\s*(.*)\?', data.question, re.IGNORECASE)
    if not match:
        raise HTTPException(status_code=400, detail="Could not extract a mathematical expression from the question.")
    
    expression_str = match.group(1).strip()
    
    # Extract variable assignments like "If x = 5,"
    var_assignments = re.findall(r'(\w+)\s*=\s*(-?\d+\.?\d*)', data.question)
    subs_dict = {sympy.symbols(var): float(val) for var, val in var_assignments}
    
    try:
        expr = parse_safe_expr(expression_str)
        if subs_dict:
            result = expr.subs(subs_dict).evalf()
        else:
            result = expr.evalf()

        # Compare result with options
        for i, option in enumerate(data.options):
            try:
                # Use a small tolerance for floating point comparisons
                if abs(float(option) - float(result)) < 1e-9:
                    option_letter = chr(ord('A') + i)
                    return {"answer": option, "option": option_letter}
            except ValueError:
                continue # Option is not a number

        return JSONResponse(
            status_code=404,
            content={"error": "Could not find a matching answer in the options provided."}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing MCQ: {str(e)}")

@app.post("/latex-to-image", tags=["Utility"], dependencies=[Depends(get_api_key)])
async def latex_to_image(data: LatexInput, request: Request):
    """Converts a LaTeX string into a clean image and returns its URL."""
    try:
        fig, ax = plt.subplots(figsize=(6, 1), dpi=300)
        # Use matplotlib's mathtext for rendering - no LaTeX installation needed
        ax.text(0.5, 0.5, f"${data.latex}$", size=15, ha='center', va='center')
        ax.axis('off')

        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("static/latex", filename)

        # Save with a transparent background
        plt.savefig(filepath, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close(fig)

        image_url = f"{get_base_url(request)}static/latex/{filename}"
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render LaTeX: {e}")

@app.post("/inequality-solver", tags=["Algebra"], dependencies=[Depends(get_api_key)])
async def solve_inequality(data: InequalityInput):
    """Solves univariate inequalities."""
    try:
        x = sympy.symbols('x')
        inequality = parse_safe_expr(data.inequality)
        
        # Check if it's a valid relational (inequality) type
        if not isinstance(inequality, sympy.logic.relational.Relational):
             raise ValueError("Input is not a valid inequality (e.g., x**2 > 4).")

        solution = sympy.solve_univariate_inequality(inequality, x, relational=False)
        
        steps = [
            f"Given inequality: {inequality}",
            "Finding critical points by solving the corresponding equation.",
            "Testing intervals around the critical points.",
            f"The final solution set is: {solution}"
        ]
        
        return {"solution": str(solution), "steps": steps}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not solve inequality: {str(e)}")


# Serve static files (plots, latex images)
# Note: In a real production setup, you'd use a CDN, but this is fine for Render.
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# To run this app locally:
# 1. Make sure you have all packages from requirements.txt installed.
# 2. Set the API_KEY environment variable (optional for local dev).
# 3. Run: uvicorn app:app --reload
