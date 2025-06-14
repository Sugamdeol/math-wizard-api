# 🔢 MathWiz API

A powerful REST API system that allows any AI model, chatbot, or web app to perform complex mathematical operations, including solving expressions, plotting graphs, calculus, and more.

## ✨ Features

- **High Performance**: Built with FastAPI and asynchronous from the ground up.
- **Secure**: Protected by API Key authentication.
- **Comprehensive**: Covers a wide range of math operations from basic arithmetic to calculus.
- **Visual**: Generates high-quality plots and LaTeX images on the fly.
- **Self-Documenting**: Interactive API documentation available at the `/docs` endpoint.
- **Deploy-Ready**: Includes configuration for one-click deployment to Render.

---

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **Framework**: FastAPI
- **Core Libraries**: SymPy (symbolic math), NumPy, Matplotlib
- **Server**: Uvicorn (development), Gunicorn (production)
- **Deployment**: Render

---

## 🚀 Deployment to Render

This project is configured for easy deployment on Render's free tier.

1.  **Fork and Clone**: Fork this repository to your own GitHub account and then clone it to your local machine.
2.  **Create a New Web Service on Render**:
    - Go to the [Render Dashboard](https://dashboard.render.com/) and click "New" -> "Web Service".
    - Connect the GitHub repository you just created.
    - Render will automatically detect the `render.yaml` file. Give your service a name (e.g., `mathwiz-api`) and approve the plan.
3.  **Deployment**: Render will build and deploy your application. The `render.yaml` file handles everything:
    - It installs dependencies from `requirements.txt`.
    - It runs the `gunicorn` production server.
    - It automatically generates a secure `API_KEY` for you.
4.  **Find Your API Key**: Once deployed, go to the "Environment" tab for your service on Render to find your auto-generated `API_KEY`. You will need this key to make requests.

---

## 💻 Local Development

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/mathwiz-api.git
    cd mathwiz-api
    ```
2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the development server**:
    ```bash
    uvicorn app:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

---

## ⚙️ API Usage

All endpoints require an `X-API-KEY` header for authentication.

**Header**: `X-API-KEY: your-render-api-key`

### Example Request (`/evaluate`)

```bash
curl -X POST "https://your-app-name.onrender.com/evaluate" \
-H "Content-Type: application/json" \
-H "X-API-KEY: your-render-api-key" \
-d '{
  "expression": "sqrt(16) + 2**5"
}'

# Response:
# { "result": 36.0 }
```

### Example Request (`/solve-equation`)

```bash
curl -X POST "https://your-app-name.onrender.com/solve-equation" \
-H "Content-Type: application/json" \
-H "X-API-KEY: your-render-api-key" \
-d '{
  "equation": "x**2 + 5*x + 6 = 0"
}'

# Response:
# {
#   "solutions": ["-2.00000000000000", "-3.00000000000000"],
#   "steps": [
#     "Given equation: Eq(x**2 + 5*x + 6, 0)",
#     "Rearranging to solve for the variable.",
#     "The solutions are: -2.00000000000000, -3.00000000000000"
#   ]
# }
```

Explore all endpoints and their inputs interactively by visiting the `/docs` route on your deployed API URL.
