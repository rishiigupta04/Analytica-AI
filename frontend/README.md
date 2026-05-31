# Marketing BI Assistant Frontend

This is the React + Tailwind frontend that connects to the existing LangGraph pipeline.
It does not replace the Streamlit app and can run side-by-side.

## Quick start

1. Install dependencies.
2. Start the FastAPI backend.
3. Start the frontend dev server.

## Environment

The frontend uses `VITE_API_BASE_URL` to reach the backend.
If not set, it defaults to `http://localhost:8000`.

Example `.env` for the frontend:

```
VITE_API_BASE_URL=http://localhost:8000
```