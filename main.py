from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from typing import Optional

# Constants
BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "2e964804-1fee-4340-bb22-099f1e785ec1"
FLOW_ID = "061c364a-1280-4b49-a56b-586d5dee8c8f"
APPLICATION_TOKEN = "AstraCS:iQNdzmNUslRgkcfrJrEaTfTS:2d2adbb2b2c0b8994365af08a53570f059ff8e2d9733a9209ca5d292087f723e"
ENDPOINT = ""  # Default endpoint

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domains instead of "*" for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class ChatRequest(BaseModel):
    message: str
    tweaks: Optional[dict] = None
    endpoint: Optional[str] = None
    output_type: str = "chat"
    input_type: str = "chat"

# Function to interact with Langflow
def run_flow(
    message: str,
    endpoint: str,
    output_type: str = "chat",
    input_type: str = "chat",
    tweaks: Optional[dict] = None,
) -> dict:
    """
    Sends a message to the Langflow API and retrieves the response.
    """
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint or FLOW_ID}"
    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    if tweaks:
        payload["tweaks"] = tweaks

    headers = {"Authorization": f"Bearer {APPLICATION_TOKEN}", "Content-Type": "application/json"}

    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    return response.json()

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Endpoint to handle chatbot interaction.
    """
    try:
        response = run_flow(
            message=request.message,
            endpoint=request.endpoint or ENDPOINT,
            output_type=request.output_type,
            input_type=request.input_type,
            tweaks=request.tweaks,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
