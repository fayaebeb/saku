from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from typing import Optional

# Constants
BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "2e964804-1fee-4340-bb22-099f1e785ec1"
FLOW_ID = "380387dc-0f44-4a20-a12f-ea11ef39ec7e"
APPLICATION_TOKEN = "AstraCS:ZlLgIzZPZbxfSEQetWjXUNZZ:76957e36b802c6015dcf8d68128c8ea0aca6f53d1a22e490078fa0e6e3ff3b08"
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
