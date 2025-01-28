from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from typing import Optional
import os
import json

# Constants
BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "2e964804-1fee-4340-bb22-099f1e785ec1"
APPLICATION_TOKEN = "AstraCS:pwaTfsLYlrLlvZKvarxsnmwg:0a27001d2abe5a73bbe28e90ed72919677d62ff42d8c6b1943af195d3c120cde"
FLOW_FILE_PATH = "flow.json"  # Path to cached flow file

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

# Function to fetch the latest flow and cache it
def fetch_and_cache_flow():
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/flows"
    headers = {"Authorization": f"Bearer {APPLICATION_TOKEN}"}

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    flow_data = response.json()
    with open(FLOW_FILE_PATH, "w") as file:
        json.dump(flow_data, file)

    return flow_data

# Function to load the cached flow
def load_flow():
    if os.path.exists(FLOW_FILE_PATH):
        with open(FLOW_FILE_PATH, "r") as file:
            return json.load(file)
    else:
        return fetch_and_cache_flow()

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
    flow_data = load_flow()
    flow_id = endpoint or flow_data.get("flows", [{}])[0].get("id")  # Default to the first flow ID

    if not flow_id:
        raise HTTPException(status_code=400, detail="No valid flow ID found.")

    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{flow_id}"
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
            endpoint=request.endpoint,
            output_type=request.output_type,
            input_type=request.input_type,
            tweaks=request.tweaks,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
