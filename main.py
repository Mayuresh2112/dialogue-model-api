# Import necessary libraries
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model-api")

# Set variables
model_name = "distilgpt2"  # Original model name
model_path = "./fine_tuned_dialogue_model"  # Path to your fine-tuned model

# Models and tokenizer will be loaded during startup
tokenizer = None
original_model = None
fine_tuned_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the input schema
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

# Define the comparison result schema
class ComparisonResult(BaseModel):
    prompt: str
    original_output: str
    fine_tuned_output: str

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Create templates/index.html
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Dialogue Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
        }
        input[type="number"], input[type="text"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .output-container {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        .output-box {
            flex: 1;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            background-color: #f9f9f9;
        }
        .output-box h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .parameters {
            display: flex;
            gap: 15px;
        }
        .parameter {
            flex: 1;
        }
        .loading {
            text-align: center;
            display: none;
            margin: 20px 0;
        }
        #error-message {
            color: red;
            display: none;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Dialogue Generator</h1>
    
    <div class="form-group">
        <label for="prompt">Prompt:</label>
        <textarea id="prompt" placeholder="Enter your prompt here...">Title: The Ancient Temple
Objective: Explore the ruins
Dialogue:</textarea>
    </div>
    
    <div class="parameters">
        <div class="parameter">
            <label for="max-length">Max Length:</label>
            <input type="number" id="max-length" value="100" min="10" max="500">
        </div>
        
        <div class="parameter">
            <label for="temperature">Temperature:</label>
            <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
        </div>
        
        <div class="parameter">
            <label for="top-p">Top P:</label>
            <input type="number" id="top-p" value="0.9" min="0.1" max="1.0" step="0.05">
        </div>
    </div>
    
    <div style="margin-top: 20px; text-align: center;">
        <button id="generate-btn">Generate Both Outputs</button>
    </div>
    
    <div id="error-message"></div>
    
    <div id="loading" class="loading">
        <p>Generating text... This may take a moment.</p>
    </div>
    
    <div class="output-container">
        <div class="output-box">
            <h3>Original Model Output</h3>
            <div id="original-output">Output will appear here...</div>
        </div>
        
        <div class="output-box">
            <h3>Fine-tuned Model Output</h3>
            <div id="fine-tuned-output">Output will appear here...</div>
        </div>
    </div>
    
    <script>
        document.getElementById('generate-btn').addEventListener('click', async () => {
            const prompt = document.getElementById('prompt').value;
            const maxLength = document.getElementById('max-length').value;
            const temperature = document.getElementById('temperature').value;
            const topP = document.getElementById('top-p').value;
            
            if (!prompt) {
                showError('Please enter a prompt');
                return;
            }
            
            showLoading(true);
            hideError();
            
            try {
                const response = await fetch('/compare-single', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_length: parseInt(maxLength),
                        temperature: parseFloat(temperature),
                        top_p: parseFloat(topP),
                        do_sample: true
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to generate text');
                }
                
                const result = await response.json();
                
                document.getElementById('original-output').textContent = result.original_output;
                document.getElementById('fine-tuned-output').textContent = result.fine_tuned_output;
            } catch (error) {
                showError(error.message || 'An error occurred');
            } finally {
                showLoading(false);
            }
        });
        
        function showError(message) {
            const errorElement = document.getElementById('error-message');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
        
        function hideError() {
            document.getElementById('error-message').style.display = 'none';
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
    </script>
</body>
</html>
    """)

# Lifespan context manager for model loading/unloading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    load_models()
    yield
    # Clean up resources on shutdown
    global original_model, fine_tuned_model
    original_model = None
    fine_tuned_model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Mount templates
templates = Jinja2Templates(directory="templates")

# Function to load models
def load_models():
    global tokenizer, original_model, fine_tuned_model
    
    logger.info("Loading model and tokenizer...")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if fine-tuned model exists
    if os.path.exists(model_path):
        logger.info(f"Fine-tuned model found at {model_path}")
    else:
        logger.error(f"Fine-tuned model not found at {model_path}")
        raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
    
    # Load models
    logger.info(f"Loading original model from {model_name} ({device})")
    original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    logger.info(f"Loading model from {model_path} ({device})")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    logger.info("Models loaded successfully")

# Function to generate text from a prompt
def generate_dialogue(model, prompt, max_length=100, temperature=0.7, top_p=0.9, do_sample=True):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Root endpoint - HTML interface
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check endpoint
@app.get("/health")
async def health_check():
    if fine_tuned_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "device": str(device)}

# Generate text from fine-tuned model endpoint
@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        generated_text = generate_dialogue(
            fine_tuned_model, 
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        return {"prompt": request.prompt, "generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Compare a single prompt with both models
@app.post("/compare-single", response_model=ComparisonResult)
async def compare_single(request: GenerationRequest):
    try:
        # Generate from both models
        original_output = generate_dialogue(
            original_model,
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        
        fine_tuned_output = generate_dialogue(
            fine_tuned_model,
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        
        return {
            "prompt": request.prompt,
            "original_output": original_output,
            "fine_tuned_output": fine_tuned_output
        }
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Compare multiple prompts with both models
@app.post("/compare-multiple")
async def compare_multiple(request: GenerationRequest):
    try:
        prompts = [request.prompt]
        results = []
        
        for prompt in prompts:
            # Generate from both models
            original_output = generate_dialogue(
                original_model,
                prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample
            )
            
            fine_tuned_output = generate_dialogue(
                fine_tuned_model,
                prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample
            )
            
            results.append({
                "prompt": prompt,
                "original_output": original_output,
                "fine_tuned_output": fine_tuned_output
            })
        
        return results
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API with uvicorn if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
