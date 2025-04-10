# Import necessary libraries
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
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

# Define the comparison request schema
class ComparisonRequest(BaseModel):
    prompts: List[str]
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

# Define the result schema
class GenerationResult(BaseModel):
    prompt: str
    generated_text: str

# Define the comparison result schema
class ComparisonResult(BaseModel):
    prompt: str
    original_output: str
    fine_tuned_output: str

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
    logger.info(f"Loading model from {model_path} ({device})")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    logger.info(f"Loading original model from {model_name} ({device})")
    original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    logger.info("Models loaded successfully")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Dialogue Generation API", "status": "running"}

# Health check endpoint
@app.get("/health")
async def health_check():
    if fine_tuned_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "device": str(device)}

# Generate text endpoint
@app.post("/generate", response_model=GenerationResult)
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

# Compare models endpoint
@app.post("/compare", response_model=List[ComparisonResult])
async def compare_models(request: ComparisonRequest):
    try:
        results = []
        for prompt in request.prompts:
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

# Save comparison results endpoint
@app.post("/save-comparison")
async def save_comparison(request: ComparisonRequest, background_tasks: BackgroundTasks):
    try:
        # Add the task to background tasks
        background_tasks.add_task(
            save_comparison_to_file, 
            request.prompts,
            request.max_length,
            request.temperature,
            request.top_p,
            request.do_sample
        )
        return {"message": "Comparison task started in background"}
    except Exception as e:
        logger.error(f"Error initiating comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task to save comparison results to file
def save_comparison_to_file(prompts, max_length, temperature, top_p, do_sample):
    output_file = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(output_file, "w") as f:
            # Write header with timestamp
            f.write(f"===== MODEL COMPARISON RESULTS =====\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original model: {model_name}\n")
            f.write(f"Fine-tuned model: {model_path}\n\n")
            
            # Compare outputs
            logger.info("Generating comparison results")
            for i, prompt in enumerate(prompts, 1):
                # Write to file
                f.write(f"PROMPT {i}: {prompt}\n\n")
                
                # Original model output
                original_output = generate_dialogue(
                    original_model, 
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                
                # Write to file
                f.write("ORIGINAL MODEL OUTPUT:\n")
                f.write(original_output)
                f.write("\n\n")
                
                # Fine-tuned model output
                fine_tuned_output = generate_dialogue(
                    fine_tuned_model, 
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                
                # Write to file
                f.write("FINE-TUNED MODEL OUTPUT:\n")
                f.write(fine_tuned_output)
                f.write("\n\n")
                
                # Separator
                separator = "-" * 50
                f.write(f"{separator}\n\n")
        
        logger.info(f"Comparison results saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving comparison results: {str(e)}")

# Run the API with uvicorn if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
