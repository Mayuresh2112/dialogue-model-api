# dialogue-model-api


## Project Overview

This project creates a fine-tuned language model specialized in generating realistic NPC (Non-Player Character) dialogues for RPG games. The system provides a simple API that allows game developers to generate contextual dialogue based on prompts, enhancing the storytelling experience in games with dynamically generated content.

### Key Features

- Fine-tuned DistilGPT-2 model specifically for RPG dialogue generation
- FastAPI-based REST API for easy integration into game development workflows  
- Comparative output between original and fine-tuned models
- Web-based interface for testing dialogue generation
- Optimized for local deployment with minimal resource requirements

## Dataset

This project uses the "dprashar/npc_dialogue_rpg_quests" dataset from Hugging Face, which contains a variety of NPC dialogues from popular RPG games. The dataset provides a rich source of examples for the model to learn dialogue patterns, quest structures, and fantasy terminology.

## Model Architecture

The core of this project is a fine-tuned DistilGPT-2 model, which offers:
- Lightweight deployment (suitable for CPU usage)
- Good quality output for the specific task of dialogue generation
- Fast inference times, making it appropriate for real-time game applications

## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)


### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rpg-dialogue-generator.git
cd rpg-dialogue-generator
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models (if not included in the repository):
```bash
# This script will download both the original and fine-tuned models
python download_models.py
```

## Running the API

Start the API server with:
```bash
python main.py
```

By default, the server runs on `http://localhost:8000`. You can access:
- Web interface: `http://localhost:8000/`
- API documentation: `http://localhost:8000/docs`

## API Endpoints

### Generate Dialogue
- **Endpoint**: `/generate`
- **Method**: GET
- **Parameters**: 
  - `prompt` (string): The context or beginning of the dialogue
  - `model` (string, optional): Choose between "original" or "finetuned" (default: "finetuned")
  - `max_length` (integer, optional): Maximum length of generated text (default: 100)
- **Returns**: JSON with generated dialogue text

### Status
- **Endpoint**: `/status`
- **Method**: GET
- **Returns**: JSON with server status, model information, and uptime

## Usage Examples

### Via Web Interface

1. Navigate to `http://localhost:8000/` in your browser
2. Enter a prompt in the text box (e.g., "Greetings traveler, I need your help with")
3. Select the model (original or fine-tuned)
4. Click "Generate" to see the results


## Example Outputs

### Original Model
```
Prompt: "Greetings traveler, I need your help with"
Output: "Greetings traveler, I need your help with this matter. I've been trying to understand why this happens but I can't seem to figure it out. Maybe you could help me understand what's going on here. I've tried many different approaches but none have worked so far."
```

### Fine-tuned Model
```
Prompt: "Greetings traveler, I need your help with"
Output: "Greetings traveler, I need your help with a dire situation. The goblins from the Darkwood Forest have stolen our village's sacred crystal. Without it, our crops will fail and winter will be harsh. If you can recover the crystal from their hideout, I'll reward you with 500 gold coins and our village's eternal gratitude. Will you accept this quest?"
```

## Project Structure

```
rpg-dialogue-generator/
├── model.py               # Model training script
├── evaluationandcompare.py # Evaluation and comparison utilities
├── main.py                # FastAPI implementation
├── templates/
│   └── index.html         # Web interface
├── models/                # Directory for storing models
├── requirements.txt       # Project dependencies
└── README.md              # This file
```


## Future Improvements

With additional time, the following enhancements could be made:

- Improved model training with a larger dataset and more epochs
- Addition of dialogue categories (shopkeeper, quest giver, villain, etc.)
- Implementation of model quantization for even faster inference
- Adding sentiment and tone controls to the API
- Expanding to multi-language support for international game development
- Creation of a more sophisticated front-end with saved dialogue history


## Acknowledgements

- Hugging Face for the dataset and model architecture
- The open-source game development community for inspiration
