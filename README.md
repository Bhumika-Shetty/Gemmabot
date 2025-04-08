# Gemma Chat Persona Builder

![Gemma Chat Persona Builder Banner](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gemma-header.png)

## Overview

Gemma Chat Persona Builder is an interactive AI chat application that allows users to converse with customizable AI personas. Powered by Google's Gemma 1.1 2B model, this application offers a unique and engaging way to interact with AI through different personality lenses.

## Features

### Multiple Personas
- **Pre-built Personas**: Start chatting immediately with our pre-configured personas:
  - **Zen Monk 🧘**: A peaceful monk who speaks in haikus
  - **Parisian Chef 🧑‍🍳**: A witty French chef with flair
  - **Bollywood Star 💃**: A dramatic Hindi-speaking actor

### Custom Persona Builder
- Create your own unique personas with:
  - Custom names (with emoji support)
  - Personality descriptions
  - Language preferences
  - Voice gender selection

### Image Interaction
- Upload images to get personalized feedback
- AI responds as if looking at you directly
- Perfect for fashion advice, selfie feedback, or just for fun

### Voice Responses
- Text responses are converted to speech
- Voice matches the persona's language and gender
- Supports multiple languages including English, French, Hindi, Spanish, German, and Japanese

### Multilingual Support
- Automatic language detection
- Responses in the persona's preferred language
- Support for multiple global languages

## Deployment Options

### Hugging Face Spaces
This application is optimized for deployment on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Upload the `app.py` and `requirements.txt` files
3. Add your Hugging Face token as an environment variable named `HF_TOKEN`
4. Trigger a "Factory Rebuild"

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gemma-chat-persona-builder.git
cd gemma-chat-persona-builder

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Technical Details

### Models Used
- **Text Generation**: Google's Gemma 1.1 2B Instruct model
- **Image Captioning**: Salesforce BLIP image captioning model
- **Text-to-Speech**: Microsoft Edge TTS

### Requirements
- Python 3.8+
- Gradio 4.0+
- PyTorch 2.0+
- Transformers 4.35+
- Edge-TTS 6.1.9+
- Other dependencies listed in requirements.txt

## Usage Guide

### Creating a Custom Persona
1. Click on "Create Custom Persona" to expand the section
2. Enter a name for your persona (e.g., "Pirate Captain ☠️")
3. Select the primary language
4. Choose the voice gender
5. Write a description of how the persona should behave
6. Click "Create/Update Persona"

### Chatting with Images
1. Select a persona from the dropdown
2. (Optional) Upload an image using the image upload area
3. Type your message and press Enter
4. To start a new conversation about a different topic, clear the image

### Voice Responses
- Voice responses play automatically
- Each persona has a unique voice based on their language and gender settings

## Privacy & Security
- All processing happens within your Hugging Face Space
- No user data is stored or shared
- Image processing is done locally

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

