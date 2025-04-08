import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from langdetect import detect
from PIL import Image
import torch
import os
import io
import asyncio
import edge_tts
import tempfile
from huggingface_hub import login

# ------------------ Authentication ------------------ #
# Login with your Hugging Face token
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
elif "HUGGING_FACE_HUB_TOKEN" in os.environ:
    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])

# Print Gradio version for debugging
import gradio as gr
print(f"Gradio version: {gr.__version__}")

# ------------------ Load Gemma (text generation) ------------------ #
model_id = "google/gemma-1.1-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# ------------------ Load BLIP (image captioning) ------------------ #
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", 
    torch_dtype=torch.float32
).to("cpu")
print("BLIP model loaded successfully")

# ------------------ Voice Configuration ------------------ #
# Define voice options by gender and language
VOICE_OPTIONS = {
    "English": {
        "Male": "en-US-GuyNeural",
        "Female": "en-US-JennyNeural",
        "Neutral": "en-US-AriaNeural"
    },
    "French": {
        "Male": "fr-FR-HenriNeural",
        "Female": "fr-FR-DeniseNeural",
        "Neutral": "fr-FR-AlainNeural"
    },
    "Hindi": {
        "Male": "hi-IN-MadhurNeural",
        "Female": "hi-IN-SwaraNeural",
        "Neutral": "hi-IN-MadhurNeural"
    },
    "Spanish": {
        "Male": "es-ES-AlvaroNeural",
        "Female": "es-ES-ElviraNeural",
        "Neutral": "es-ES-AlvaroNeural"
    },
    "German": {
        "Male": "de-DE-ConradNeural",
        "Female": "de-DE-KatjaNeural",
        "Neutral": "de-DE-KatjaNeural"
    },
    "Japanese": {
        "Male": "ja-JP-KeitaNeural",
        "Female": "ja-JP-NanamiNeural",
        "Neutral": "ja-JP-NanamiNeural"
    },
    "Other": {
        "Male": "en-US-GuyNeural",
        "Female": "en-US-JennyNeural",
        "Neutral": "en-US-AriaNeural"
    }
}

# ------------------ Text-to-Speech Function ------------------ #
async def text_to_speech(text, voice):
    try:
        print(f"Converting text to speech using voice: {voice}")
        communicate = edge_tts.Communicate(text, voice)
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_path = temp_file.name
        
        # Generate speech and save to the temporary file
        await communicate.save(temp_path)
        print(f"Speech saved to {temp_path}")
        
        return temp_path
    except Exception as e:
        print(f"Error in text_to_speech: {str(e)}")
        return None

# Helper function to run async functions in sync context
def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If there's already an event loop running
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# ------------------ Define Personas ------------------ #
personas = {
    "Zen Monk 🧘": {
        "prompt_prefix": "You are a peaceful Zen monk who speaks in haikus. Respond calmly.",
        "language": "English",
        "gender": "Male"
    },
    "Parisian Chef 🧑‍🍳": {
        "prompt_prefix": "You are a witty French chef who comments on food and style with flair.",
        "language": "French",
        "gender": "Male"
    },
    "Bollywood Star 💃": {
        "prompt_prefix": "You are a dramatic Bollywood actor. Respond in Hindi with flair, as if you're in a movie scene.",
        "language": "Hindi",
        "gender": "Female"
    }
}

# ------------------ Helper: Describe Image ------------------ #
def describe_image(image_path):
    try:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        
        # Add image size check
        width, height = image.size
        print(f"Image dimensions: {width}x{height}")
        if width < 10 or height < 10:  # Arbitrary small size check
            return "The image appears to be too small or empty."
            
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_length=50)  # Increase max length
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        print(f"Generated caption: {caption}")
        
        # If caption is empty or too generic, try a different approach
        if not caption or caption.lower().strip() in ["", "a", "an", "the"]:
            return "I see an image, but I'm having trouble describing it in detail."
            
        return caption
    except Exception as e:
        print(f"Error in describe_image: {str(e)}")
        return f"an image was uploaded, but I couldn't process it properly. Error: {str(e)}"

# ------------------ Custom Persona Creation ------------------ #
def update_custom_persona(name, description, language, gender):
    global personas
    if name and description:
        # Make a deep copy of the personas dictionary
        updated_personas = dict(personas)
        # Add the new persona
        updated_personas[name] = {
            "prompt_prefix": description,
            "language": language,
            "gender": gender
        }
        # Update the global personas dictionary
        personas = updated_personas
        print(f"Added persona: {name} with language: {language} and gender: {gender}")
        print(f"Current personas: {list(personas.keys())}")
        # Return the updated dropdown
        return gr.update(choices=list(personas.keys()), value=name)
    return gr.update(choices=list(personas.keys()))

# ------------------ Chat Function ------------------ #
def chat(user_input, chat_history, persona_choice, image=None):
    if persona_choice not in personas:
        print(f"Warning: Selected persona '{persona_choice}' not found in personas dictionary")
        # Fallback to first persona if selected one doesn't exist
        persona_choice = list(personas.keys())[0]
        
    persona = personas[persona_choice]
    prefix = persona["prompt_prefix"]
    
    # Get persona language and gender for voice
    language = persona.get("language", "English")
    gender = persona.get("gender", "Neutral")

    # Detect input language
    try:
        lang = detect(user_input)
        prefix += f"\n(User message detected in: {lang})"
    except:
        pass

    # Handle image input
    if image:
        print(f"Image received: {image}")
        caption = describe_image(image)
        
        # Using exactly the user's preferred prompt style
        prefix += (
            f"\nYou are chatting with someone who just uploaded a photo described as: '{caption}'. "
            f"Respond **directly to them** as their selected persona. "
            f"Speak warmly, personally, and as if you're looking at them. "
            f"Give your opinion, compliments, or thoughts on their appearance. "
            f"Do not narrate or describe the image — speak to them as a person."
        )
    
    # Build conversation prompt
    history_formatted = ""
    for u, r in chat_history[-3:]:  # Only include last 3 turns
        history_formatted += f"<|user|>\n{u}\n<|assistant|>\n{r}\n"
    prompt = f"{prefix}\n{history_formatted}<|user|>\n{user_input}\n<|assistant|>\n"

    result = pipe(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        top_p=0.85,
        temperature=0.6,
        eos_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]

    # Extract response
    if "<|assistant|>" in result:
        reply = result.split("<|assistant|>")[-1].strip()
    else:
        reply = result.strip()

    # Generate speech for the reply
    try:
        # Select voice based on persona language and gender
        voice = VOICE_OPTIONS.get(language, VOICE_OPTIONS["English"]).get(gender, VOICE_OPTIONS["English"]["Neutral"])
        audio_path = run_async(text_to_speech(reply, voice))
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        audio_path = None

    chat_history.append((user_input, reply))
    return chat_history, chat_history, audio_path

# ------------------ Build Gradio UI ------------------ #
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Gemma Chat Persona Builder")
    gr.Markdown("Choose a persona, ask something, and (optionally) upload an image for feedback.")

    with gr.Accordion("Create Custom Persona", open=False):
        with gr.Row():
            persona_name = gr.Textbox(label="Persona Name (e.g., 'Pirate Captain ☠️')")
            persona_language = gr.Dropdown(
                list(VOICE_OPTIONS.keys()), 
                label="Language", 
                value="English"
            )
            persona_gender = gr.Dropdown(
                ["Male", "Female", "Neutral"],
                label="Voice Gender",
                value="Neutral"
            )
        persona_description = gr.Textbox(
            label="Persona Description", 
            placeholder="Describe how this persona should behave (e.g., 'You are a fearsome pirate captain who speaks with nautical terms and is always looking for treasure.')",
            lines=3
        )
        create_btn = gr.Button("Create/Update Persona")

    persona_choice = gr.Dropdown(choices=list(personas.keys()), label="Choose a Persona", value="Zen Monk 🧘")
    image_input = gr.Image(
        type="filepath", 
        label="(Optional) Upload an Image",
        height=300,  # Set a fixed height
        width=300    # Set a fixed width
    )
    
    # Add user instructions for image removal
    gr.Markdown("""
    **Note:** Upload an image to get feedback on your appearance. 
    To start a new conversation about a different topic, click the 'Clear' button next to the image upload area.
    """)

    chatbot = gr.Chatbot(label="Chat with your AI Persona")
    audio_output = gr.Audio(label="Voice Response", autoplay=True)
    msg = gr.Textbox(label="Type your message", placeholder="Say hello... or ask for advice.")
    state = gr.State([])

    # Connect components
    create_btn.click(
        fn=update_custom_persona, 
        inputs=[persona_name, persona_description, persona_language, persona_gender], 
        outputs=[persona_choice]
    )
    
    msg.submit(chat, [msg, state, persona_choice, image_input], [chatbot, state, audio_output])

    gr.Examples(
        examples=[
            ["Kaise ho?", "Bollywood Star 💃"],
            ["What is the meaning of peace?", "Zen Monk 🧘"],
            ["Do you like my dish?", "Parisian Chef 🧑‍🍳"]
        ],
        inputs=[msg, persona_choice]
    )

# ------------------ Launch App ------------------ #
if __name__ == "__main__":
    print("Models loaded:")
    print("BLIP model loaded:", blip_model is not None)
    print("Gemma model loaded:", model is not None)
    demo.launch(share=False)
