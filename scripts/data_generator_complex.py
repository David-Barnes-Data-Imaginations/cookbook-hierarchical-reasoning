import json
import time
import os
import random
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "gemini").lower()
OUTPUT_FILE = "reasoning_dataset_v2.json"
NUM_SAMPLES = 100 

# --- DYNAMIC PROMPT ENGINE ---
# This forces variance by asking for specific, different things every time.
DOMAINS = [
    "Probability & Statistics", "Linear Algebra", "Number Theory", 
    "Logic & Game Theory", "Physics (Kinematics)", "Financial Math",
    "Geometry", "Combinatorics", "Set Theory"
]

CONCEPTS = [
    "optimization", "finding the maximum", "calculating expected value",
    "deducing the truth", "calculating rates", "compound interest",
    "identifying the outlier", "solving for unknown variables", "proof by contradiction"
]

DIFFICULTIES = [
    "challenging high school level", "undergraduate competition level", 
    "Math Olympiad (AIME) level", "complex puzzle"
]

def build_dynamic_prompt():
    domain = random.choice(DOMAINS)
    concept = random.choice(CONCEPTS)
    difficulty = random.choice(DIFFICULTIES)
    
    return f"""
You are a dataset generator. Generate 5 unique {difficulty} word problems.
Focus specifically on the domain of **{domain}** involving **{concept}**.

CRITICAL INSTRUCTION: 
The problems MUST require multi-step reasoning to solve. 
Do not generate simple "plug-and-play" arithmetic.

Format the output strictly as a JSON list of objects.
Each object must have:
- "prompt": The complex word problem.
- "answer": The final numeric or short answer ONLY. Do NOT include the steps.

Example format:
[
  {{"prompt": "...", "answer": "..."}}
]
"""

# --- PROVIDER SETUP ---
if DATA_PROVIDER == "anthropic":
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("🔧 Using Anthropic (Claude) API")
    
    def generate_batch():
        prompt = build_dynamic_prompt() # <--- NEW: Get a unique prompt
        try:
            message = client.messages.create(
                model="claude-opus-4-5",  # Latest Sonnet model, also use these for additional variety: claude-3-5-haiku-20241022, claude-opus-4-20250514, claude-sonnet-4-20250514
                max_tokens=2048,
                temperature=0.8, # <--- NEW: Increased slightly for creativity
                messages=[{"role": "user", "content": prompt}]
            )
            text = message.content[0].text
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            print(f"Error: {e}")
            return []

elif DATA_PROVIDER == "huggingface":
    # Hugging Face Router API Setup (supports inference providers like :novita)
    import requests
    
    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
    HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    print("🔧 Using Hugging Face Router (DeepSeek-V3.2 via Novita) API")
    
    def generate_batch():
        prompt = build_dynamic_prompt()
        try:
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json={
                    "model": "openai/gpt-oss-120b:cerebras", # deepseek-ai/DeepSeek-V3.2:novita, MiniMaxAI/MiniMax-M2:novita
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.2,
                }
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            print(f"Error: {e}")
            return []

else:
    # Gemini Setup
    import google.generativeai as genai
    
    # Simpler setup if using API key (or stick to Vertex if you prefer)
    # genai.configure(api_key=os.environ["GEMINI_API_KEY"]) 
    
    # Assuming Vertex setup from your file:
    from google.oauth2 import service_account
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    SERVICE_ACCOUNT_FILE = "persona-forge-470514-c46d9ea81277.json"
    PROJECT_ID = "persona-forge-470514"
    LOCATION = "global" #"us-central1"
    
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    
    model = GenerativeModel("gemini-3-pro-preview") #  also use: gemini-2.5-pro, gemini-2.5-flash, gemini-3-flash-preview, gemini-3-pro-preview
    print("🔧 Using Gemini (Vertex AI) API")
    
    def generate_batch():
        prompt = build_dynamic_prompt() # <--- NEW: Get a unique prompt
        
        config = GenerationConfig(
            temperature=0.8, # <--- NEW: Higher temp for variety
            response_mime_type="application/json" # Gemini specific JSON mode
        )
        
        try:
            response = model.generate_content(prompt, generation_config=config)
            return json.loads(response.text)
        except Exception as e:
            print(f"Error: {e}")
            return []

# --- MAIN LOOP ---
all_data = []
print(f"🚀 Generating {NUM_SAMPLES} samples...")

pbar = tqdm(total=NUM_SAMPLES)
while len(all_data) < NUM_SAMPLES:
    batch = generate_batch()
    if batch:
        all_data.extend(batch)
        pbar.update(len(batch))
        
        # Save incrementally
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_data, f, indent=2)
            
    time.sleep(1)

pbar.close()
print(f"✅ Done! Saved to {OUTPUT_FILE}")