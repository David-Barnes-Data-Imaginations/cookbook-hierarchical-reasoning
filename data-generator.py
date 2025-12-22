import json
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "gemini").lower()  # "gemini" or "anthropic"
OUTPUT_FILE = "reasoning_dataset.json"
NUM_SAMPLES = 100  # 10 requests * 5 samples/request

# The prompt forces "Question" and "Answer" only.
GENERATION_PROMPT = """
You are a dataset generator. Generate 5 unique, high-school level math or logic word problems.
Format the output strictly as a JSON list of objects.
Each object must have:
- "prompt": The word problem.
- "answer": The final numeric or short answer ONLY. Do NOT include the steps.

Example:
[
  {"prompt": "If a train travels 60mph for 2 hours, how far does it go?", "answer": "120 miles"},
  {"prompt": "Solve for x: 2x + 5 = 15", "answer": "x = 5"}
]

"""

# --- PROVIDER SETUP ---
if DATA_PROVIDER == "anthropic":
    import anthropic
    
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    print("🔧 Using Anthropic (Claude) API")
    
    def generate_batch():
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": GENERATION_PROMPT}
                ]
            )
            # Extract text from the response
            text = message.content[0].text
            # Clean up code blocks if Claude adds them
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            print(f"Error generating batch: {e}")
            time.sleep(1)
            return []

else:
    # Default to Gemini via Vertex AI
    from google.oauth2 import service_account
    import vertexai
    from vertexai.generative_models import GenerativeModel
    
    SERVICE_ACCOUNT_FILE = "persona-forge-470514-c46d9ea81277.json"
    PROJECT_ID = "persona-forge-470514"
    LOCATION = "us-central1"
    
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    
    model = GenerativeModel("gemini-2.5-flash")
    print("🔧 Using Gemini (Vertex AI) API")
    
    def generate_batch():
        try:
            response = model.generate_content(GENERATION_PROMPT)
            # Clean up code blocks if Gemini adds them
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            print(f"Error generating batch: {e}")
            time.sleep(1)
            return []

all_data = []
provider_name = "Anthropic (Claude)" if DATA_PROVIDER == "anthropic" else "Gemini"
print(f"🚀 Generating {NUM_SAMPLES} samples using {provider_name}...")

pbar = tqdm(total=NUM_SAMPLES)
while len(all_data) < NUM_SAMPLES:
    batch = generate_batch()
    if batch:
        all_data.extend(batch)
        pbar.update(len(batch))
        # Save incrementally just in case
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_data, f, indent=2)
    time.sleep(1) # Be nice to the API rate limit

pbar.close()
print(f"✅ Done! Saved {len(all_data)} items to {OUTPUT_FILE}")