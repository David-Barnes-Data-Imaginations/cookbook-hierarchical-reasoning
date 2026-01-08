"""
Strategic Grams Filter (HICRA)
==============================
This script filters the extracted strategic grams to keep only "Domain-Agnostic Scaffolding" phrases.
It uses the same Gemini (Vertex AI) API pattern from hicra-extract.py.

Logic:
- KEEP phrases that represent cognitive actions (planning, doubting, verifying, branching)
- REJECT domain-specific terms (math jargon, code jargon, meaningless fragments)
"""

import os
import time
import json
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
INPUT_FILE = "strategic_grams_list.json"  # Your ~9k grams
OUTPUT_FILE = "strategic_grams_filtered.json"  # Gold standard output
BATCH_SIZE = 50  # How many grams to send per API call
API_DELAY = 1.0  # Seconds between API calls (rate limit politeness)

# --- PROVIDER SETUP (from hicra-extract.py) ---
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "gemini").lower()
print(f"⚙️  Provider set to: {DATA_PROVIDER}")

generate_text_fn = None

if DATA_PROVIDER == "anthropic":
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("🔧 Using Anthropic (Claude) API")

    def generate_text_anthropic(prompt):
        try:
            message = client.messages.create(
                model="claude-opus-4-5", 
                max_tokens=4096,
                temperature=0.0,  # Deterministic for filtering
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )
            return message.content[0].text
        except Exception as e:
            print(f"Anthropic Error: {e}")
            return ""
    generate_text_fn = generate_text_anthropic

elif DATA_PROVIDER == "huggingface":
    import requests
    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
    HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    print("🔧 Using Hugging Face Router API")

    def generate_text_hf(prompt):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                timeout=60,
                json={
                    "model": "openai/gpt-oss-120b:cerebras", 
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.0,
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"HF Error: {e}")
            return ""
    generate_text_fn = generate_text_hf

elif DATA_PROVIDER == "ollama":
    import requests
    print(f"🔧 Using Ollama Local API (Model: {os.getenv('OLLAMA_MODEL', 'gemma2:9b')})")
    
    def generate_text_ollama(prompt):
        try:
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": os.getenv("OLLAMA_MODEL", "gemma2:9b"),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as e:
            print(f"Ollama Error: {e}")
            return ""
    generate_text_fn = generate_text_ollama

else:
    # Default to Gemini (Vertex AI) - same as hicra-extract.py
    from google.oauth2 import service_account
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    SERVICE_ACCOUNT_FILE = os.getenv("GCP_SERVICE_ACCOUNT_FILE", "persona-forge-470514-c46d9ea81277.json")
    PROJECT_ID = "persona-forge-470514" 
    LOCATION = "global"

    try:
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        model = GenerativeModel("gemini-3-flash-preview")
        print("🔧 Using Gemini (Vertex AI) API")

        def generate_text_gemini(prompt):
            try:
                def _call_api():
                    return model.generate_content(prompt)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_call_api)
                    try:
                        response = future.result(timeout=120)
                        return response.text
                    except concurrent.futures.TimeoutError:
                        print(f"\n⚠️ Gemini API timed out after 120s, skipping this batch...")
                        return ""
            except Exception as e:
                print(f"Gemini Error: {e}")
                return ""
        generate_text_fn = generate_text_gemini
    except Exception as e:
        print(f"⚠️ Gemini Setup Failed: {e}")
        generate_text_fn = lambda p: ""


# --- THE FILTERING FUNCTION ---
def filter_grams_batch(batch_of_grams):
    """
    Sends a batch of grams to the LLM and asks it to filter for domain-agnostic scaffolding.
    Returns only the accepted grams.
    """
    prompt = f"""
You are an expert Linguist and Logic Engineer.
I have a list of n-grams extracted from an AI's "Chain of Thought" reasoning.

Your Goal: Identify the **"Strategic Scaffolding"** phrases.
These are phrases that represent *cognitive actions* (planning, doubting, verifying, branching).

Rules for Acceptance:
1. MUST be Domain-Agnostic: Works for Math, Coding, History, Philosophy, or any reasoning task.
2. MUST be Metacognitive: Describes *how* the model is thinking, not *what* it is calculating.
3. PREFER phrases that signal:
   - Critical Doubt: "but is that necessarily", "might be wrong", "let me verify"
   - Verification: "let's verify with what", "let me check", "double-check"
   - Hypothesis Setting: "assume variables are such", "let's suppose", "if we consider"
   - Knowledge Retrieval: "i recall that", "this reminds me of"
   - Logical Branching: "alternatively", "on the other hand", "what if instead"
   - Planning: "first we need to", "the approach is", "let me break this down"

Rules for Rejection:
1. REJECT specific Math terms: "least common multiple", "integrate the function", "segments are equal", "derivative of", "discriminant", "quadratic equation", "roots of", "sum of digits"
2. REJECT specific Code terms: "python function", "return the string", "variable is"
3. REJECT meaningless fragments: "and then the", "is equal to", "so we have", "but then"
4. REJECT phrases that are too content-specific: "determined by the number", "is always negative for", "the two lines"
5. REJECT geometry/equation jargon: "parallel to the", "on the boundary", "the denominator is"

Examples of GOOD grams (KEEP):
- "but is that necessarily" (Critical Doubt)
- "let's verify with what" (Verification)
- "is there a way to" (Planning)
- "i recall that sometimes" (Knowledge Retrieval)
- "but wait" (Pause/Verification)
- "notice that if" (Observation)
- "this suggests that" (Inference)
- "what if instead" (Branching)

Examples of BAD grams (REJECT):
- "determined by the number" (Too specific)
- "the least common multiple of" (Math jargon)
- "segments are equal" (Geometry jargon)
- "is parallel to the" (Geometry jargon)
- "so we have" (Meaningless fragment)
- "product of the roots is" (Math-specific)

Input List:
{json.dumps(batch_of_grams)}

Output:
Return a JSON list containing ONLY the accepted grams. Do not include any explanation or markdown formatting.
Just output the raw JSON array like: ["gram1", "gram2", "gram3"]
"""

    try:
        text = generate_text_fn(prompt)
        if not text:
            return []
        
        # Clean up response
        text = text.strip()
        
        # Remove markdown code block formatting if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Parse JSON
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [g for g in result if isinstance(g, str)]
        except json.JSONDecodeError:
            # Try to find a JSON array in the response
            import re
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    if isinstance(result, list):
                        return [g for g in result if isinstance(g, str)]
                except json.JSONDecodeError:
                    pass
        
        print(f"⚠️ Could not parse response: {text[:200]}...")
        return []
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []


# --- MAIN EXECUTION ---
def main():
    # 1. Load the raw grams
    print(f"📚 Loading grams from '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, "r") as f:
            raw_grams = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{INPUT_FILE}'")
        print("   Make sure you're running this from the project root directory.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in '{INPUT_FILE}': {e}")
        return
    
    print(f"   Found {len(raw_grams)} grams to filter.")
    
    # 2. Process in batches
    filtered_grams = []
    total_batches = (len(raw_grams) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n🔍 Filtering grams in batches of {BATCH_SIZE}...")
    print(f"   Total batches: {total_batches}")
    print(f"   Estimated time: ~{total_batches * API_DELAY / 60:.1f} minutes\n")
    
    for i in tqdm(range(0, len(raw_grams), BATCH_SIZE), desc="Filtering"):
        batch = raw_grams[i:i + BATCH_SIZE]
        accepted = filter_grams_batch(batch)
        filtered_grams.extend(accepted)
        
        # Progress update every 10 batches
        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"   Progress: {len(filtered_grams)} kept so far...")
        
        # Rate limiting - be polite to the API
        if DATA_PROVIDER != "ollama":
            time.sleep(API_DELAY)
    
    # 3. Remove any duplicates
    filtered_grams = list(set(filtered_grams))
    filtered_grams.sort()  # Sort alphabetically for nice output
    
    # 4. Save the results
    print(f"\n💾 Saving {len(filtered_grams)} filtered grams to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(filtered_grams, f, indent=2)
    
    # 5. Summary
    print(f"\n{'='*50}")
    print(f"🏆 FILTERING COMPLETE")
    print(f"{'='*50}")
    print(f"   Input:  {len(raw_grams)} raw grams")
    print(f"   Output: {len(filtered_grams)} filtered grams")
    print(f"   Reduction: {100 * (1 - len(filtered_grams)/len(raw_grams)):.1f}%")
    print(f"\n   Saved to: {OUTPUT_FILE}")
    
    # Show some examples
    print(f"\n📝 Sample filtered grams:")
    for gram in filtered_grams[:10]:
        print(f"   • {gram}")
    print("   ...")


if __name__ == "__main__":
    main()
