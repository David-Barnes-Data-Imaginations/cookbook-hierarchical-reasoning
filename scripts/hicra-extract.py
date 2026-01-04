import os
import time
import json
import requests
import numpy as np
import re
from collections import defaultdict
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# Load environment variables
load_dotenv()


# --- CONFIGURATION ---
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "gemini").lower()
print(f"⚙️  Provider set to: {DATA_PROVIDER}")

# --- PROVIDER SETUP ---
generate_text_fn = None

if DATA_PROVIDER == "anthropic":
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("🔧 Using Anthropic (Claude) API")

    def generate_text_anthropic(prompt):
        try:
            message = client.messages.create(
                model="claude-opus-4-5", 
                max_tokens=2048,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )
            return message.content[0].text
        except Exception as e:
            print(f"Anthropic Error: {e}")
            return ""
    generate_text_fn = generate_text_anthropic

elif DATA_PROVIDER == "huggingface":
    # Hugging Face Router API Setup
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
                    "max_tokens": 2048,
                    "temperature": 0.5,
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
    print(f"🔧 Using Ollama Local API (Model: {os.getenv('OLLAMA_MODEL', 'embeddinggemma:300m')})")
    
    def generate_text_ollama(prompt):
        try:
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": os.getenv("OLLAMA_MODEL", "embeddinggemma:300m"),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # "options": {"temperature": 0.7} # Optional
            }
            response = requests.post(url, json=payload, timeout=120) # Longer timeout for local CPU execution
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as e:
            print(f"Ollama Error: {e}")
            return ""
    generate_text_fn = generate_text_ollama

else:
    # Default to Gemini (Vertex AI)
    from google.oauth2 import service_account
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    # Vertex AI Setup
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
                # Vertex AI SDK doesn't always strictly respect client timeouts easily, 
                # but we can try passing it if the version supports it, else we rely on global default.
                # Adding a print to track it.
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini Error: {e}")
                return ""
        generate_text_fn = generate_text_gemini
    except Exception as e:
        print(f"⚠️ Gemini Setup Failed: {e}")
        # Fallback dummy function to prevent crash if env vars missing but user selected gemini
        generate_text_fn = lambda p: "" 

# --- LOADING & OFFSET LOGIC ---
OFFSET_FILE = "hicra_offset.txt"
start_offset = 0

if os.path.exists(OFFSET_FILE):
    with open(OFFSET_FILE, "r") as f:
        try:
            start_offset = int(f.read().strip())
            print(f"🔄 Resuming from offset: {start_offset}")
        except ValueError:
            print("⚠️ Invalid offset file, starting from 0")

# Loading the "Thinking" Dataset
print("📚 Loading Nemotron Dataset...")
# We stream it to avoid downloading the whole massive file immediately
dataset = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v1", split="math", streaming=True)
dataset = dataset.skip(start_offset) # Skip already processed items

# --- STEP 1: SMART EXTRACTION (The Gemini "Miner") ---
# Instead of brute-force n-grams, we ask Gemini to find "Structural Phrases"

def extract_candidates(thought_trace):
    """
    Asks the LLM to act as a Linguist and extract linking phrases.
    """
    prompt = f"""
    Analyze the following reasoning trace. Identify the "Structural Scaffolding" phrases—the abstract connecting words that organize the logic.
    
    Target Criteria (Strategic Grams):
    1.  **Length:** 3 to 5 words long.
    2.  **Function:** Markers of planning, decomposition, verification, or conclusion.
    3.  **Content-Agnostic:** MUST NOT contain specific numbers or nouns from the problem (e.g., "5 apples").
    
    Examples of Good Grams:
    - Beginning a thought: "let's analyze", "first we need", "to solve this", "let's assume",
    - Logic Connectors (The most important ones): "implies that", "consequently", "therefore", "thus", "because", "since", "given that", "conversely", "alternatively",
    - Process Checks (Metacognition): "checking the", "verifying", "double check", "but wait", "identifying", "notice that", "recall that", "we can conclude",
    - Mathematical Actions: "substituting", "calculating", "simplifying", "solving for", "derivative of"
    
    Examples of Bad Grams:
    - "The interest rate is" (Specific content)
    - "and then" (Too short)
    - "so" (Too common)

    Return the list of phrases as a Python list of strings.
    
    Trace:
    {thought_trace[:15000]} 
    """ # We truncate to 15k chars just to be safe with limits

    try:
        print("⏳ Generating...", end="\r")
        text = generate_text_fn(prompt)
        if not text: return []
        
        # Debug: show raw response (first 500 chars)
        print(f"\n📝 Raw response preview: {text[:500]}...")
        
        candidates = []
        
        # Try parsing as Python list first (e.g., ["phrase 1", "phrase 2"])
        try:
            # Find anything that looks like a Python list
            import ast
            list_match = re.search(r'\[.*?\]', text, re.DOTALL)
            if list_match:
                parsed = ast.literal_eval(list_match.group())
                if isinstance(parsed, list):
                    for item in parsed:
                        clean = str(item).strip()
                        if 3 <= len(clean.split()) <= 6:
                            candidates.append(clean.lower())
                    if candidates:
                        print(f"   ✅ Parsed {len(candidates)} grams from list format")
                        return candidates
        except (SyntaxError, ValueError):
            pass  # Fall through to line-by-line parsing
        
        # Fallback: line-by-line parsing for bullet points
        for line in text.split('\n'):
            # Strip common prefixes: bullets, numbers, quotes, brackets
            clean = re.sub(r'^[\s\-\*•\d\.\)\"\[\]]+', '', line).strip().strip('"\',').strip()
            if 3 <= len(clean.split()) <= 6:
                candidates.append(clean.lower())
        
        print(f"   ✅ Parsed {len(candidates)} grams from line format")
        return candidates
    except Exception as e:
        print(f"Extraction Error: {e}")
        return []

# Run Extraction on a sample (e.g., 200 traces gives a good distribution)
raw_grams = []
print("⛏️  Mining Strategic Grams with Gemini...")

count = 0
for example in tqdm(dataset):
    # DEBUG: Show what keys are available in this example
    if count == 0:
        print(f"\n🔍 DEBUG - Example keys: {list(example.keys())}")
    
    # Nemotron dataset has a 'reasoning' column that just contains "on" as a flag,
    # NOT the actual reasoning. The real thinking is in messages[assistant].content
    # wrapped in <think> tags. So we prioritize extracting from messages.
    thought_stream = ""
    
    # Check if 'reasoning' has actual content (not just a short flag like "on")
    raw_reasoning = example.get('reasoning', "")
    if raw_reasoning and len(raw_reasoning) > 20:
        thought_stream = raw_reasoning
        if count == 0:
            print(f"🔍 DEBUG - Using 'reasoning' column (length: {len(thought_stream)})")
    
    # If no valid reasoning column, extract from messages
    if not thought_stream and 'messages' in example:
        msgs = example['messages']
        
        # DEBUG: Show message structure for first example
        if count == 0:
            print(f"🔍 DEBUG - Number of messages: {len(msgs)}")
            for i, msg in enumerate(msgs):
                role = msg.get('role', 'unknown')
                content_preview = str(msg.get('content', ''))[:200]
                print(f"   Message {i} [{role}]: {content_preview}...")
        
        # iterate backwards to find the last assistant message
        for msg in reversed(msgs):
            if msg['role'] == 'assistant':
                content = msg['content']
                
                # DEBUG: Show content info
                if count == 0:
                    print(f"🔍 DEBUG - Assistant content length: {len(content)}")
                    print(f"🔍 DEBUG - Contains '<think>': {'<think>' in content}")
                    print(f"🔍 DEBUG - Contains '</think>': {'</think>' in content}")
                
                # Check for <think> tags - use GREEDY matching (.*) not (.*?)
                # to capture everything between the tags
                match = re.search(r'<think>(.*)</think>', content, re.DOTALL)
                if match:
                    thought_stream = match.group(1).strip()
                    if count == 0:
                        print(f"🔍 DEBUG - Extracted think stream length: {len(thought_stream)}")
                else:
                    # No think tags - maybe the whole content is the reasoning?
                    # Or try alternative tag formats
                    alt_match = re.search(r'<thinking>(.*)</thinking>', content, re.DOTALL)
                    if alt_match:
                        thought_stream = alt_match.group(1).strip()
                    else:
                        # Use the full content as fallback
                        thought_stream = content
                    if count == 0:
                        print(f"🔍 DEBUG - No <think> tags found, using full content")
                break
    
    # DEBUG: Show what we're actually sending to the API
    if count == 0:
        print(f"🔍 DEBUG - Final thought_stream length: {len(thought_stream) if thought_stream else 0}")
        print(f"🔍 DEBUG - thought_stream preview: {thought_stream[:300] if thought_stream else 'EMPTY'}...")

    if not thought_stream: continue
        
    grams = extract_candidates(thought_stream)
    raw_grams.extend(grams)
    
    count += 1
    print(f"✅ Processed: {count}") # Explicit counter as requested
    
    # Only sleep if we are hitting an external API to be polite
    if DATA_PROVIDER != "ollama":
        time.sleep(30) 
        
    if count >= 5: # Stop after 200 samples to save API credits
        break

# Save new offset
new_offset = start_offset + count
with open(OFFSET_FILE, "w") as f:
    f.write(str(new_offset))
print(f"💾 Updated offset to {new_offset}")

print(f"✅ Extracted {len(raw_grams)} raw candidates.")

# --- STEP 2: SEMANTIC CLUSTERING (The Paper's Method) ---
# Now we use the logic from "The Architecture of Strategic Gram Acquisition"

print("🧠 Vectorizing Grams...")
print("🧠 Vectorizing Grams (using Ollama: embeddinggemma:300m)...")

def get_ollama_embedding(text):
    try:
        url = "http://localhost:11434/api/embeddings"
        payload = {
            "model": "embeddinggemma:300m",
            "prompt": text
        }
        response = requests.post(url, json=payload, timeout=30)
        return response.json()["embedding"]
    except Exception as e:
        print(f"Embedding Error for '{text}': {e}")
        return []

unique_grams = list(set(raw_grams))
embeddings = []
print(f"    - Embedding {len(unique_grams)} unique phrases...")
for gram in tqdm(unique_grams):
    emb = get_ollama_embedding(gram)
    if emb:
        embeddings.append(emb)
    else:
        # Handle failure (maybe append zero vector or skip)
        # For simplicity in clustering, we might need to remove this gram from unique_grams to align indices, 
        # but let's assume reliability for now or filter.
        embeddings.append([0.0]*768) # Placeholder size, ideally we'd know dimensions

embeddings = np.array(embeddings)

print("Isolating Clusters...")

# Guard: Skip clustering if we have no embeddings
if len(embeddings) == 0 or embeddings.size == 0:
    print("⚠️  No grams to cluster! Check that the LLM is returning valid phrases.")
    print("   This could mean:")
    print("   1. The response format isn't being parsed correctly")
    print("   2. The thought traces don't contain enough structural phrases")
    print("   3. The API is returning errors")
    exit(1)

if len(unique_grams) < 2:
    print(f"⚠️  Only {len(unique_grams)} gram(s) found - need at least 2 for clustering.")
    print("   Try processing more samples by increasing the count limit.")
    exit(1)

# Agglomerative Clustering (as per HICRA paper usually) or DBSCAN
# Distance threshold determines how "strict" synonyms must be
clustering_model = AgglomerativeClustering(
    n_clusters=None, 
    distance_threshold=0.5, # Tune this: Lower = stricter synonyms
    metric='cosine', 
    linkage='average'
)
cluster_labels = clustering_model.fit_predict(embeddings)

# Group grams by cluster
clusters = defaultdict(list)
for gram, label in zip(unique_grams, cluster_labels):
    clusters[label].append(gram)

# --- STEP 3: FREQUENCY FILTERING (The "Reusability" Test) ---
# Calculate "Cluster Document Frequency" (How many *traces* did this cluster appear in?)

cluster_scores = defaultdict(int)

# Re-scan our raw extraction results to count frequency
# (In a full production run, we'd map back to documents, but simple count works for this proxy)
gram_to_cluster = {gram: label for gram, label in zip(unique_grams, cluster_labels)}

for gram in raw_grams:
    if gram in gram_to_cluster:
        cluster_id = gram_to_cluster[gram]
        cluster_scores[cluster_id] += 1

# Select Top 20% of Clusters (The Paper's Threshold)
sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)

# Debug: Show clustering stats
print(f"📊 Clustering stats:")
print(f"   - Total clusters: {len(clusters)}")
print(f"   - Clusters with scores: {len(sorted_clusters)}")
if sorted_clusters:
    print(f"   - Top 5 cluster scores: {[s for _, s in sorted_clusters[:5]]}")

# Ensure at least 1 cluster is selected (avoid rounding to 0)
top_n = max(1, int(len(sorted_clusters) * 0.20))
top_clusters = sorted_clusters[:top_n]

# --- FINAL OUTPUT ---
final_strategic_grams = []
print(f"\n🏆 Top {top_n} Strategic Clusters Identified:\n")

for cluster_id, score in top_clusters:
    # Pick the "Canonical" gram (usually the shortest or most frequent in the cluster)
    members = clusters[cluster_id]
    canonical = min(members, key=len) # Pick shortest as representative
    final_strategic_grams.append(canonical)
    print(f"Cluster {cluster_id} (Score {score}): {canonical} | Variations: {members[:3]}")

# Save to file for your training loop (Append mode)
print("💾 Appending to strategic_grams_nemotron.txt...")
with open("strategic_grams_nemotron.txt", "a") as f:
    f.write("\n" + str(final_strategic_grams))

print("\n✅ Done! Appended new grams to 'strategic_grams_nemotron.txt'.")