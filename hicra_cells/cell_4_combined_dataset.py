# Cell 4: Load and Combine Datasets
from datasets import load_dataset, Dataset
import json

# === Configuration ===
MAX_PROMPT_TOKENS = 400    # Filter out prompts longer than this
MAX_ANSWER_TOKENS = 600    # Filter out answers longer than this  
NEMOTRON_SAMPLE_SIZE = 3000  # How many Nemotron examples to use

# System prompt for reasoning format
SYSTEM_PROMPT = """
You are a mathematical reasoning assistant. Think through problems step by step.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def format_prompt(example):
    """Format dataset for GRPO training with chat template."""
    return {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT.strip()},
            {'role': 'user', 'content': example['prompt']}
        ],
        'answer': str(example['answer'])
    }

def format_nemotron(example):
    """Convert Nemotron format to our format."""
    messages = example.get('messages', [])
    
    # Extract user prompt and assistant answer
    user_content = ""
    assistant_content = ""
    
    for msg in messages:
        if msg['role'] == 'user':
            user_content = msg['content']
        elif msg['role'] == 'assistant':
            assistant_content = msg['content']
    
    # Get expected answer (fallback to assistant content if not available)
    expected = example.get('expected_answer', '')
    if not expected:
        # Try to extract from assistant's <answer> tags if present
        if '<answer>' in assistant_content and '</answer>' in assistant_content:
            expected = assistant_content.split('<answer>')[-1].split('</answer>')[0].strip()
        else:
            expected = assistant_content[-200:] if len(assistant_content) > 200 else assistant_content
    
    return {
        'prompt': user_content,
        'answer': str(expected)
    }

def estimate_tokens(text):
    """Rough token estimate (1 token ≈ 4 chars for English)."""
    return len(str(text)) // 4

def filter_by_length(example):
    """Filter out examples that are too long."""
    prompt_tokens = estimate_tokens(example['prompt'])
    answer_tokens = estimate_tokens(example['answer'])
    return prompt_tokens <= MAX_PROMPT_TOKENS and answer_tokens <= MAX_ANSWER_TOKENS

# === 1. Load Your HICRA Synthetic Data ===
print("📂 Loading HICRA dataset...")
my_dataset = load_dataset(
    "json", 
    data_files="reasoning_dataset_v2_train.json", 
    split="train"
)
print(f"   ✅ Loaded {len(my_dataset)} HICRA examples")

# === 2. Load Nemotron Math Data (Streaming) ===
print(f"🌊 Streaming {NEMOTRON_SAMPLE_SIZE} Nemotron math examples...")
try:
    nemotron_stream = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v1", 
        split="math", 
        streaming=True
    )
    
    # Take a sample and convert to list
    nemotron_list = []
    for i, example in enumerate(nemotron_stream):
        if i >= NEMOTRON_SAMPLE_SIZE:
            break
        formatted = format_nemotron(example)
        # Only keep if it's not too long
        if filter_by_length(formatted):
            nemotron_list.append(formatted)
        
        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1} examples, kept {len(nemotron_list)}...")
    
    nemotron_dataset = Dataset.from_list(nemotron_list)
    print(f"   ✅ Loaded {len(nemotron_dataset)} Nemotron examples (after length filter)")
    
except Exception as e:
    print(f"   ⚠️ Could not load Nemotron: {e}")
    print("   Continuing with HICRA data only...")
    nemotron_dataset = None

# === 3. Combine Datasets ===
print("🔀 Combining datasets...")

# Filter HICRA by length too
my_dataset_filtered = my_dataset.filter(filter_by_length)
print(f"   HICRA after filter: {len(my_dataset_filtered)} examples")

if nemotron_dataset and len(nemotron_dataset) > 0:
    from datasets import concatenate_datasets
    
    # Make sure both have the same columns
    combined_dataset = concatenate_datasets([my_dataset_filtered, nemotron_dataset])
    print(f"   ✅ Combined dataset: {len(combined_dataset)} examples")
else:
    combined_dataset = my_dataset_filtered
    print(f"   ✅ Using HICRA only: {len(combined_dataset)} examples")

# === 4. Format for GRPO Training ===
print("📝 Formatting for GRPO...")
dataset_train = combined_dataset.map(format_prompt)

# Shuffle to mix the datasets
dataset_train = dataset_train.shuffle(seed=42)

# === 5. Load Test Set (HICRA only) ===
dataset_test = load_dataset(
    "json", 
    data_files="reasoning_dataset_v2_test.json", 
    split="train"
).map(format_prompt)

print(f"\n✅ Final Training Set: {len(dataset_train)} examples")
print(f"✅ Test Set: {len(dataset_test)} examples")
print(f"\nSample prompt format:")
print(dataset_train[0]['prompt'])
