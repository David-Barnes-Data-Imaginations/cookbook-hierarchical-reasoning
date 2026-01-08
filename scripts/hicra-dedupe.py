"""
Strategic Grams Deduplicator (HICRA)
====================================
This script deduplicates similar strategic grams by:
1. Embedding each gram using Ollama's embeddinggemma:300m
2. Clustering similar grams using Agglomerative Clustering
3. Keeping the shortest gram from each cluster (as long as it's meaningful)

The idea is that "but in general" and "but in general, for" are semantically
similar, so we keep the shorter one as the canonical representative.
"""

import os
import json
import numpy as np
import requests
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

# --- CONFIGURATION ---
INPUT_FILE = "strategic_grams_filtered.json"  # Your ~4k filtered grams
OUTPUT_FILE = "strategic_grams_deduplicated.json"  # Final deduplicated output
MIN_WORDS = 3  # Minimum word count to keep (avoid too-short phrases)
DISTANCE_THRESHOLD = 0.15  # Cosine distance threshold for clustering (lower = stricter)

# --- EMBEDDING FUNCTION (from hicra-extract.py) ---
def get_ollama_embedding(text):
    """Get embedding from Ollama's embeddinggemma:300m model."""
    try:
        url = "http://localhost:11434/api/embeddings"
        payload = {
            "model": "embeddinggemma:300m",
            "prompt": text
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Embedding Error for '{text}': {e}")
        return None


def select_best_from_cluster(grams):
    """
    Select the best gram from a cluster of similar grams.
    
    Strategy:
    1. Prefer shorter grams (they're more general/reusable)
    2. But ensure minimum word count (MIN_WORDS) to keep meaning
    3. Prefer grams without trailing punctuation like commas
    """
    # Filter out grams that are too short to be meaningful
    valid_grams = [g for g in grams if len(g.split()) >= MIN_WORDS]
    
    if not valid_grams:
        # If all are too short, just take the longest of the short ones
        valid_grams = grams
    
    # Sort by:
    # 1. Whether it ends with punctuation (prefer without)
    # 2. Length (prefer shorter)
    def score(gram):
        ends_with_punct = gram.rstrip().endswith((',', '.', ':', ';'))
        word_count = len(gram.split())
        return (ends_with_punct, word_count)
    
    valid_grams.sort(key=score)
    return valid_grams[0]


def main():
    # 1. Load the filtered grams
    print(f"📚 Loading grams from '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, "r") as f:
            grams = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{INPUT_FILE}'")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in '{INPUT_FILE}': {e}")
        return
    
    # Remove duplicates first
    unique_grams = list(set(grams))
    print(f"   Found {len(grams)} grams ({len(unique_grams)} unique)")
    
    # 2. Generate embeddings for all grams
    print(f"\n🧠 Generating embeddings using Ollama (embeddinggemma:300m)...")
    print(f"   This may take a few minutes...")
    
    embeddings = []
    valid_grams = []  # Track which grams have valid embeddings
    
    for gram in tqdm(unique_grams, desc="Embedding"):
        emb = get_ollama_embedding(gram)
        if emb:
            embeddings.append(emb)
            valid_grams.append(gram)
        else:
            print(f"   ⚠️ Skipping gram with failed embedding: '{gram}'")
    
    embeddings = np.array(embeddings)
    print(f"   ✅ Generated {len(embeddings)} embeddings")
    
    if len(embeddings) < 2:
        print("❌ Not enough embeddings to cluster. Check Ollama is running.")
        return
    
    # 3. Cluster similar grams
    print(f"\n🔗 Clustering similar grams (threshold={DISTANCE_THRESHOLD})...")
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Group grams by cluster
    clusters = defaultdict(list)
    for gram, label in zip(valid_grams, cluster_labels):
        clusters[label].append(gram)
    
    print(f"   Found {len(clusters)} clusters from {len(valid_grams)} grams")
    
    # Show some example clusters
    print(f"\n📊 Example clusters (showing first 5 with >1 member):")
    multi_member_clusters = [(label, members) for label, members in clusters.items() if len(members) > 1]
    for label, members in multi_member_clusters[:5]:
        best = select_best_from_cluster(members)
        print(f"   Cluster {label}: keeping '{best}'")
        print(f"      from: {members[:4]}{'...' if len(members) > 4 else ''}")
    
    # 4. Select best gram from each cluster
    print(f"\n🏆 Selecting best gram from each cluster...")
    deduplicated_grams = []
    
    for label, members in clusters.items():
        best = select_best_from_cluster(members)
        deduplicated_grams.append(best)
    
    # Sort alphabetically
    deduplicated_grams.sort()
    
    # 5. Save results
    print(f"\n💾 Saving {len(deduplicated_grams)} deduplicated grams to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(deduplicated_grams, f, indent=2)
    
    # 6. Summary
    print(f"\n{'='*50}")
    print(f"🏆 DEDUPLICATION COMPLETE")
    print(f"{'='*50}")
    print(f"   Input:  {len(unique_grams)} unique grams")
    print(f"   Output: {len(deduplicated_grams)} deduplicated grams")
    print(f"   Reduction: {100 * (1 - len(deduplicated_grams)/len(unique_grams)):.1f}%")
    print(f"\n   Saved to: {OUTPUT_FILE}")
    
    # Show sample of final grams
    print(f"\n📝 Sample deduplicated grams:")
    for gram in deduplicated_grams[:15]:
        print(f"   • {gram}")
    print("   ...")


if __name__ == "__main__":
    main()
