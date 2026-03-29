# Model Evaluation Guide

## Quick Start

```bash
# Inside your Docker container
python scripts/evaluate_model.py
```

## Interactive Workflow

The script will guide you through 4 steps:

### Step 1: Select Model Directory
```
Available model directories:
  1. ministral-sft-curriculum
  2. ministral-sft-unsloth

Select model directory (enter number): 1
✅ Selected: ministral-sft-curriculum
```

### Step 2: Select Checkpoint
```
Available checkpoints (newest first):
  1. checkpoint-1000
  2. checkpoint-950
  3. checkpoint-900
  4. checkpoint-850
  ...

Select checkpoint (enter number): 1
✅ Selected: checkpoint-1000
```

### Step 3: Select Tasks
```
Available tasks:
  1. ARC Challenge (arc_challenge)
  2. HellaSwag (hellaswag)
  3. Winogrande (winogrande)
  4. PIQA (piqa)
  5. MMLU (mmlu)
  6. GSM8K (gsm8k)
  7. TruthfulQA MC2 (truthfulqa_mc2)

Enter task numbers separated by commas (e.g., '1,3,5')
Or press Enter to select all tasks

Enter task selection: 
✅ Selected all 7 tasks
```

### Step 4: Set Evaluation Limit
```
The limit determines how many samples to evaluate per task.
Higher values give more accurate results but take longer.

Enter a number for 'limit' (0 for no limit): 50
✅ Limit set to 50 samples per task
```

## Example Scenarios

### Scenario 1: Quick Check (Small Sample)
- **Model**: ministral-sft-curriculum
- **Checkpoint**: checkpoint-1000
- **Tasks**: All (or just MMLU and GSM8K for reasoning)
- **Limit**: 25

**Use case**: Quick check of model performance after training

### Scenario 2: Comprehensive Evaluation
- **Model**: ministral-sft-curriculum
- **Checkpoint**: checkpoint-1000
- **Tasks**: All
- **Limit**: 0 (no limit - evaluate all available samples)

**Use case**: Final evaluation before moving to next training stage

### Scenario 3: Compare Checkpoints
Run the same evaluation on multiple checkpoints:
- checkpoint-500 (limit: 50)
- checkpoint-750 (limit: 50)
- checkpoint-1000 (limit: 50)

**Use case**: Track improvement across training progress

## Understanding Results

### Output Files
Results are saved to `logs/` directory with format:
```
{model_name}_{checkpoint_name}_eval_{timestamp}.json
```

Example:
```
logs/ministral-sft-curmium_checkpoint-1000_eval_20260327_143052.json
```

### Metrics Explained

- **Accuracy (acc)**: Percentage of correct answers
- **ARC Challenge**: Grade-school science questions
- **HellaSwag**: Common sense reasoning
- **Winogrande**: Pronoun resolution
- **PIQA**: Physical commonsense
- **MMLU**: Multi-subject knowledge
- **GSM8K**: Grade school math
- **TruthfulQA**: Truthfulness in generation

### Interpreting Scores

| Task | Good | Excellent |
|------|------|-----------|
| arc_challenge | 0.50+ | 0.70+ |
| hellaswag | 0.65+ | 0.80+ |
| winogrande | 0.60+ | 0.75+ |
| piqa | 0.70+ | 0.85+ |
| mmlu | 0.35+ | 0.60+ |
| gsm8k | 0.30+ | 0.70+ |
| truthfulqa_mc2 | 0.35+ | 0.55+ |

## Tips

1. **Start with small limits**: Use limit=25 for quick tests
2. **Compare systematically**: Same limit across checkpoints for fair comparison
3. **Focus on relevant tasks**: For reasoning, prioritize MMLU and GSM8K
4. **Save results**: Results are automatically saved with timestamps
5. **Monitor trends**: Look for consistent improvement across checkpoints

## Troubleshooting

### "lm_eval not found"
```bash
pip install lm_eval
```

### "Model not found"
Ensure your checkpoints are in the `checkpoints/` directory:
```
checkpoints/
  ├─ ministral-sft-curriculum/
  │   ├─ checkpoint-1000/
  │   └─ checkpoint-950/
  └─ ministral-sft-unsloth/
      └─ checkpoint-750/
```

### OOM (Out of Memory)
- Reduce batch_size in the script (line ~195)
- Use smaller limit values
- Evaluate on fewer tasks at once

## Advanced Usage

### Custom Tasks
You can add more tasks by editing the `DEFAULT_TASKS` list in the script:

```python
DEFAULT_TASKS = [
    ("arc_challenge", "ARC Challenge"),
    ("hellaswag", "HellaSwag"),
    # Add your custom tasks here
    ("custom_task", "Custom Task Name"),
]
```

### Batch Evaluation
For evaluating multiple checkpoints automatically, create a script:

```python
#!/usr/bin/env python3
"""Batch evaluation script for comparing checkpoints"""
import subprocess
import sys

checkpoints = [
    ("ministral-sft-curriculum", "checkpoint-500"),
    ("ministral-sft-curriculum", "checkpoint-750"),
    ("ministral-sft-curriculum", "checkpoint-1000"),
]

for model, checkpoint in checkpoints:
    print(f"\n{'='*70}")
    print(f"Evaluating {model}/{checkpoint}")
    print('='*70)
    
    # Run evaluation with predefined inputs
    subprocess.run([
        "python", "scripts/evaluate_model.py",
        "--model", model,
        "--checkpoint", checkpoint,
        "--limit", "50",
        "--tasks", "arc_challenge,mmlu,gsm8k"
    ])
```

## Next Steps After Evaluation

1. **Analyze Results**: Compare scores across checkpoints
2. **Identify Weaknesses**: Note which tasks show poor performance
3. **Adjust Training**: Modify curriculum or hyperparameters based on results
4. **Continue Training**: Resume from best checkpoint if needed
5. **Move to RL**: Once satisfied with SFT results, proceed to RL training
