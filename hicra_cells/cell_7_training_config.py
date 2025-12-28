# Cell 7: Training Configuration
from torch.optim import AdamW

# Configuration
GRAD_ACCUM_STEPS = 4
EPOCHS = 1
MAX_STEPS = 250
LEARNING_RATE = 5e-6
MAX_NEW_TOKENS = 300
NUM_GENERATIONS = 4  # Group size (G)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Ensure model is in training mode
model.train()

print("✅ Training configuration set")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Gradient accumulation: {GRAD_ACCUM_STEPS}")
print(f"   Max steps: {MAX_STEPS}")
