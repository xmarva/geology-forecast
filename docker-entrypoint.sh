#!/bin/bash
set -e

# W&B is optional for most notebooks; the sweep in models-and-hyperparameters.ipynb needs it.
if [ -n "$WANDB_API_KEY" ]; then
  echo "Logging in to Weights & Biases..."
  wandb login --relogin "$WANDB_API_KEY"
else
  echo "WANDB_API_KEY is not set: W&B logging is off."
fi

exec "$@"
