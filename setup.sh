#!/bin/bash
mkdir -p ~/.kaggle
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
  echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  echo "✅ Kaggle credentials configured successfully."
else
  echo "⚠️ Kaggle credentials not found. Will use synthetic data fallback."
fi