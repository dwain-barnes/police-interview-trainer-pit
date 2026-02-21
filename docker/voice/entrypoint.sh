#!/bin/bash
set -e

# If HF_TOKEN is provided, log in so huggingface_hub can find it
# (Chatterbox-Turbo calls snapshot_download with token=True)
if [ -n "$HF_TOKEN" ]; then
  echo "==> Logging in to Hugging Face..."
  python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
fi

echo "==> Starting PIT voice service on port 8001..."
exec python3 -m uvicorn serve_voice:app --host 0.0.0.0 --port 8001
