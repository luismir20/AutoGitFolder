#!/bin/bash
cd "$(dirname "$0")"  # Ensure the script runs in the folder's directory
git add .
git commit -m "Auto-commit: $(date)"
git push origin main
