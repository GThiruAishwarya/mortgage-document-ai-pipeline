#!/usr/bin/env bash
set -e

echo "============================================="
echo " Label Studio Setup - Mortgage Review Tool"
echo "============================================="
echo

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+ and try again."
    exit 1
fi

echo "Python version: $(python3 --version)"
echo

echo "Installing Label Studio and requests..."
pip3 install label-studio requests

echo
echo "============================================="
echo " Installation complete!"
echo "============================================="
echo
echo "To start Label Studio:"
echo "  label-studio start"
echo
echo "Then open http://localhost:8080 in your browser."
echo
echo "If 'label-studio' is not found, add it to PATH:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  (Add this line to ~/.bashrc or ~/.zshrc for permanent effect)"
