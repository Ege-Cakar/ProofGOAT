#!/bin/bash
# Installation script for Lean 4 and Lake on SLURM cluster
# This installs elan (Lean version manager) which provides the 'lake' command

set -e

echo "Installing elan (Lean version manager)..."
echo "This will install elan to ~/.elan"

# Download and run elan installer
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh

# Add elan to PATH for current session
export PATH="$HOME/.elan/bin:$PATH"

# Check if .bashrc exists and add elan to PATH if not already present
if [ -f "$HOME/.bashrc" ]; then
    if ! grep -q '\.elan/bin' "$HOME/.bashrc"; then
        echo "" >> "$HOME/.bashrc"
        echo "# Add elan to PATH for Lean/Lake" >> "$HOME/.bashrc"
        echo 'export PATH="$HOME/.elan/bin:$PATH"' >> "$HOME/.bashrc"
        echo "Added elan to ~/.bashrc"
    else
        echo "elan already in ~/.bashrc"
    fi
fi

# Reload shell config or use the PATH we just set
export PATH="$HOME/.elan/bin:$PATH"

echo ""
echo "Installing Lean 4 v4.26.0-rc2 (required by repl/)..."
elan toolchain install leanprover/lean4:v4.26.0-rc2
elan default leanprover/lean4:v4.26.0-rc2

echo ""
echo "Verifying installation..."
if command -v elan &> /dev/null; then
    echo "✓ elan installed: $(elan --version)"
else
    echo "✗ elan not found in PATH"
    echo "   Please run: export PATH=\"\$HOME/.elan/bin:\$PATH\""
    exit 1
fi

if command -v lake &> /dev/null; then
    echo "✓ lake installed: $(lake --version)"
else
    echo "✗ lake not found"
    echo "   Please ensure elan is in your PATH and try: elan default leanprover/lean4:v4.26.0-rc2"
    exit 1
fi

if command -v lean &> /dev/null; then
    echo "✓ lean installed: $(lean --version)"
else
    echo "✗ lean not found"
    exit 1
fi

echo ""
echo "Installation complete! 🎉"
echo ""
echo "If you're in a new shell, you may need to run:"
echo "  export PATH=\"\$HOME/.elan/bin:\$PATH\""
echo "Or restart your terminal/session to load ~/.bashrc"

