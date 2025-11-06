#!/bin/bash
# Example shell script demonstrating positional argument support in params-proto v3

set -e  # Exit on error

echo "=== Testing positional argument support ==="
echo ""

echo "1. Positional argument only (seed=42):"
uv run python ./samples/train.py 42
echo ""

echo "2. Positional with named args (seed=42, lr=0.01):"
uv run python ./samples/train.py 42 --lr 0.01
echo ""

echo "3. All named arguments (seed=99, lr=0.01, batch_size=128, epochs=20):"
uv run python ./samples/train.py --seed 99 --lr 0.01 --batch-size 128 --epochs 20
echo ""

echo "4. Positional with multiple named args:"
uv run python ./samples/train.py 123 --lr 0.005 --batch-size 64 --epochs 50
echo ""

echo "=== All tests passed! ==="
