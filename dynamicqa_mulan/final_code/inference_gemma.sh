#!/bin/bash

# Gemma 1B
echo "=== Gemma 1B : static ==="
python inference_answer_gemma.py --model_name google/gemma-3-1b-it --mode static

echo "=== Gemma 1B : static --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-1b-it --mode static --context

echo "=== Gemma 1B : temporal ==="
python inference_answer_gemma.py --model_name google/gemma-3-1b-it --mode temporal 

echo "=== Gemma 1B : temporal --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-1b-it --mode temporal --context

echo "=== Gemma 1B : dispute --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-1b-it --mode dispute --context


# Gemma 4B
echo "=== Gemma 4B : static ==="
python inference_answer_gemma.py --model_name google/gemma-3-4b-it --mode static

echo "=== Gemma 4B : static --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-4b-it --mode static --context

echo "=== Gemma 4B : temporal ==="
python inference_answer_gemma.py --model_name google/gemma-3-4b-it --mode temporal 

echo "=== Gemma 4B : temporal --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-4b-it --mode temporal --context

echo "=== Gemma 4B : dispute --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-4b-it --mode dispute --context


# Gemma 12B
echo "=== Gemma 12B : static ==="
python inference_answer_gemma.py --model_name google/gemma-3-12b-it --mode static

echo "=== Gemma 12B : static --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-12b-it --mode static --context

echo "=== Gemma 12B : temporal ==="
python inference_answer_gemma.py --model_name google/gemma-3-12b-it --mode temporal 

echo "=== Gemma 12B : temporal --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-12b-it --mode temporal --context

echo "=== Gemma 12B : dispute --context ==="
python inference_answer_gemma.py --model_name google/gemma-3-12b-it --mode dispute --context
