# Main libraries changed from transformers to vLLM
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import argparse
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
import random

class inference():
    def __init__(self, args):
        self.args = args
        self.load_model()
        self.data = self.load_data()
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Define sampling parameters for vLLM.
        # Temperature=0 makes the output deterministic (greedy decoding), which is good for evaluation.
        # max_tokens corresponds to the original max_new_tokens.
        self.sampling_params = SamplingParams(max_tokens=20, temperature=0)

    def load_model(self):
        """
        Loads the model using the vLLM engine.
        Quantization and device mapping are handled automatically by vLLM.
        """
        print("Loading model with vLLM...")
        
        # Determine tensor parallel size (number of GPUs to use)
        if self.args.tensor_parallel_size:
            tp_size = self.args.tensor_parallel_size
        else:
            tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        print(f"Using tensor_parallel_size = {tp_size}")

        self.model = LLM(
            model=self.args.model_name, 
            tensor_parallel_size=tp_size,
            seed=self.args.seed # Pass seed for reproducible results
        )
        
        # The tokenizer is still needed to correctly format the prompts using chat templates.
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
    # end of load_model
    
    # The bit4_config method is no longer needed with vLLM.

    def load_data(self):
        if self.args.mode == "dispute":
            data = pd.read_csv("../Data/Disputable_final.csv", index_col=0)
        elif self.args.mode == "temporal":
            data = pd.read_csv("../Data/Temporal_final.csv", index_col=0)
        elif self.args.mode == "static":
            data = pd.read_csv("../Data/Static_final.csv", index_col=0)
        return data

    def run(self):
        # The commented-out logic suggests run_with_csv is the primary function.
        output_list = self.run_with_csv()

        # Save the output to a JSON file.
        if self.args.context:
            filename = "../Output/%s/%s_%s_context.json" % (self.args.mode, self.args.mode, self.args.model_name.split("/")[-1])
        else:
            filename = "../Output/%s/%s_%s_q_only.json" % (self.args.mode, self.args.mode, self.args.model_name.split("/")[-1])
        
        print(f"Saving results to {filename}")
        json.dump(output_list, open(filename, "w"), indent=4)

    def run_with_json(self):
        """
        Modified to use vLLM's batch processing.
        """
        prompts = []
        print("Preparing prompts from JSON data...")
        for d in tqdm(self.data):
            for idx, ctx in enumerate(d["contexts"]):
                q = d["questions"][0] or d["questions"][1]

                if self.args.context:
                    content = f"You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]. This article is about {d['subject']}. [Context]: {ctx} [Question]: {q} [Answer]:"
                else:
                    content = f"You'll be given a question about the article and answer it with a one word. Answer the [Question]. This article is about {d['subject']}. [Question]: {q} [Answer]:"
                
                user_msg = {"role": "user", "content": content}
                
                prompt_str = self.tokenizer.apply_chat_template([user_msg], tokenize=False)
                prompts.append(prompt_str)

        print(f"Generated {len(prompts)} prompts. Starting batch inference...")
        vllm_outputs = self.model.generate(prompts, self.sampling_params)
        
        output_list = [output.outputs[0].text.strip() for output in vllm_outputs]
        return output_list

    def run_with_csv(self):
        """
        Modified to prepare all prompts first, then run inference in a single batch with vLLM.
        """
        prompts = []
        golden_list = []
        
        print("Preparing prompts from CSV data...")
        # 1. First loop: Prepare all prompts and corresponding golden answers
        for i in tqdm(range(len(self.data))):
            line = self.data.iloc[i]
            answer_list = [line["obj"], line["replace_name"]]

            for gold_answer in answer_list:
                if self.args.mode == "dispute":
                    context = line["context"].replace(line["obj"], gold_answer)
                    article_info = f"This article is about {line['subj']}"
                    context = f"{article_info} {context}"
                else:
                    context = line["context"].replace("[ENTITY]", gold_answer)
                
                # --- Prompt Templating Logic (remains the same) ---
                use_system_prompt = "Qwen" in self.args.model_name or "llama" in self.args.model_name
                
                if self.args.context:
                    if use_system_prompt:
                        system_content = "You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]."
                        user_content = f"[Context]: {context} [Question]: {line['question']} [Answer]:"
                    else:
                        user_content = f"You'll be given a question and a context about the article and answer it with a one word. Answer the [Question]. [Context]: {context} [Question]: {line['question']} [Answer]:"
                else:
                    if use_system_prompt:
                        system_content = "You'll be given a question about the article and answer it with a one word. Answer the [Question]."
                        user_content = f"This article is about {line['subj']}. [Question]: {line['question']} [Answer]:"
                    else:
                        user_content = f"You'll be given a question about the article and answer it with a one word. Answer the [Question]. This article is about {line['subj']}. [Question]: {line['question']} [Answer]:"
                
                input_msgs = [{"role": "user", "content": user_content}]
                if use_system_prompt:
                    input_msgs.insert(0, {"role": "system", "content": system_content})

                # --- Convert to final prompt string ---
                prompt_str = self.tokenizer.apply_chat_template(input_msgs, tokenize=False)
                prompts.append(prompt_str)
                golden_list.append(gold_answer)

        # 2. Single call to vLLM for batch inference
        print(f"\nGenerated {len(prompts)} prompts. Starting batch inference...")
        vllm_outputs = self.model.generate(prompts, self.sampling_params)
        print("Inference complete. Processing outputs...")

        # 3. Second loop: Process results
        output_list = []
        score_list = []
        for i, output in enumerate(tqdm(vllm_outputs)):
            # vLLM returns the clean generated text, so no complex parsing is needed.
            answer = output.outputs[0].text.strip()
            gold_answer = golden_list[i]

            output_list.append(answer)
            score = self.evaluate(answer, gold_answer)
            score_list.append(score)
            
            # Print some examples to verify
            if i < 20: # prints the first 10 pairs
                print(f"Generated: '{answer}' | Golden: '{gold_answer}' | Score: {score}")

        print("\n--- Evaluation ---")
        print(self.args)
        print(f"Accuracy: {np.mean(score_list):.4f}")

        return output_list

    def evaluate(self, pred_str, gold_str):
        score_dict = self.scorer.score(str(pred_str), str(gold_str))
        tmp_f1 = score_dict["rougeL"].fmeasure
        return 1.0 if tmp_f1 > 0.3 else 0.0

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        print('No GPU available, using the CPU instead.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--mode", type=str, default="dispute", choices=["dispute", "temporal", "static"])
    parser.add_argument("--context", action="store_true", help="Include context in the prompt")
    parser.add_argument("--seed", type=int, default=10)
    # Replaced --bit4 and --device with vLLM specific arguments
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Number of GPUs for tensor parallelism. Defaults to all available GPUs.")

    args = parser.parse_args()
    set_seed(args)
    
    module = inference(args)
    module.run()