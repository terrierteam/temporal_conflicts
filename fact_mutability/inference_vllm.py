import argparse
import json
import os
import numpy as np
import wandb
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    T5TokenizerFast,
    PreTrainedTokenizerFast,
)
from vllm import LLM, SamplingParams

# Note: vLLM handles device management automatically, so the 'device' global is no longer needed.

# Constants from the original script
MAX_ANSWER_LENGTH = 50 # Was 50 in GenerationConfig, let's use that
DEF_TEMPLATE_TO_USE = "query_in_response"
DEF_INSTRUCTION = "Complete the fact in as few words as possible"

TEMPLATES = {
    "query_in_instructions": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}: {}\n\n### Response:"
    ),
    "query_in_response": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response: {}"
    ),
    "query_in_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
    ),
}


def prepare_prompt(query, model_name_or_path, instruction, template=None):
    # This function remains unchanged as it's about string formatting
    if "alpaca" in model_name_or_path:
        instruction = instruction
        template = TEMPLATES[template]
        return template.format(instruction, query)
    elif "flan" in model_name_or_path:
        if len(instruction):
            return "{}: {}".format(instruction, query)
        else:
            return query
    elif "instruct" in model_name_or_path:
        return "{}\n{}".format(instruction, query)
    elif "chat" in model_name_or_path:
        return "[INST] {}: {} [/INST] ".format(instruction, query)
    else:
        return query


# This helper is no longer needed as vLLM's output separates prompt and completion.
# get_sequence = { ... }

# Tokenizer-specific IDs for post-processing
ids_to_ignore = {
    # Ignore BOS, EOS.
    LlamaTokenizer: [1, 2],
    # Ignore EOS.
    T5TokenizerFast: [1],
    # Ignore EOS.
    PreTrainedTokenizerFast: [11],
}
# Token id of a full stop
full_stop = {LlamaTokenizer: 29889, T5TokenizerFast: 5, PreTrainedTokenizerFast: 25}


def process_vllm_output(vllm_req_output, tokenizer):
    """
    Processes a single RequestOutput from vLLM to calculate scores and perplexity.
    Assumes greedy decoding (n=1, temperature=0).
    """
    completion_output = vllm_req_output.outputs[0]
    
    # vLLM's output token_ids are only for the completion, not the prompt
    sequence = completion_output.token_ids
    logprobs = completion_output.logprobs

    token_scores = []
    trimmed_sequence = []
    
    # logprobs is a list of dicts: [{token_id: log_prob}, ...]
    for idx, logprob_dict in zip(sequence, logprobs):
        # The logprob_dict contains the log probabilities of the top tokens.
        # We need the log probability of the token that was actually generated.
        token_logprob = logprob_dict[idx]
        
        if idx not in ids_to_ignore.get(type(tokenizer), []):
            # Convert log probability to probability
            token_scores.append(np.exp(token_logprob))
            trimmed_sequence.append(idx)

    if trimmed_sequence and trimmed_sequence[-1] == full_stop.get(type(tokenizer)):
        token_scores = token_scores[:-1]
        trimmed_sequence = trimmed_sequence[:-1]
        
    answer = tokenizer.decode(trimmed_sequence).strip()
    words = answer.split()
    
    # Same warning logic as the original script
    if (
        not token_scores
        or not words
        or (
            (len(token_scores) == 1 or len(words) == 1)
            and words[0].lower() in ["the", "a", "an"]
        )
    ):
        print(
            "Warning: Empty or trivial generation. Prompt: '{}', Output sequence: {}".format(
                vllm_req_output.prompt, sequence
            )
        )
        return "", [], 0, float("inf")
        
    first_token_score = (
        token_scores[1] if words[0].lower() in ["the", "a", "an"] and len(token_scores) > 1 else token_scores[0]
    )
    
    # Calculate perplexity from probabilities
    perplexity = np.exp(-np.mean(np.log(token_scores)))

    return answer, token_scores, first_token_score, perplexity


def inference(dataset, tokenizer, llm, args):
    # 1. Prepare SamplingParams for vLLM
    # We use greedy decoding to match the original script's behavior (num_beams=1, do_sample=False)
    # logprobs=1 is needed to get the score of the generated token.
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=MAX_ANSWER_LENGTH,
        logprobs=1, # Request log probabilities for the top 1 token
    )
    
    # 2. Prepare all prompts in a batch
    prompts = []
    qcodes_and_queries = []
    for line in dataset:
        qcode, query = line.split("\t")
        prompt = prepare_prompt(
            query, args.model_name_or_path, args.instruction, args.template
        )
        prompts.append(prompt)
        qcodes_and_queries.append((qcode, query))
        
    # 3. Run batch inference with vLLM
    vllm_outputs = llm.generate(prompts, sampling_params)

    # 4. Process the results
    outputs = {"raw_predictions": [], "predictions": []}
    for vllm_output, (qcode, query) in tqdm(zip(vllm_outputs, qcodes_and_queries), total=len(prompts)):
        answer, token_scores, first_token_score, perplexity = process_vllm_output(
            vllm_output, tokenizer
        )
        
        raw_answer = vllm_output.outputs[0].text
        raw_output_ids = vllm_output.prompt_token_ids + vllm_output.outputs[0].token_ids

        outputs["raw_predictions"].append(
            {
                "qcode": qcode,
                "query": query,
                "predictions": [
                    {
                        "output_ids": raw_output_ids,
                        "answer": tokenizer.decode(raw_output_ids),
                    }
                ],
            }
        )
        outputs["predictions"].append(
            {
                "qcode": qcode,
                "query": query,
                "predictions": [
                    {
                        "answer": answer,
                        "per_token_probability": token_scores,
                        "first_token_probability": first_token_score,
                        "perplexity": perplexity,
                    }
                ],
            }
        )
    return outputs


def main(args):
    experiment_name = "{}--{}".format(
        args.exp_name, args.model_name_or_path.replace("/", "-")
    )
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print("Loading model with vLLM")
    # vLLM handles multi-GPU tensor parallelism automatically.
    # You can specify `tensor_parallel_size` if needed, e.g., LLM(..., tensor_parallel_size=2)
    llm = LLM(model=args.model_name_or_path)

    print("Loading tokenizer")
    use_fast = "llama" not in args.model_name_or_path.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=use_fast
    )
    # vLLM does not require setting pad_token for generation like transformers does.

    print("Loading dataset")
    with open(args.queries_path) as f:
        dataset = f.read().strip().split("\n")

    print("Running inference with vLLM")
    outputs = inference(dataset, tokenizer, llm, args)

    print("Writing outputs")
    for key in outputs:
        with open(os.path.join(experiment_dir, key + ".json"), "w") as outfile:
            # Use json.dump for proper JSON formatting of each line
            for item in outputs[key]:
                outfile.write(json.dumps(item) + "\n")

    with open(os.path.join(experiment_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with vLLM")
    parser.add_argument(
        "--queries_path",
        type=str,
        default="data/templama/val.txt",
        help="Path to txt file, one query per line",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=DEF_TEMPLATE_TO_USE,
        help="query_in_instructions, query_in_response or query_in_input",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEF_INSTRUCTION,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="huggyllama/llama-7b",
        help="Model name or path",
    )
    # The --cache_dir argument is no longer needed as vLLM uses its own cache mechanism.
    # We can remove it or ignore it. Let's remove it for clarity.
    # parser.add_argument("--cache_dir", type=str, default=None)
    
    args = parser.parse_args()

    # The original script had wandb integration, keeping it here.
    project_name = "lm_mutability_preds_eval_vllm"
    wandb.init(
        project=project_name,
        name="(inference) " + args.exp_name,
        config=args,
    )

    main(args)