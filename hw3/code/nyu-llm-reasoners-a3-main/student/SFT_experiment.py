from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel                          
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import torch 
import torch.optim as optim
import wandb
from unittest.mock import patch
from student.SFT import (
    tokenize_prompt_and_output,
    compute_entropy, 
    get_response_log_probs, 
    masked_normalize, 
    sft_microbatch_train_step,
)
import random

from pathlib import Path

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams

from student.drgrpo_grader import question_only_reward_fn

def load_prompt(name: str = "intellect") -> str:
    path = Path(__file__).parent / "prompts" / f"{name}.prompt"
    return path.read_text()


def evaluate(llm, prompts, ground_truths):
    """Run evaluation and return accuracy."""
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    outputs = llm.generate(prompts, params)

    correct = 0
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Grading")):
        text = output.outputs[0].text
        reward = question_only_reward_fn(text, ground_truths[i])
        results.append({
            "idx": i,
            "prompt": prompts[i][:500],
            "ground_truth": ground_truths[i],
            "response": text,
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
            "reward": reward["reward"],           
        })
        correct += reward["reward"]

    return correct / len(outputs), results


def train_step(model, tokenizer, all_prompts, all_outputs, optimizer, batch_size, gradient_accumulation_steps, device):
    """
        One full training step = one optimizer update.
    """
    microbatch_size = batch_size // gradient_accumulation_steps
    
    # 3a. Sample a batch of question-response pairs
    indices = random.sample(range(len(all_prompts)), batch_size)
    
    optimizer.zero_grad()
    total_loss = 0

    for mb in range(gradient_accumulation_steps):
        mb_start = mb * microbatch_size
        mb_end = mb_start + microbatch_size
        mb_indices = indices[mb_start:mb_end]
        prompt_strs = [all_prompts[i] for i in mb_indices]
        output_strs = [all_outputs[i] for i in mb_indices]

        # 3b. Tokenize them using tokenize_prompt_and_output
        batch = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)

        # 3c. Get lob probs using get_response_log_probs
        policy_log_probs = get_response_log_probs(model,
                            input_ids,
                            labels,
                            return_token_entropy = False)["log_probs"]
        # 3d. Compute loss and update using sft_microbatch_train_step

        loss, metadata = sft_microbatch_train_step(policy_log_probs,
                                response_mask,
                                gradient_accumulation_steps
                                )

        total_loss += loss.item()
        
    # 3e. Step optimizer  every gradient_accumulation_step microbatches.
    optimizer.step()
    return total_loss    



def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

# Monkeypatch from TRL:
# https://github.com/huggingface/trl/blob/
# 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
# Patch vLLM to make sure we can
# (1) place the vLLM model on the desired device (world_size_patch) and
# (2) avoid a test that is not designed for our setting (profiling_patch).

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
        model=model_id,
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        )
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())




def main():
    import argparse

    # 1. Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    device1 = "cuda:0"
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0  # Use AdamW if you want proper decoupled weight decay
    )
    output_dir = "C:\\Users\\zyr11\\OneDrive\\桌面\\job search\\ML\\reasoning-rl-from-scratch\\hw3\\code\\nyu-llm-reasoners-a3-main\\checkpoints\\sft"
    model.to(device1)
    batch_size = 32
    eval_interval = 30
    gradient_accumulation_steps = 4
    n_sft_steps = 1024

    # 2. Load the dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--intellect-path", default="data/intellect_math_train_dev_test/train")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    prompt_template = load_prompt("intellect")
    dataset = load_from_disk(args.intellect_path)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    
    all_prompts, all_outputs = [], []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assist_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        all_prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        all_outputs.append(assist_msg if assist_msg else "")    

    math_ds = load_dataset("hiyouga/math12k", split="test")
    if args.max_examples:
        math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

    prompts_m = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts_m = [ex["answer"] for ex in math_ds]


    # 3. Set up the training loop 
    llm = init_vllm(model_id="Qwen/Qwen2.5-Math-1.5B",
            device="cuda:1", 
            seed=42, 
            gpu_memory_utilization=0.85)

    wandb.init(
        project="sft-experimnet",  # Name of your W&B project
        config={  # Optional: hyperparameters & metadata
            "learning_rate": 1e-5,
            "batch_size": 32,
            "gradient_accumulation_steps": 4,
            "n_sft_steps": 1024,
            "eval_interval": 30,
            "num_examples": 512
        },
    )

    # Setup wandb metrics
    wandb.define_metric("train_step") # the x-axis for training
    wandb.define_metric("eval_step") # the x-axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    for step in range(n_sft_steps):
        loss = train_step(model, tokenizer, all_prompts, all_outputs, optimizer, batch_size, gradient_accumulation_steps, device1)
        wandb.log({"train/loss": loss, "train_step": step})

    # 4. Periodically evaluate
        if (step + 1) % eval_interval == 0:
            load_policy_into_vllm_instance(model, llm)
            acc, _ = evaluate(llm, prompts_m, gts_m)

    # 5. Log metrics
            wandb.log({"eval/math_accuracy": acc, "eval_step": step})


    # Save the model weights
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    wandb.finish() 

if __name__ == "__main__":
    main()


