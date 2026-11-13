
import torch
import wandb
from student.evaluate import load_prompt
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM

def wandb_init(args):
    wandb.init(
        project=args.wandb_project, 
        name=args.wandb_run_name, 
        config=vars(args)
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")



def init_vllm(model_id, seed, gpu_memory_utilisation=1.0):
    from vllm.model_executor import set_random_seed

    set_random_seed(seed)

    llm = LLM(
        model_id,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilisation= gpu_memory_utilisation
    )

    return llm



def train_loop(args):

    n_grpo_steps = 200
    learning_rate = 1e-5
    advantage_eps = 1e-6
    rollout_batch_size = 16
    group_siz = 8
    sampling_tempreture = 0.7
    sampling_min_tokens = 4
    sampling_max_tokens = 1024
    epochs_per_rollout_batch = 1
    train_batch_size = 64
    gradient_accumulation_steps = 128
    gpu_memory_utilization = 0.8
    loss_type = ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    use_std_normalization = True

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95)
    )

    torch.manual_seed(args.seed)

    wandb_init(args)

    prompt = load_prompt("countdown")

    print(f"\n=== Countdown Test ({args.countdown_path}) ===")

    dataset =  load_dataset(args.countdown_path)

    print("first exmaples", dataset[0])

    #create a dataloader

    prompts, gts = [], []
    for ex in dataset: 
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        # this is for reasoning trace
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"),"")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        gts.append(ex.get("ground_truth", ""))


    
    # load the policy model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    policy = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)



    # run vLLM for specific rollouts for given promo (host policy on vLLM)

    init_vllm()

    # score each rollout


    #load the updated policy into model

    # compute grouped normalised advantages

    # compute grpo clip loss 

    # run microbatch step






def main(): 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--intellect-path", default="/gpfs/scratch/sd6701/personal/salt/grpo/data-distrib/intellect_math/test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--wandb_project", type=str, default="sft-intellect")
    parser.add_argument("--wandb_run_name", type=str, default="qwen-math-sft")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--normalize_constant", type=float, default=1.0)

    parser.add_argument("--save_dir", type=str, default="./checkpoints") 

    args = parser.parse_args()
    train_loop()








if __name__ == "__main__":
    main()

