from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch 
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
# Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
# other tokens (prompt or padding).
# Args:
#     prompt_strs: list[str] List of prompt strings.
#     output_strs: list[str] List of output strings.
#     tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
# Returns:
#     dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
#     the tokenized prompt and output strings. Then the returned dictionary should have the
#     following keys:
#     input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
#     the tokenized prompt and output strings, with the final token sliced off.
#     labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
#     shifted input ids, i.e., the input ids without the first token.
#     response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
#     1): a mask on the response tokens in the labels
    
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    prompt_tok = tokenizer(prompt_strs)
    output_tok = tokenizer(output_strs)

    raw_combined = [p + o for p, o in zip(prompt_tok["input_ids"],output_tok["input_ids"])]
    
    prompt_lens = [len(ids) for ids in prompt_tok["input_ids"]]
    max_len = max(len(c) for c in raw_combined)

    padded_ids = []
    attention_mask = []

    for c in raw_combined:
        pad_len = max_len - len(c)
        padded_ids.append(c + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * len(c) + [0] * pad_len)
    raw_input_ids = torch.tensor(padded_ids)
    response_mask = torch.tensor(attention_mask)

    for i, plen in enumerate(prompt_lens):
        response_mask[i, :plen] = 0
    response_mask = response_mask[:, 1:]
    
    input_ids = raw_input_ids[:, :-1]
    labels = raw_input_ids[:, 1:]

    dictionary = {"input_ids": input_ids,
                  "labels": labels,
                  "response_mask": response_mask}

    return dictionary


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    # Args:
    # logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    # containing unnormalized logits.
    # Returns:
    # torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    # prediction
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

    return entropy


def get_response_log_probs(model: PreTrainedModel,
                           input_ids: torch.Tensor,
                           labels: torch.Tensor,
                           return_token_entropy: bool = False,
                           )-> dict[str, torch.Tensor]:
    # Args:
    # 8
    # model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
    # and in inference mode if gradients should not be computed).
    # input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
    # response tokens as produced by your tokenization method.
    # labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
    # tokenization method.
    # return_token_entropy: bool If True, also return per-token entropy by calling
    # compute_entropy.
    # Returns:
    # dict[str, torch.Tensor].
    # "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
    # log pθ (xt | x<t).
    # "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
    # for each position (present only if return_token_entropy=True).
    

    output = model(input_ids)
    logits = output.logits
    log_prob_vocab = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_prob_vocab, dim=-1, index=labels.unsqueeze(-1))
    log_probs = log_probs.squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs": log_probs,
                "token_entropy": token_entropy} 
    return {"log_probs": log_probs}
        


def masked_normalize(tensor: torch.Tensor,
                     mask: torch.Tensor,
                     normalize_constant: float,
                     dim: int | None = None,
                     ) -> torch.Tensor:
# Sum over a dimension and normalize by a constant, considering only those elements where mask
# == 1.
# Args:
# tensor: torch.Tensor The tensor to sum and normalize.
# mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
# 9
# normalize_constant: float the constant to divide by for normalization.
# dim: int | None the dimension to sum along before normalization. If None, sum over all
# dimensions.
# Returns:
# torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
# the sum.
    tensor_new = tensor * mask
    return tensor_new.sum(dim) / normalize_constant

def sft_microbatch_train_step(policy_log_probs: torch.Tensor,
                              response_mask: torch.Tensor,
                              gradient_accumulation_steps: int,
                              normalize_constant: float = 1.0,
                              ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
# Execute a forward-and-backward pass on a microbatch.
# Args:
# policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
# SFT policy being trained.
# response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
# prompt/padding.
# gradient_accumulation_steps Number of microbatches per optimizer step.
# normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
# Returns:
# tuple[torch.Tensor, dict[str, torch.Tensor]].
# loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
# this so we can log it.
# metadata Dict with metadata from the underlying loss call, and any other statistics you
# might want to log.
    sft_loss = masked_normalize(-policy_log_probs, response_mask, normalize_constant, dim=-1).mean() / gradient_accumulation_steps
    sft_loss.backward()
    metadata = {}
    return sft_loss, metadata
    
    



def main():
    # model = AutoModelForCausalLM.from_pretrained(
    # "Qwen/Qwen2.5-Math-1.5B",
    # torch_dtype=torch.bfloat16,
    # )
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")


    # input_ids = train_batch["input_ids"].to(device)
    # labels = train_batch["labels"].to(device)
    # logits = model(input_ids).logits
    # loss = F.cross_entropy(..., ...)

    # model.save_pretrained(save_directory=output_dir)
    # tokenizer.save_pretrained(save_directory=output_dir)

    pass    
    
if __name__ == "__main__":
    main()


