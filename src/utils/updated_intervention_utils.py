import collections
from tqdm import tqdm
import einops
import torch

from src.utils.intervention_utils import (
    is_llama_tokenizer,
    get_dataloader,
    remove_invalid_token_id,
)


def nnsight_intervention(
        nnsight_model,
        nnsight_tracer_kwargs, 
        layer, 
        autoencoder, 
        inv_dims, 
        inputs, 
        split_to_inv_locations, 
        n_generated_tokens, 
        device, 
        tokenizer,
        add_reconstruction_error=True, 
        inv_positions=None, 
        verbose=False
    ):
    batch_size = inputs['input_ids'].shape[0]
    submodule = nnsight_model.model.layers[layer]
    
    # Organize intervention positions
    if inv_positions is None: # Default, custom intervention positions only for testing
        base_inv_positions = torch.tensor([split_to_inv_locations[inputs["split"][i]]["inv_position"] for i in range(batch_size)], device=device)
        source_inv_positions = torch.tensor([split_to_inv_locations[inputs["source_split"][i]]["inv_position"] for i in range(batch_size)], device=device)
    else:
        base_inv_positions, source_inv_positions = inv_positions
    
    # Indexing preparation
    if isinstance(inv_dims, range):
        inv_dims = torch.tensor(list(inv_dims), device=device, dtype=torch.int)
    if len(base_inv_positions.shape) > 1:
        base_inv_positions = base_inv_positions.squeeze(dim=-1)
    if len(source_inv_positions.shape) > 1:
        source_inv_positions = source_inv_positions.squeeze(dim=-1)

    batch_arange = einops.repeat(torch.arange(batch_size, device=device, dtype=torch.int), 'b -> b d', d=inv_dims.shape[0])
    base_inv_positions = einops.repeat(base_inv_positions, 'b -> b d', d=inv_dims.shape[0])
    source_inv_positions = einops.repeat(source_inv_positions, 'b -> b d', d=inv_dims.shape[0])
    inv_dims = einops.repeat(inv_dims, 'd -> b d', b=batch_size)

    # Forward pass on source input
    with torch.no_grad(), nnsight_model.trace(inputs['source_input_ids'], attention_mask=inputs['source_attention_mask'], **nnsight_tracer_kwargs):
        source_sae_acts = autoencoder.encode(submodule.output[0])
        source_sae_acts = source_sae_acts[batch_arange, source_inv_positions, inv_dims].save()

    # Forward pass on base input with intervention
    generated_inputs_shape = inputs['input_ids'].shape[:-1] + (inputs['input_ids'].shape[-1] + n_generated_tokens,)
    generated_inputs = torch.zeros(generated_inputs_shape, device=device, dtype=inputs['input_ids'].dtype)
    generated_inputs[:, :inputs['input_ids'].shape[-1]] = inputs['input_ids']
    generated_attn_mask = torch.cat([inputs['attention_mask'], torch.ones(batch_size, n_generated_tokens, device=device)], dim=-1)
    n_tokens = generated_attn_mask.sum(dim=-1).to(torch.int)
    generated_pos_ids = torch.zeros_like(generated_inputs, device=device, dtype=inputs['input_ids'].dtype)
    for batch_idx in range(batch_size):
        generated_pos_ids[batch_idx, -n_tokens[batch_idx]:] = torch.arange(n_tokens[batch_idx], device=device, dtype=inputs['input_ids'].dtype)

    for i in range(n_generated_tokens):
        with torch.no_grad(), nnsight_model.trace(generated_inputs, attention_mask=generated_attn_mask, position_ids=generated_pos_ids, **nnsight_tracer_kwargs):
            llm_acts = submodule.output[0]
            base_sae_acts = autoencoder.encode(llm_acts)
            llm_acts_reconstructed = autoencoder.decode(base_sae_acts)

            base_sae_acts[batch_arange, base_inv_positions, inv_dims] = source_sae_acts
            llm_acts_intervened = autoencoder.decode(base_sae_acts)

            if not add_reconstruction_error:
                submodule.output = (llm_acts_intervened.to(llm_acts.dtype),)
            else:
                reconstruction_error = llm_acts - llm_acts_reconstructed
                corrected_acts = llm_acts_intervened + reconstruction_error
                submodule.output = (corrected_acts.to(llm_acts.dtype),)

            counterfactual_logits = nnsight_model.lm_head.output.save()

        # Append generation
        final_token_pos = i - n_generated_tokens
        next_token = torch.argmax(counterfactual_logits[:, final_token_pos-1, :], dim=-1)
        generated_inputs[:, final_token_pos] = next_token

        if verbose:
            print(f'counterfactual_out_tokens decoded: {tokenizer.batch_decode(generated_inputs, skip_special_tokens=True)}')

    return generated_inputs


def eval_with_interventions(
    hf_model, # Native Hugging Face model
    nnsight_model,  # NNsight model wrapper
    nnsight_tracer_kwargs,  # NNsight tracer kwargs
    split_to_dataset,
    split_to_inv_locations,
    tokenizer,
    inv_dims,
    compute_metrics_fn,
    autoencoder,
    layer_idx,
    max_new_tokens=1,
    eval_batch_size=16,
    debug_print=False,
    forward_only=False,
    use_nnsight_replication=False,
    device='cuda',
):
    if not use_nnsight_replication:
        print("No intervention will take place.")
    else:
        num_inv = 1

    split_to_eval_metrics = {}
    padding_offset = 3 if is_llama_tokenizer(tokenizer) else 0
    for split in tqdm(split_to_dataset):
        # Asssume all inputs have the same max length.
        prompt_max_length = split_to_inv_locations[split_to_dataset[split][0]["split"]][
            "max_input_length"
        ]

        eval_dataloader = get_dataloader(
            split_to_dataset[split],
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            prompt_max_length=prompt_max_length,
            output_max_length=padding_offset + max_new_tokens,
            first_n=max_new_tokens,
        )
        #print(f'eval_dataloader: {next(iter(eval_dataloader))}')
        eval_labels = collections.defaultdict(list)
        eval_preds = []
        with torch.no_grad():
            if debug_print:
                epoch_iterator = tqdm(eval_dataloader, desc=f"Test")
            else:
                epoch_iterator = eval_dataloader
            for step, inputs in enumerate(epoch_iterator):
                b_s = len(inputs["input_ids"])
                position_ids = {
                    f"{prefix}position_ids": hf_model.prepare_inputs_for_generation(
                        input_ids=inputs[f"{prefix}input_ids"],
                        attention_mask=inputs[f"{prefix}attention_mask"],
                    )["position_ids"]
                    for prefix in ("", "source_")
                }
                inputs.update(position_ids)
                for key in inputs:
                    if key in (
                        "input_ids",
                        "source_input_ids",
                        "attention_mask",
                        "source_attention_mask",
                        "position_ids",
                        "source_position_ids",
                        "labels",
                        "base_labels",
                    ):
                        inputs[key] = inputs[key].to(hf_model.device)

                intervention_locations = {
                    "sources->base": (
                        [
                            [
                                split_to_inv_locations[inputs["source_split"][i]]["inv_position"]
                                for i in range(b_s)
                            ]
                        ]
                        * num_inv,
                        [
                            [
                                split_to_inv_locations[inputs["split"][i]]["inv_position"]
                                for i in range(b_s)
                            ]
                        ]
                        * num_inv,
                    )
                }

                if not use_nnsight_replication:
                    print("Canceling evaluation. No intervention will take place.")
                else:
                    base_outputs = hf_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=inputs["input_ids"].shape[1] + max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        num_beams=1,
                        do_sample=False,
                        output_scores=True,
                    )
                    counterfactual_out_tokens = nnsight_intervention(
                        nnsight_model,
                        nnsight_tracer_kwargs,
                        layer_idx,
                        autoencoder,
                        inv_dims,
                        inputs,
                        split_to_inv_locations,
                        n_generated_tokens=max_new_tokens,
                        device=device,
                        tokenizer=tokenizer,
                        add_reconstruction_error=True,
                    )
                    eval_preds.append(counterfactual_out_tokens)
                    
                for label_type in ["base_labels", "labels"]:
                    eval_labels[label_type].append(inputs[label_type])
                eval_labels["base_outputs"].append(base_outputs[:, -max_new_tokens:])

                if debug_print and step < 3:
                    print("\nInputs:")
                    print("Base:", inputs["input"][:3])
                    print("Source:", inputs["source_input"][:3])
                    print("Tokens to intervene:")
                    print(
                        "Base:",
                        tokenizer.batch_decode(
                            [
                                inputs["input_ids"][i][
                                    intervention_locations["sources->base"][1][0][i]
                                ]
                                for i in range(len(inputs["split"]))
                            ]
                        ),
                    )
                    print(
                        "Source:",
                        tokenizer.batch_decode(
                            [
                                inputs["source_input_ids"][i][
                                    intervention_locations["sources->base"][0][0][i]
                                ]
                                for i in range(len(inputs["split"]))
                            ]
                        ),
                    )
                    base_output_text = tokenizer.batch_decode(
                        base_outputs[:, -max_new_tokens:], skip_special_tokens=True
                    )
                    print("Base Output:", base_output_text)
                    print(
                        "Output:    ",
                        tokenizer.batch_decode(counterfactual_out_tokens[:, -max_new_tokens:]),
                    )
                    print(
                        "Inv Label: ",
                        tokenizer.batch_decode(
                            remove_invalid_token_id(
                                inputs["labels"][:, :max_new_tokens], tokenizer.pad_token_id
                            )
                        ),
                    )
                    base_label_text = tokenizer.batch_decode(
                        remove_invalid_token_id(
                            inputs["base_labels"][:, :max_new_tokens], tokenizer.pad_token_id
                        ),
                        skip_special_tokens=True,
                    )
                    print("Base Label:", base_label_text)
                    if base_label_text != base_output_text:
                        print("WARNING: Base outputs does not match base labels!")
                        
        eval_metrics = {
            label_type: compute_metrics_fn(
                tokenizer,
                eval_preds,
                eval_labels[label_type],
                last_n_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                extra_labels=eval_labels,
                eval_label_type=label_type,
            )
            for label_type in eval_labels
            if label_type.endswith("labels")
        }
        print("\n", repr(split) + ":", eval_metrics)
        split_to_eval_metrics[split] = {
            "metrics": eval_metrics,
            "inv_outputs": tokenizer.batch_decode(counterfactual_out_tokens[:, -max_new_tokens:]),
            "inv_labels": tokenizer.batch_decode(
                remove_invalid_token_id(
                    inputs["labels"][:, :max_new_tokens], tokenizer.pad_token_id
                )
            ),
            "base_labels": tokenizer.batch_decode(
                remove_invalid_token_id(
                    inputs["base_labels"][:, :max_new_tokens], tokenizer.pad_token_id
                )
            ),
        }

    return split_to_eval_metrics