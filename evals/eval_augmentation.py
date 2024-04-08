from tqdm import tqdm
from data_module import TextDatasetQA, get_model_identifiers_from_yaml, custom_data_collator
from unlearn_author.data_module import get_batch_loss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, hydra
import json
from pathlib import Path

from ..utils import get_model_identifiers_from_yaml


@hydra.main(version_base=None, config_path="config", config_name="eval_augmentation")
def main(cfg):
    # create cfg.save_dir if it doesn't exist
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)


    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    folder = cfg.data_path
    max_length = 500
    torch_format_dataset = TextDatasetQA(folder, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split, question_key=cfg.question_key, answer_key=cfg.base_answer_key)

    perturb_torch_format_dataset = TextDatasetQA(folder, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split, question_key=cfg.question_key, answer_key=cfg.compare_answer_key)
    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(range(min(cfg.ds_size, len(perturb_torch_format_dataset.data))))

    batch_size = 16
    model = None
    for attempt in range(3):
        try:
        # do thing
            if cfg.use_pretrained:
                model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            else:
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
        except Exception as e:
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")
    eval_logs = {}
    #write custom eval loop using compute_metrics

    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=batch_size, collate_fn=custom_data_collator
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=batch_size, collate_fn=custom_data_collator
    )

    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}


        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)


        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)

        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        # eval_logs['average_perturb_loss'] = eval_logs.get('average_perturb_loss', []) + [(perturb_loss.sum()/num_token_perturb).item()]
        # eval_logs['average_ground_truth_loss'] = eval_logs.get('average_ground_truth_loss', []) + [(gt_loss.sum()/num_token_gt).item()]

        eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + gt_loss.tolist()
        eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + perturb_loss.tolist()

        eval_logs['num_token_gt'] = eval_logs.get('num_token_gt', []) + num_token_gt.tolist()
        eval_logs['num_token_perturb'] = eval_logs.get('num_token_perturb', []) + num_token_perturb.tolist()

    #average the metrics
    # for k, v in eval_logs.items():
    #     eval_logs[k] = sum(v) / len(v)

    with open(os.path.join(cfg.save_dir, f"{cfg.eval_task}.json"), "w") as f:
        # pretty write json to f
        json.dump(eval_logs, f, indent=4)

if __name__ == "__main__":
    main()

