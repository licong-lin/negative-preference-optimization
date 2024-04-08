from tqdm import tqdm
from data_module import TextDatasetQA
from unlearn_author.data_module import custom_data_collator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os, hydra
import evaluate
import json
from pathlib import Path
from rouge_score import rouge_scorer
from ..utils import get_model_identifiers_from_yaml


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg):
    # create cfg.save_dir if it doesn't exist
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    save_filename = os.path.join(cfg.save_dir, f"{cfg.eval_task}.json")
    if os.path.exists(save_filename) and not cfg.overwrite:
        print(f"Skipping {cfg.eval_task} because {save_filename} already exists")
        return

    # if config.json doesn't exist in model_path, return directly

    # if not os.path.exists(os.path.join(cfg.model_path, "config.json")):
    #     print(f"Skipping {cfg.eval_task} because {cfg.model_path} doesn't exist")
    #     return


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
    torch_format_dataset = TextDatasetQA(folder, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split, question_key=cfg.question_key, answer_key=cfg.answer_key)

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))


    batch_size = 16

    model = None
    config = AutoConfig.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", trust_remote_code = True, device_map=device_map)
    for attempt in range(3):
        try:
        # do thing
            if cfg.use_pretrained:
                model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            else:
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
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
    gen_outputs = []
    ground_truths = []
    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
        res = eval_accuracy(logits=outputs.logits, labels=batch["labels"])
        #add loss to res
        res["eval loss"] = outputs.loss.item()

        for k, v in res.items():
            eval_logs[k] = eval_logs.get(k, []) + [v]

    #average the metrics
    # for k, v in eval_logs.items():
    #     eval_logs[k] = sum(v) / len(v)

    # if any(name in cfg.eval_task for name in ['real_author', 'real_world']):
    #     eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))

    # else:
    #     eval_logs.update(eval_bleu(gen_outputs, ground_truths))
    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))

    if cfg.save_generated_text:
        eval_logs['generated_text'] = [(a, b) for a, b in zip(gen_outputs,ground_truths)]
        # eval_logs['ground_truth'] = ground_truths
    #json save the logs at cfg.save_dir
    # if cfg.save_dir is None:
    #     cfg.save_dir = cfg.model_path

    with open(save_filename, "w") as f:
        # pretty write json to f
        json.dump(eval_logs, f, indent=4)

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    #add ["/INST "] to the end of each string
    if cfg.model_family == 'llama2-7b':
        input_strings = [s + split_symbol for s in input_strings]
        
    #we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]
    
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id


    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    #now generate
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=cfg.generation.max_length, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return strs, ground_truth

def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores['rouge1'].recall)
        rougeL_recall.append(rouge_scores['rougeL'].recall)

    # rouge1_recall /= len(gen_outputs)
    # rougeL_recall /= len(gen_outputs)

    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}

if __name__ == "__main__":
    main()

