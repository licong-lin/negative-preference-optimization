from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA, TextForgetDatasetKTOQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml, set_random_seed

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):

    seed = cfg.seed
    set_random_seed(seed)

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    if os.path.exists(cfg.save_dir):
        print("Directory already exists")
        if not cfg.overwrite_dir:
            exit()

    max_length = 500

    # determine the data path.
    if cfg.split in ['forget01','forget05','forget10']:
        data_path = 'locuslab/TOFU'
    elif cfg.split in ['forget20','forget35','forget50','forget90']:
        data_path = './TOFU_data'
    else:
        raise NotImplementedError

    if cfg.forget_loss in ["dpo","dpo_KL","dpo_grad_diff"]:
        torch_format_dataset = TextForgetDatasetDPOQA(data_path, 
                                                    tokenizer=tokenizer, 
                                                    model_family = cfg.model_family, 
                                                    max_length=max_length, 
                                                    split=cfg.split)

    elif 'kto' in cfg.forget_loss:
        torch_format_dataset = TextForgetDatasetKTOQA(data_path, 
                                                    tokenizer=tokenizer, 
                                                    model_family = cfg.model_family, 
                                                    max_length=max_length, 
                                                    split=cfg.split)

    else:
        torch_format_dataset = TextForgetDatasetQA(data_path, 
                                                    tokenizer=tokenizer, 
                                                    model_family = cfg.model_family, 
                                                    max_length=max_length, 
                                                    split=cfg.split, 
                                                    loss_type=cfg.forget_loss)
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"The length of dataset: {len(torch_format_dataset)},\nmax_steps: {max_steps},\nbatch_size: {batch_size},\naccumulation_step: {gradient_accumulation_steps}.")

    if isinstance(cfg.eval_steps, int):
        eval_steps = cfg.eval_steps
    elif cfg.eval_steps == 'steps_per_epoch':
        eval_steps = steps_per_epoch
    else:
        raise NotImplementedError("The eval_steps must be an integer or step_per_epoch.")
    
    if isinstance(cfg.warmup_steps, int):
        warmup_steps = cfg.warmup_steps
    elif cfg.warmup_steps == 'steps_per_epoch':
        warmup_steps = steps_per_epoch
    else:
        raise NotImplementedError("The warmup_steps must be an integer or step_per_epoch.")

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max_steps+1, # do not save the model
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps+1, # do not save the model
            ddp_find_unused_parameters= False,
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            evaluation_strategy = "steps",
            eval_steps = eval_steps,
    )
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    oracle_model = None

    if path_found:
        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
        
        oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)

    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, device_map=device_map)
        #now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
        #save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        #save the model for next time
        model.save_pretrained(cfg.model_path)
    
    

    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # edit the evaluation split when we aim to forget beyond 10 percent of the data.
    if cfg.split in ['forget01','forget05','forget10']:
        pass
    elif cfg.split in ['forget20','forget35','forget50','forget90']:
        cfg.eval.data_path = ['locuslab/TOFU', 'locuslab/TOFU', 'locuslab/TOFU', './TOFU_data']
        cfg.eval.split = 'forget10_perturbed' # we use the commonly available forget10 to evaluate the truth ratio on the forget set when we do forget20 - forget90.
        cfg.eval.split_list = ['retain_perturbed', 'real_authors_perturbed', 'world_facts_perturbed', 'forget10_perturbed']
    else:
        raise NotImplementedError
    
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset = torch_format_dataset,
        eval_dataset = torch_format_dataset,
        compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model = oracle_model,
        forget_loss = cfg.forget_loss,
        eval_cfg = cfg.eval,
        seed = seed,
        ref_policy = cfg.ref_policy,
        beta = cfg.beta,
        npo_coeff=cfg.npo_coeff,
        grad_diff_coeff=cfg.grad_diff_coeff,
        KL_coeff=cfg.KL_coeff,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    #save the tokenizer
    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()
