"""Controlled QLoRA SFT: normal-correct versus shortest-correct, full length 6400."""
import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/datasets/math/cot_teacher_sft_no_think_max6400_v4'
TARGETS = ['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj','gate_proj','up_proj','down_proj']


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def template(tokenizer,messages,generation):
    value=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation,
                                        enable_thinking=False,truncation=False,return_dict=False)
    return list(value['input_ids'] if isinstance(value,Mapping) else value)


def encode_row(tokenizer,row,max_length=6400):
    m=row['messages']
    if [r['role'] for r in m]!=['system','user','assistant']:
        raise ValueError('Expected system/user/assistant only')
    if re.search(r'</?think\b',m[-1]['content'],re.I) or not m[-1]['content'].strip():
        raise ValueError('Empty or thinking-marked assistant answer')
    full=template(tokenizer,m,False)
    prefix=template(tokenizer,m[:-1],True)
    if len(full)>max_length:
        raise ValueError(f"{row['question_id']} full length {len(full)} exceeds {max_length}; no truncation")
    if full[:len(prefix)]!=prefix or len(full)<=len(prefix):
        raise ValueError('Unsafe assistant loss boundary')
    labels=[-100]*len(prefix)+full[len(prefix):]
    return {'input_ids':full,'attention_mask':[1]*len(full),'labels':labels}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--variant',choices=['normal','shortest'],required=True)
    p.add_argument('--data-dir',type=Path,default=DATA)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true',help='Tokenize/check all pairs; no model weights or training')
    p.add_argument('--smoke',action='store_true',help='Two updates on the same 16 longest paired questions')
    p.add_argument('--resume',action='store_true',help='Resume latest optimizer checkpoint, never silently restart')
    p.add_argument('--epochs',type=float,default=2)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    names={'normal':'normal_correct_sft','shortest':'shortest_correct_sft'}
    rows={k:[json.loads(s) for s in (args.data_dir/f'{v}.jsonl').read_text(encoding='utf-8').splitlines()] for k,v in names.items()}
    assert [r['question_id'] for r in rows['normal']]==[r['question_id'] for r in rows['shortest']]
    assert len(rows['normal'])==len({r['question_id'] for r in rows['normal']})==1940
    validation=json.loads((args.data_dir/'validation.json').read_text(encoding='utf-8'))['examples']
    assert not {r['question_id'] for r in rows['normal']} & {r['id'] for r in validation}
    assert all(a['messages'][:2]==b['messages'][:2] for a,b in zip(rows['normal'],rows['shortest']))
    tokenized={}
    for k,rs in rows.items():
        tokenized[k]=[encode_row(tokenizer,r) for r in rs]
        print(f'{k}: {len(rs)} rows; max_length={max(len(r["input_ids"]) for r in tokenized[k])}; masks OK',flush=True)
    indices=list(range(len(rows['normal'])))
    if args.smoke:
        indices=sorted(indices,key=lambda i:(-max(len(tokenized[k][i]['input_ids']) for k in names),rows['normal'][i]['question_id']))[:16]
    config={'variant':args.variant,'model':str(Path(args.model).resolve()),'max_length':6400,
            'epochs':args.epochs,'lr':args.lr,'seed':args.seed,'batch_size':1,'gradient_accumulation_steps':8,
            'lora_r':64,'lora_alpha':128,'lora_dropout':0.05,'target_modules':TARGETS,
            'quantization':'nf4_double','gradient_checkpointing':True,'smoke':args.smoke,
            'max_steps':2 if args.smoke else -1,'packing':False,'enable_thinking':False,
            'loss':'assistant continuation plus end marker; system/user/empty-thinking prefix masked',
            'data_hashes':{k:digest(v) for k,v in rows.items()},'validation_hash':digest(validation),
            'template_hash':digest(tokenizer.chat_template),
            'selected_ids':[rows[args.variant][i]['question_id'] for i in indices],
            'script_hash':hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()}
    if args.preflight:
        print(json.dumps({**{k:v for k,v in config.items() if k!='selected_ids'},'training_rows':len(indices)},indent=2))
        return
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this QLoRA training script')
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to a single GPU for controlled single-GPU training')
    out=args.output.resolve()
    manifest=out/'training_manifest.json'
    resume=None
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8'))['config']!=config:
            raise ValueError('Resume requires identical data and training configuration')
        resume=get_last_checkpoint(str(out))
        if not resume:
            raise ValueError('No optimizer checkpoint to resume')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('Output exists; use --resume or choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    bf16=torch.cuda.is_bf16_supported()
    dtype=torch.bfloat16 if bf16 else torch.float16
    set_seed(args.seed)
    manifest.write_text(json.dumps({'config':config,'device':torch.cuda.get_device_name(0),'dtype':str(dtype)},ensure_ascii=False,indent=2),encoding='utf-8')
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype),device_map={'':0})
    model.config.use_cache=False
    if hasattr(model.config,'text_config'):
        model.config.text_config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=.05,target_modules=TARGETS,bias='none',task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    training_args=TrainingArguments(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,max_steps=2 if args.smoke else -1,learning_rate=args.lr,
        bf16=bf16,fp16=not bf16,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=1 if args.smoke else 10,logging_first_step=True,save_strategy='steps',save_steps=1 if args.smoke else 50,
        save_total_limit=2,optim='adamw_torch_fused',report_to='none',seed=args.seed,data_seed=args.seed,
        remove_unused_columns=True,dataloader_num_workers=0)
    trainer=Trainer(model=model,args=training_args,train_dataset=Dataset.from_list([tokenized[args.variant][i] for i in indices]),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100,return_tensors='pt'))
    result=trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out/'final'))
    tokenizer.save_pretrained(str(out/'final'))
    trainer.save_state()
    (out/'training_result.json').write_text(json.dumps({'global_step':trainer.state.global_step,'metrics':result.metrics},indent=2),encoding='utf-8')
    print(f'Finished. Adapter and tokenizer: {out / "final"}',flush=True)


if __name__=='__main__':
    main()
