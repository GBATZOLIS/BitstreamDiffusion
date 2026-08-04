"""Exact-length generation for the 20-way categorical diffusion baseline."""
from __future__ import annotations
import argparse, copy, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from data.proteins import DIMA_CANONICAL_AA
from evaluation.generate_dima_bitstream import sha256
from evaluation.utils import load_checkpoint, load_config, sample_text_sequences_for_external, unwrap_all
from models import create_model
from utils.ema import EMA


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--num-samples",type=int,default=2048); p.add_argument("--micro-batch-size",type=int,default=128)
    p.add_argument("--num-steps",type=int,default=250); p.add_argument("--terminal-sigma",type=float,default=0.08)
    p.add_argument("--sampler",default="heun_karras"); p.add_argument("--seed",type=int,default=0); p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); cfg=load_config(a.config)
    if str(cfg.data.representation).lower()!="tokens": raise ValueError("Expected token representation")
    cfg.train.generation.num_sampling_steps=a.num_steps; cfg.train.generation.terminal_sigmas=[a.terminal_sigma]
    cfg.train.generation.entropic_blend_alpha=0.0; cfg.train.generation.entropy_ckpt_path=None
    torch.manual_seed(a.seed); np.random.seed(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=create_model(cfg).to(device); ema=EMA(unwrap_all(model),decay=0.0); checkpoint=Path(a.checkpoint)
    load_checkpoint(model,ema,checkpoint,device,apply_ema=True); model.eval()
    probability_path=Path(cfg.evaluation.length_distribution); probabilities=np.load(probability_path); probabilities/=probabilities.sum()
    lengths=np.random.default_rng(a.seed).choice(np.arange(len(probabilities)),size=a.num_samples,p=probabilities).astype(np.int64)
    sequences=[None]*a.num_samples
    for length in sorted(np.unique(lengths).tolist()):
        positions=np.flatnonzero(lengths==length); local=copy.deepcopy(cfg); local.data.sequence_len_tokens=length; local.data.sequence_len=length
        for start in range(0,len(positions),a.micro_batch_size):
            selected=positions[start:start+a.micro_batch_size]
            ids,_,_=sample_text_sequences_for_external(cfg=local,model=model,device=device,num_samples=len(selected),sampler_name=a.sampler,return_dict=False,decode_strategy="argmax_tokens",warmup=False,ddp=False)
            for index,row in zip(selected.tolist(),ids.cpu().tolist()): sequences[index]="".join(DIMA_CANONICAL_AA[token] for token in row)
    a.out_dir.mkdir(parents=True,exist_ok=True); fasta=a.out_dir/"generated.fasta"
    with fasta.open("w") as f:
        for i,s in enumerate(sequences): f.write(f">gen_{i} length={len(s)}\n{s}\n")
    np.save(a.out_dir/"sampled_lengths.npy",lengths)
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"config":{"path":a.config,"sha256":sha256(Path(a.config))},"checkpoint":{"path":str(checkpoint),"sha256":sha256(checkpoint)},"generation":vars(a)|{"out_dir":str(a.out_dir)},"fasta_sha256":sha256(fasta)}
    with (a.out_dir/"manifest.json").open("w") as f: json.dump(manifest,f,indent=2,sort_keys=True,default=str)
    print(f"Saved {len(sequences)} categorical samples to {fasta}")
if __name__=="__main__": main()
