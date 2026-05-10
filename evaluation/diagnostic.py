import argparse
import math
import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- PROJECT IMPORTS ---
from models import create_model
from utils.ema import EMA
from data import get_loader
from evaluation.utils import load_config, unwrap_all

# -----------------------------------------------------------------------------
# Baseline Logic (CPU)
# -----------------------------------------------------------------------------

def compute_theoretical_uncontextual_metrics(sigmas: np.ndarray, num_samples: int = 1_000_000) -> Dict[str, np.ndarray]:
    """Computes BER/MSE for an optimal Uncontextual denoiser on CPU."""
    mses, bers = [], []
    x0 = np.random.randint(0, 2, size=num_samples).astype(np.float32)
    
    for sigma in sigmas:
        noise = np.random.randn(num_samples).astype(np.float32)
        xt = x0 + sigma * noise
        # Logits for P(x=1|xt) with p=0.5 prior is (xt - 0.5) / sigma^2
        logits = (xt - 0.5) / (sigma**2)
        probs = 1.0 / (1.0 + np.exp(-logits))
        
        mses.append(np.mean((probs - x0)**2))
        bers.append(np.mean(np.abs((probs > 0.5).astype(np.float32) - x0)))
        
    return {"mse": np.array(mses), "ber": np.array(bers)}

# -----------------------------------------------------------------------------
# Worker Logic (GPU)
# -----------------------------------------------------------------------------

class DenoisingWorker:
    def __init__(self, device: torch.device):
        self.device = device
        self.current_cfg_path = None
        self.model = None
        self.loader = None

    def load_model(self, cfg_path: str, batch_size_override: int):
        if self.current_cfg_path == cfg_path:
            return 

        self.model = None
        torch.cuda.empty_cache()

        print(f"[GPU {self.device.index}] Loading: {Path(cfg_path).stem}")
        self.cfg = load_config(cfg_path)
        if batch_size_override:
            self.cfg.train.batch_size = batch_size_override

        self.model = create_model(self.cfg).to(self.device)
        
        # Determine Checkpoint Path
        ckpt_path = Path(f"runs/{self.cfg.experiment}/checkpoints/last.pt")
        if not ckpt_path.exists():
            ckpt_path = Path(getattr(self.cfg.evaluation, "checkpoint_path", ""))
        
        if not ckpt_path.exists():
             raise FileNotFoundError(f"Checkpoint not found for {cfg_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ema_helper = EMA(unwrap_all(self.model), decay=0.0)
        
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema_helper.load_state_dict(ckpt["ema"])
            ema_helper.to(self.device)
            ema_helper.apply(self.model)
        else:
            unwrap_all(self.model).load_state_dict(ckpt["model"])
            
        self.model.eval()
        self.use_amp = self.cfg.evaluation.use_amp
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        
        self.loader = get_loader(self.cfg, split="test")
        self.current_cfg_path = cfg_path

    @torch.no_grad()
    def process_batch(self, sigmas: np.ndarray, num_batches: int) -> Dict[str, List[float]]:
        mses, bers = [], []
        for s_val in sigmas:
            s_tensor = torch.tensor(s_val, device=self.device, dtype=torch.float32)
            total_mse, total_ber, count = 0.0, 0.0, 0
            
            for i, batch in enumerate(self.loader):
                if num_batches > 0 and i >= num_batches: break
                
                x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
                x0 = x0.to(self.device, non_blocking=True).float()
                x0_flat = x0.view(x0.shape[0], -1)
                
                xt = x0_flat + s_tensor * torch.randn_like(x0_flat)
                sig_batch = s_tensor.expand(x0.shape[0])
                
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                    # Pass zeros for self-conditioning input to match standard denoiser call
                    try:
                        logits = self.model(xt, sig_batch, torch.zeros_like(x0_flat))
                    except TypeError:
                        logits = self.model(xt, sig_batch)
                    
                    if logits.dim() == 3: logits = logits.squeeze(-1)
                    probs = torch.sigmoid(logits.float())

                total_mse += (probs - x0_flat).pow(2).mean().item()
                total_ber += (probs.gt(0.5).float() != x0_flat).float().mean().item()
                count += 1
            
            mses.append(total_mse / count)
            bers.append(total_ber / count)
            
        return {"mse": mses, "ber": bers, "sigmas": sigmas.tolist()}

def worker_fn(rank: int, task_queue: mp.Queue, result_queue: mp.Queue, batch_size: int):
    device = torch.device(f"cuda:{rank}")
    worker = DenoisingWorker(device)
    while True:
        task = task_queue.get()
        if task is None: break
        cfg_path, sigmas, num_batches = task
        try:
            worker.load_model(cfg_path, batch_size)
            metrics = worker.process_batch(sigmas, num_batches)
            result_queue.put({
                "status": "ok", "name": Path(worker.cfg.experiment).name, "metrics": metrics
            })
        except Exception as e:
            result_queue.put({"status": "error", "error": str(e), "config": cfg_path})

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--num_batches", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="diagnostic_plots")
    args = parser.parse_args()
    
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    full_sigmas = np.logspace(np.log10(args.sigma_min), np.log10(args.sigma_max), args.num_bins)
    n_gpus = torch.cuda.device_count()
    
    # Task distribution: split sigmas into chunks to keep all GPUs busy
    sigma_chunks = np.array_split(full_sigmas, n_gpus * 2)
    tasks = [(c, chunk, args.num_batches) for c in args.configs for chunk in sigma_chunks if len(chunk) > 0]
    
    task_queue, result_queue = mp.Queue(), mp.Queue()
    for t in tasks: task_queue.put(t)
    for _ in range(n_gpus): task_queue.put(None)
    
    processes = [mp.Process(target=worker_fn, args=(i, task_queue, result_queue, args.batch_size)) for i in range(n_gpus)]
    for p in processes: p.start()
        
    results_map = {}
    for _ in range(len(tasks)):
        res = result_queue.get()
        if res["status"] == "error": continue
        name = res["name"]
        if name not in results_map: results_map[name] = {"mse": [], "ber": [], "sigmas": []}
        for k in ["mse", "ber", "sigmas"]: results_map[name][k].extend(res["metrics"][k])

    for p in processes: p.join()

    # Sort and Finalize
    final_data = {}
    for name, d in results_map.items():
        idx = np.argsort(d["sigmas"])
        final_data[name] = {"mse": np.array(d["mse"])[idx], "ber": np.array(d["ber"])[idx]}

    print("Computing theoretical baseline...")
    theo = compute_theoretical_uncontextual_metrics(full_sigmas)
    
    # 6. VERSION AGNOSTIC SAVE (NPZ)
    save_dict = {"sigmas": full_sigmas, "theo_mse": theo["mse"], "theo_ber": theo["ber"]}
    for name, res in final_data.items():
        save_dict[f"model_{name}_mse"] = res["mse"]
        save_dict[f"model_{name}_ber"] = res["ber"]
    
    np.savez(out_path / "diagnostic_results.npz", **save_dict)
    print(f"Results saved to {out_path / 'diagnostic_results.npz'}")

    # 7. Plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(final_data)))
    for m, title, scale in [("mse", "MSE", "log"), ("ber", "BER", "linear"), ("ber", "BER (Log)", "log")]:
        plt.figure(figsize=(10, 6))
        plt.plot(full_sigmas, theo[m], label="Theoretical", color='black', linestyle='--', alpha=0.6)
        for i, (name, res) in enumerate(final_data.items()):
            plt.plot(full_sigmas, res[m], label=name, color=colors[i], linewidth=2)
        plt.xscale('log'); plt.yscale(scale)
        if scale == 'log' and m == 'ber': plt.ylim(1e-4, 0.6)
        plt.title(f"Denoising {title} vs Sigma"); plt.xlabel("Sigma"); plt.ylabel(title)
        plt.legend(); plt.grid(True, which="both", alpha=0.2)
        plt.savefig(out_path / f"plot_{title.lower().replace(' ', '_')}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()