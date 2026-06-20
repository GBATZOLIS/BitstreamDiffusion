"""
Faithful local reproduction of the EM per-step noise/drift profile WITHOUT the
model. Drives the REAL SigmaSchedule (entropic inverse-CDF grid) and the REAL
EntropyRateLambdaProfile(normalize='as_saved') from BitstreamDiffusion, using a
synthetic entropy table reconstructed to match the EXACT stated tinygsm stats:
  K=128, sigma in [0.00209, 76.76], as_saved table peak 0.362 @ sigma~1.85,
  mean 0.094, Delta(log sigma)=0.08279, sum(pdf)=1.
Goal: resolve whether per-step injected noise std/sigma on the ACTUAL entropic
grid is ~1.6*sigma (handoff lead, assumed uniform Delta-sigma) or bounded
~0.8*sigma (entropy-grid cancellation). Also report the gamma_step cap behavior.
"""
import os, sys, math, tempfile
from types import SimpleNamespace
from pathlib import Path
import torch

REPO = "/home/gb511/BitstreamDiffusion"
sys.path.insert(0, REPO)
torch.set_grad_enabled(False)
dev = torch.device("cpu")

# ---------------------------------------------------------------- synth table
K = 128
sigma_min_edge, sigma_max_edge = 0.00209, 76.76          # midpoint range target
# Build K+1 log-spaced EDGES so that midpoints span ~[sigma_min_edge, sigma_max_edge].
# midpoints are geometric centers of edges; to land midpoints at the stated range
# we extend edges half a bin beyond on each side.
log_lo_mid, log_hi_mid = math.log(sigma_min_edge), math.log(sigma_max_edge)
dlog = (log_hi_mid - log_lo_mid) / (K - 1)               # midpoint spacing
edges_log = torch.linspace(log_lo_mid - dlog/2, log_hi_mid + dlog/2, K + 1)
edges = edges_log.exp()
mids = (0.5 * (edges_log[:-1] + edges_log[1:])).exp()
log_mids = mids.log()

# as_saved table = pi_alpha(log sigma) is a normal density in log-sigma.
mu = math.log(1.85)
s = 1.0 / (0.362 * math.sqrt(2 * math.pi))               # peak value -> s
dens = torch.exp(-(log_mids - mu) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))
# pdf = dens * Dlog, then renormalize sum->1 (keeps table=pdf/Dlog ~ dens)
Dlog = float(((edges_log[-1] - edges_log[0]) / K))
pdf = dens * Dlog
pdf = pdf / pdf.sum()
cdf = torch.cumsum(pdf, 0); cdf[-1] = 1.0
table_assaved = pdf / Dlog

print("== synthetic table sanity (target: peak 0.362@~1.85, mean .094, Dlog .08279) ==")
print(f"  K={K}  sigma_mid range [{mids[0]:.5f}, {mids[-1]:.3f}]  Dlog={Dlog:.5f}")
ip = int(table_assaved.argmax())
print(f"  table peak {table_assaved.max():.4f} @ sigma={mids[ip]:.3f}   table mean {table_assaved.mean():.4f}  sum(pdf)={pdf.sum():.4f}")

run_dir = Path(tempfile.mkdtemp(prefix="entropy_tinygsm_synth_"))
torch.save(pdf, run_dir / "entropy_pdf.pt")
torch.save(cdf, run_dir / "entropy_cdf.pt")
torch.save(mids, run_dir / "entropy_sigmas.pt")
torch.save(edges, run_dir / "entropy_edges.pt")

# ---------------------------------------------------------------- real code
from diffusion.continuous.samplers import SigmaSchedule
from diffusion.continuous.lambda_profiles import EntropyRateLambdaProfile

cfg = SimpleNamespace(
    evaluation=SimpleNamespace(schedule="entropic", num_sampling_steps=1024,
                               entropic_blend_alpha=0.0, checkpoint_path=str(run_dir / "checkpoints/x.pt")),
    diffusion=SimpleNamespace(continuous=SimpleNamespace(sigma_min=0.00209, sigma_max=76.76, rho=7.0)),
)
sched = SigmaSchedule(process=None, cfg=cfg, device=dev)

def analyze(N, lambda_zero, gamma_cap=1.0):
    sigmas = sched.prepare(schedule="entropic", num_steps=N, entropy_run_dir=run_dir)
    prof = EntropyRateLambdaProfile(lambda_zero=lambda_zero, entropy_run_dir=run_dir,
                                    device=dev, normalize="as_saved")
    sc = sigmas[:-1]; sn = sigmas[1:]
    dsig = (sc - sn).clamp_min(0.0)
    lam = prof.evaluate(sc)
    # raw (no cap)
    g_raw = lam * dsig / sc.clamp_min(1e-12)
    std_raw = (2.0 * lam * sc * dsig).clamp_min(0).sqrt()
    # capped (sampler default em_step_gamma_cap=1.0)
    lam_cap = gamma_cap * sc / dsig.clamp_min(1e-12)
    lam_c = torch.minimum(lam, lam_cap)
    g_cap = lam_c * dsig / sc.clamp_min(1e-12)
    std_cap = (2.0 * lam_c * sc * dsig).clamp_min(0).sqrt()
    drift_cap = (1.0 + lam_c) * dsig / sc.clamp_min(1e-12)   # |h(1+lam)d|/x-scale ~ (1+lam)dsig/sigma along score
    return dict(sigmas=sigmas, sc=sc, dsig=dsig, lam=lam,
                stdrel_raw=std_raw/sc, stdrel_cap=std_cap/sc,
                g_raw=g_raw, g_cap=g_cap, drift_cap=drift_cap)

for N in (1024, 256):
    print(f"\n================= N={N} entropic grid =================")
    # show uniform-log spacing for reference (what the handoff assumed)
    R = math.log(76.76/0.00209)
    print(f"  uniform-log Dlog would be {R/N:.5f}; uniform Delta-sigma @sigma=1.85 = {1.85*R/N:.4f}")
    a0 = analyze(N, 0.0)
    # report the ACTUAL grid spacing near sigma~1.85 (peak band)
    sc = a0["sc"]; idx = int((sc - 1.85).abs().argmin())
    print(f"  ACTUAL entropic Delta-sigma @sigma~1.85 = {a0['dsig'][idx]:.5f}  (grid idx {idx}/{N})")
    for LZ in (0.5, 1, 2, 5, 10, 20, 50, 340, 419):
        a = analyze(N, LZ)
        sr = a["stdrel_raw"]; src = a["stdrel_cap"]
        i_raw = int(sr.argmax()); i_cap = int(src.argmax())
        print(f"  LZ={LZ:>5} | RAW max std/sig={sr.max():.3f} @sig={a['sc'][i_raw]:.3f} (g={a['g_raw'][i_raw]:.2f}) "
              f"| CAPPED max std/sig={src.max():.3f} @sig={a['sc'][i_cap]:.3f} max_g={a['g_cap'].max():.3f} "
              f"max_drift={a['drift_cap'].max():.3f}")

print("\n\n############ BULK vs TAIL decomposition (N=1024) ############")
for LZ in (419, 340, 50, 10):
    a = analyze(1024, LZ)
    sr = a["stdrel_raw"]; sc = a["sc"]
    q = torch.quantile(sr, torch.tensor([0.5, 0.9, 0.99, 1.0]))
    # bulk = entropy mass band sigma in [0.5, 10]; tail = sigma < 0.3
    bulk = sr[(sc >= 0.5) & (sc <= 10.0)]
    tail = sr[sc < 0.3]
    print(f" LZ={LZ:>4}: std/sig  median={q[0]:.3f} p90={q[1]:.3f} p99={q[2]:.3f} max={q[3]:.3f} | "
          f"bulk[0.5,10] mean={bulk.mean():.3f} max={bulk.max():.3f} | tail<0.3 mean={tail.mean():.3f} max={tail.max():.3f}")

print("\n  last 16 steps at LZ=419 (sigma, dsig, lam, std/sig RAW, std/sig CAPPED@1.0):")
a = analyze(1024, 419)
for j in range(len(a["sc"]) - 16, len(a["sc"])):
    print(f"   step{j:4d} sig={a['sc'][j]:.5f} dsig={a['dsig'][j]:.5f} lam={a['lam'][j]:.3f} "
          f"raw={a['stdrel_raw'][j]:.3f} cap={a['stdrel_cap'][j]:.3f}")

print("\n  EDM-churn reference: gamma_step=0.34 -> std/sig =", round(((1.34**2-1)**0.5),3),
      "; gamma_step=0.41 -> std/sig =", round(((1.41**2-1)**0.5),3), "(uniform across all sigma)")

# What per-step gamma_cap keeps the WHOLE trajectory <= EDM's 0.41-equiv std/sig=0.806, at LZ=419?
print("\n  Cap sweep @ LZ=419 (N=1024): max std/sig over trajectory vs em_step_gamma_cap")
a = analyze(1024, 419)
for cap in (1.0, 0.5, 0.41, 0.325, 0.2, 0.1):
    lam = a["lam"]; sc = a["sc"]; dsig = a["dsig"]
    lam_cap = cap * sc / dsig.clamp_min(1e-12)
    lam_c = torch.minimum(lam, lam_cap)
    std = (2.0*lam_c*sc*dsig).clamp_min(0).sqrt()/sc
    bulk = std[(sc>=0.5)&(sc<=10.0)]
    print(f"    cap={cap:<5}: max std/sig={std.max():.3f}  bulk mean={bulk.mean():.3f}")
