import pathlib
import importlib.util as ilu, sys, os, time, json, torch
REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
os.chdir(REPO)
from models import create_model
cfgp, tag, outp = sys.argv[1], sys.argv[2], sys.argv[3]
spec=ilu.spec_from_file_location("c",cfgp); m=ilu.module_from_spec(spec); spec.loader.exec_module(m); cfg=m.get_config()
dev=torch.device("cuda"); S=int(cfg.data.sequence_len)
model=create_model(cfg).to(dev); nparam=sum(p.numel() for p in model.parameters())
model=torch.compile(model, fullgraph=False)
opt=torch.optim.AdamW(model.parameters(), lr=1e-4)
B=256
x=torch.randint(0,2,(B,S),device=dev).float(); sig=torch.full((B,),1.0,device=dev); sc=torch.zeros_like(x)
def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        y=model(x+0.1*torch.randn_like(x), sig, sc)
        y=y[0] if isinstance(y,(tuple,list)) else y
        loss=(y.float()**2).mean()
    loss.backward(); opt.step()
for _ in range(6): step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0=time.time(); REP=25
for _ in range(REP): step()
torch.cuda.synchronize()
ms=1000*(time.time()-t0)/REP
r={"arm":tag,"positions":int(cfg.data.num_positions),"bits":S,"patch":int(cfg.model.patch_size),
   "params_M":round(nparam/1e6,1),"ms_per_step_B256":round(ms,1),
   "peak_GiB_B256":round(torch.cuda.max_memory_allocated()/2**30,2)}
print(json.dumps(r)); json.dump(r, open(outp,"w"))
