"""Four-GPU smoke for the padding-free DiMA VLB callback."""
from configs.proteins.dima35m_bitstream_smoke import get_config as _smoke


def get_config():
    cfg = _smoke()
    cfg.experiment = "proteins/dima35m_bitstream_vlb_ddp_smoke"
    cfg.train.vlb.enabled = True
    cfg.train.vlb.every_k_epochs = 1
    cfg.train.vlb.batch_size = 4
    cfg.train.vlb.max_batches_val = 2
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    return cfg
