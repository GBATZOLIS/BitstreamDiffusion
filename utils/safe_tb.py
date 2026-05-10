# utils/safe_tb.py
from torch.utils.tensorboard import SummaryWriter

class SafeSummaryWriter(SummaryWriter):
    """
    SummaryWriter that will NEVER crash training on OSError/IO issues.
    After the first write failure it disables itself.
    """
    def __init__(self, *args, fail_silently: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._disabled = False
        self._fail_silently = fail_silently

    def _safe(self, fn, *a, **k):
        if self._disabled:
            return
        try:
            return fn(*a, **k)
        except Exception as e:
            msg = f"[TB] Exception while writing events ({e}). Disabling TensorBoard logging."
            if self._fail_silently:
                print(msg)
                self._disabled = True
                try:
                    super().close()
                except Exception:
                    pass
                return
            raise  # strict mode if you want it

    def add_scalar(self, *a, **k): return self._safe(super().add_scalar, *a, **k)
    def add_image(self, *a, **k):  return self._safe(super().add_image, *a, **k)
    def add_text(self, *a, **k):   return self._safe(super().add_text, *a, **k)
    def add_figure(self, *a, **k): return self._safe(super().add_figure, *a, **k)
    def add_histogram(self, *a, **k): return self._safe(super().add_histogram, *a, **k)
    def flush(self): return self._safe(super().flush)
    def close(self): return self._safe(super().close)
