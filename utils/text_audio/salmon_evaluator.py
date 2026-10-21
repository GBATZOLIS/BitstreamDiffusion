import numpy as np

import torch

SALMON_JUDGES = [
    "nvidia/speakerverification_en_titanet_large", 
    "ALM/hubert-large-audioset",
    "ALM/wav2vec2-large-audioset"
]

SALMON_JUDGE_PER_TASK = {
    'sentiment_consistency': "nvidia/speakerverification_en_titanet_large",
    'speaker_consistency': "nvidia/speakerverification_en_titanet_large",
    'gender_consistency': "nvidia/speakerverification_en_titanet_large",
    'bg_domain_consistency': "ALM/hubert-large-audioset",
    'bg_all_consistency': "ALM/hubert-large-audioset",
    'rir_consistency': "ALM/wav2vec2-large-audioset"
}

class SALMONEvaluator:
    def __init__(
            self, 
            sr: int = 16_000, 
            max_len: float = 5.0,
            ref_dir: str = 'datasets/test_common/salmon',
            _dbg_func=None
        ):
        self._sr = sr
        self._max_len = max_len
        self._dbg_func = _dbg_func
        self.judges = None
        self.embeddings_per_task = None
        self.ref_dir = ref_dir

    def _dbg(self, msg: str):
        if self._dbg_func:
            self._dbg_func(msg)
        else:
            print(msg)

    def _ensure_judges(self, device):
        from utils.text_audio.judge_models import JudgeModel
        if not self.judges:
            self.judges = {}
            for judge_name in SALMON_JUDGES:
                self.judges[judge_name] = JudgeModel(judge_name, self._sr, device, self._max_len)
        return self.judges
    
    def _ensure_embeddings(self, device):
        if not self.embeddings_per_task:
            self.embeddings_per_task = {}
            for task in SALMON_JUDGE_PER_TASK.keys():
                embeddings_path = f'{self.ref_dir}/judge_embeddings_{task}.npz'
                embeddings = np.load(embeddings_path)
                self.embeddings_per_task[task] = {
                    'pos': torch.from_numpy(embeddings['positive']).to(device),
                    'neg': torch.from_numpy(embeddings['negative']).to(device)
                }
        return self.embeddings_per_task
    
    def evaluate_SALMON(self, gen_wavs_per_task: dict, device):
        metrics = {}

        judges = self._ensure_judges(device)
        ref_embeddings = self._ensure_embeddings(device)

        for task, judge in SALMON_JUDGE_PER_TASK.items():
            gen_embeddings = judges[judge].embed_batch(gen_wavs_per_task[task])
            metrics[task] = judges[judge].score(
                gen_embeddings, ref_embeddings[task]['pos'], ref_embeddings[task]['neg']
            )

        return metrics
