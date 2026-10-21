import torch

from transformers import AutoModel, AutoFeatureExtractor

from nemo.collections.asr.models import EncDecSpeakerLabelModel

class JudgeModel:
    """
    Judge Model class for Generation-based SALMon evaluation:
    The judge model is used to compute embedding for the postivie,
    the negative and the generated continuations for a given speech
    prompt and to calculate the SALMon score based on whether the 
    generated continuation is closer to the positive (1) or the 
    negative (0) continuation.

    The following judge models are used for each SALMon consistency task:
    background (domain): HuBERT-large-audioset
    background (random/all): HuBERT-large audioset
    gender: TITANET-large
    room impulse response (rir): Wav2vec 2.0-large-audioset
    sentiment: TITANET-large
    speaker: TITANET-large
    """
    TRIM_SECONDS = 0.2

    def __init__(self, model_id, sr, device="cpu", max_length_seconds=5.0):
        self.sr = sr
        self._is_titanet = model_id == "nvidia/speakerverification_en_titanet_large"
        self.device = device
        self._max_length_samples = int(round(max_length_seconds * sr))
        if self._is_titanet:
            self._extractor = None
            self._model = EncDecSpeakerLabelModel.from_pretrained(model_id).to(device).eval()
        else:
            self._extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self._model = AutoModel.from_pretrained(model_id).to(device).eval()

    def trim_continuation(self, wav, is_generated=False, prompt_len=0):
        trim_samples = int(round(self.TRIM_SECONDS * self.sr))
        if is_generated:
            start = min(prompt_len + trim_samples, wav.shape[0])
        else:
            start = min(trim_samples, wav.shape[0])
        return wav[start:]

    @torch.no_grad()
    def embed_batch(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        if self._is_titanet:
            lengths = [w.shape[0] for w in waveforms]
            max_len = max(lengths)
            batch = torch.zeros(len(waveforms), max_len, device=self.device)
            for i, w in enumerate(waveforms):
                batch[i, :lengths[i]] = w.to(self.device)
            lens = torch.tensor(lengths, device=self.device)
            _, embs = self._model.forward(input_signal=batch, input_signal_length=lens)
            return embs.squeeze(1)
        else:
            arrays = [w.cpu().numpy() if isinstance(w, torch.Tensor) else w for w in waveforms]
            inputs = self._extractor(
                arrays,
                sampling_rate=self.sr,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self._max_length_samples,
            ).to(self.device)
            hidden = self._model(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            if mask.shape[1] != hidden.shape[1]:
                ratio = mask.shape[1] / hidden.shape[1]
                lens = (mask.sum(dim=1).float() / ratio).long().clamp(min=1)
                mask = torch.zeros(hidden.shape[:2], device=self.device)
                for i, l in enumerate(lens):
                    mask[i, :l] = 1.0
            mask = mask.unsqueeze(-1)
            embs = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            return embs

    def score(self, gen_emb, pos_emb, neg_emb):
        sim_pos = torch.nn.functional.cosine_similarity(gen_emb, pos_emb, dim=-1)
        sim_neg = torch.nn.functional.cosine_similarity(gen_emb, neg_emb, dim=-1)
        return (sim_pos > sim_neg).float().mean().item()