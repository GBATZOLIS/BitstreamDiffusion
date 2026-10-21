import math
import json
import os
import sys
from pathlib import Path

import numpy as np

import torch


# -----------------------------------------------------------------------------
# Joint-generation speaker evaluation
# -----------------------------------------------------------------------------

class JointSpeakerEvaluator:
    """Evaluate sampled speaker codes against their waveform realization.

    This consolidates the CAM++ extraction, BiCodec re-tokenization, and the
    token/latent diversity analyses previously kept in standalone scripts and
    ``joint.ipynb``. Models are loaded lazily and reused across sampler tags.
    """

    SPEAKER_OFFSET = 200_019
    SPEAKER_VOCAB_SIZE = 4_096
    SPEAKER_SEQ_LEN = 32

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        segment_seconds: float = 3.0,
        campp_checkpoint: str | None = None,
        bicodec_model_dir: str = 'Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec',
        campp_batch_size: int = 128,
        bicodec_batch_size: int = 8,
        pad_alignment: int = 640,
        shuffle_trials: int = 100,
        shuffle_seed: int = 42,
        _dbg_func=None,
    ):
        if sample_rate != 16_000:
            raise ValueError('CAM++ and BiCodec speaker evaluation requires 16 kHz audio.')
        for name, value in (
            ('campp_batch_size', campp_batch_size),
            ('bicodec_batch_size', bicodec_batch_size),
            ('pad_alignment', pad_alignment),
            ('shuffle_trials', shuffle_trials),
        ):
            if value < 1:
                raise ValueError(f'{name} must be positive.')
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.segment_samples = int(round(sample_rate * segment_seconds))
        self.campp_checkpoint = campp_checkpoint
        self.bicodec_model_dir = str(bicodec_model_dir)
        self.campp_batch_size = int(campp_batch_size)
        self.bicodec_batch_size = int(bicodec_batch_size)
        self.pad_alignment = int(pad_alignment)
        self.shuffle_trials = int(shuffle_trials)
        self.shuffle_seed = int(shuffle_seed)
        self._dbg_func = _dbg_func
        self._campp_session = None
        self._campp_input_name = None
        self._bicodec = None

    def _dbg(self, message: str) -> None:
        if self._dbg_func:
            self._dbg_func(message)
        else:
            print(message, flush=True)

    @staticmethod
    def _atomic_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('w', encoding='utf-8') as handle:
                json.dump(value, handle, indent=2, ensure_ascii=False)
                handle.write('\n')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_npz(path: Path, **arrays) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('wb') as handle:
                np.savez(handle, **arrays)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_torch_save(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        try:
            torch.save(value, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_campp(self):
        if self._campp_session is None:
            import onnxruntime
            from huggingface_hub import hf_hub_download

            checkpoint = self.campp_checkpoint
            if checkpoint is None:
                checkpoint = hf_hub_download(
                    repo_id='MediaTek-Research/Llama-1B-TASTE-V0',
                    filename='cosyvoice/speaker_embed.onnx',
                )
            self._dbg(f'Loading CAM++ from {checkpoint}')
            self._campp_session = onnxruntime.InferenceSession(
                str(checkpoint), providers=['CPUExecutionProvider']
            )
            self._campp_input_name = self._campp_session.get_inputs()[0].name
        return self._campp_session, self._campp_input_name

    def _ensure_bicodec(self, device):
        if self._bicodec is None:
            spark_tts_root = Path('Spark-TTS').resolve()
            if str(spark_tts_root) not in sys.path:
                sys.path.insert(0, str(spark_tts_root))
            from sparktts.models.bicodec import BiCodec

            self._dbg(f'Loading BiCodec from {self.bicodec_model_dir}')
            self._bicodec = BiCodec.load_from_checkpoint(
                model_dir=self.bicodec_model_dir
            ).to(device).eval()
        return self._bicodec

    @torch.inference_mode()
    def _extract_campp(self, wavs, wav_names, output_path: Path):
        import torchaudio.compliance.kaldi as kaldi

        session, input_name = self._ensure_campp()
        durations = np.asarray(
            [len(np.asarray(wav).reshape(-1)) / self.sample_rate for wav in wavs],
            dtype=np.float32,
        )
        valid_indices = np.flatnonzero(durations >= self.segment_seconds).astype(np.int64)
        if not len(valid_indices):
            raise ValueError(
                f'No joint waveform is at least {self.segment_seconds:g} seconds long.'
            )

        embedding_batches = []
        for start in range(0, len(valid_indices), self.campp_batch_size):
            indices = valid_indices[start:start + self.campp_batch_size]
            features = []
            for index in indices:
                waveform = torch.as_tensor(
                    np.asarray(wavs[int(index)], dtype=np.float32).reshape(-1)[
                        :self.segment_samples
                    ]
                ).unsqueeze(0)
                feature = kaldi.fbank(
                    waveform,
                    num_mel_bins=80,
                    sample_frequency=self.sample_rate,
                    dither=0.0,
                )
                feature = feature - feature.mean(dim=0, keepdim=True)
                features.append(feature.cpu().numpy().astype(np.float32))
            embeddings = session.run(
                None, {input_name: np.stack(features, axis=0)}
            )[0].reshape(len(features), -1)
            embeddings = embeddings / np.maximum(
                np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
            )
            embedding_batches.append(embeddings.astype(np.float32))
            self._dbg(f'CAM++ embedded {min(start + len(indices), len(valid_indices))}/{len(valid_indices)}')

        embeddings = np.concatenate(embedding_batches, axis=0)
        valid_paths = np.asarray([str(wav_names[index]) for index in valid_indices])
        self._atomic_npz(
            output_path,
            embeddings=embeddings,
            indices=valid_indices,
            paths=valid_paths,
            durations=durations,
            segment_seconds=np.float32(self.segment_seconds),
            sample_rate=np.int32(self.sample_rate),
        )
        return embeddings, valid_indices, durations

    @torch.inference_mode()
    def _extract_bicodec_tokens(self, wavs, device) -> torch.Tensor:
        bicodec = self._ensure_bicodec(device)
        result = torch.empty(
            (len(wavs), self.SPEAKER_SEQ_LEN), dtype=torch.long
        )
        for start in range(0, len(wavs), self.bicodec_batch_size):
            stop = min(start + self.bicodec_batch_size, len(wavs))
            chunk = [np.asarray(wav, dtype=np.float32).reshape(-1) for wav in wavs[start:stop]]
            if any(not len(wav) or not np.isfinite(wav).all() for wav in chunk):
                raise ValueError(f'Invalid joint waveform in rows {start}:{stop}.')
            maximum = max(len(wav) for wav in chunk)
            padded_length = math.ceil(maximum / self.pad_alignment) * self.pad_alignment
            padded = np.stack([
                np.pad(wav, (0, padded_length - len(wav))) for wav in chunk
            ])
            waveform_batch = torch.from_numpy(padded).unsqueeze(1).to(
                device=device, dtype=torch.float32
            )
            mel = bicodec.mel_transformer(waveform_batch).squeeze(1)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == 'cuda',
            ):
                tokens = bicodec.speaker_encoder.tokenize(
                    mel.transpose(1, 2)
                ).squeeze(1)
            expected = (stop - start, self.SPEAKER_SEQ_LEN)
            if tuple(tokens.shape) != expected:
                raise ValueError(f'BiCodec returned {tuple(tokens.shape)}, expected {expected}.')
            tokens = tokens.to(device='cpu', dtype=torch.long)
            if tokens.min() < 0 or tokens.max() >= self.SPEAKER_VOCAB_SIZE:
                raise ValueError('BiCodec returned a token outside its speaker vocabulary.')
            result[start:stop] = tokens
            self._dbg(f'BiCodec re-tokenized {stop}/{len(wavs)}')
        return result

    def _save_bicodec_archive(
        self, path: Path, tokens: torch.Tensor, wav_names, task_dir: Path,
        sampler_tag: str, header: dict,
    ) -> None:
        archive = {
            'tokens': tokens,
            'global_token_ids': tokens + self.SPEAKER_OFFSET,
            'token_space': 'bicodec_speaker_code',
            'global_token_space': 'global_tokenizer_id',
            'speaker_offset': self.SPEAKER_OFFSET,
            'speaker_vocab_size': self.SPEAKER_VOCAB_SIZE,
            'speaker_seq_len': self.SPEAKER_SEQ_LEN,
            'num_samples': len(tokens),
            'sample_indices': torch.arange(len(tokens), dtype=torch.long),
            'wav_files': list(wav_names),
            'source_samples_json': str(task_dir / 'samples.json'),
            'sampler_tag': sampler_tag,
            'task': 'joint',
            'sampler': header.get('sampler'),
            'checkpoint': header.get('checkpoint'),
            'sample_rate': self.sample_rate,
            'speaker_model_dir': self.bicodec_model_dir,
            'batch_size': self.bicodec_batch_size,
            'pad_alignment': self.pad_alignment,
        }
        self._atomic_torch_save(path, archive)
        written = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.equal(written["tokens"], tokens):
            raise IOError(f"Post-write verification failed for {path}.")

    @staticmethod
    def _unit_latent_metrics(lhs, rhs, valid_positions):
        import torch.nn.functional as F

        mask = valid_positions.bool().unsqueeze(-1)
        lhs = F.normalize(
            torch.where(mask, torch.nan_to_num(lhs), torch.zeros_like(lhs)),
            p=2, dim=-1, eps=1e-12,
        )
        rhs = F.normalize(
            torch.where(mask, torch.nan_to_num(rhs), torch.zeros_like(rhs)),
            p=2, dim=-1, eps=1e-12,
        )
        squared_error = (lhs - rhs).square()
        position_rmse = squared_error.mean(dim=-1).sqrt().masked_fill(
            ~valid_positions, torch.nan
        )
        expanded_mask = mask.expand_as(squared_error)
        component_count = expanded_mask.sum(dim=(1, 2))
        sequence_rmse = (
            squared_error.masked_fill(~expanded_mask, 0.0).sum(dim=(1, 2))
            / component_count.clamp_min(1)
        ).sqrt().masked_fill(component_count == 0, torch.nan)
        position_cosine = (lhs * rhs).sum(dim=-1).masked_fill(
            ~valid_positions, torch.nan
        )
        position_count = valid_positions.sum(dim=-1)
        sequence_cosine = (
            position_cosine.nan_to_num(0.0).sum(dim=-1)
            / position_count.clamp_min(1)
        ).masked_fill(position_count == 0, torch.nan)
        return position_rmse, sequence_rmse, position_cosine, sequence_cosine

    @torch.inference_mode()
    def _tokens_to_latents(self, tokens: torch.Tensor, device):
        quantizer = self._ensure_bicodec(device).speaker_encoder.quantizer
        tokens = tokens.detach().to(device='cpu', dtype=torch.long)
        valid = (tokens >= 0) & (tokens < int(quantizer.codebook_size))
        flat_tokens = tokens.reshape(-1)
        flat_valid = valid.reshape(-1)
        indices = flat_tokens[flat_valid].to(device).reshape(-1, 1, 1)
        valid_latents = quantizer.get_output_from_indices(indices).squeeze(1).cpu().float()
        latents = torch.full(
            (flat_tokens.numel(), valid_latents.shape[-1]), float('nan')
        )
        latents[flat_valid] = valid_latents
        return latents.reshape(*tokens.shape, -1), valid

    @staticmethod
    def _derangements(num_samples: int, trials: int, seed: int) -> torch.Tensor:
        if num_samples < 2:
            raise ValueError('Joint speaker metrics require at least two samples.')
        generator = torch.Generator().manual_seed(seed)
        identity = torch.arange(num_samples)
        permutations = []
        for _ in range(trials):
            while True:
                candidate = torch.randperm(num_samples, generator=generator)
                if not candidate.eq(identity).any():
                    permutations.append(candidate)
                    break
        return torch.stack(permutations)

    @staticmethod
    def _token_diversity(tokens: torch.Tensor, vocab_size: int) -> dict:
        valid = (tokens >= 0) & (tokens < vocab_size)
        entropies = torch.full((tokens.shape[1],), torch.nan, dtype=torch.float64)
        valid_per_position = valid.sum(dim=0)
        for position in range(tokens.shape[1]):
            values = tokens[valid[:, position], position]
            if values.numel():
                counts = torch.bincount(values, minlength=vocab_size).double()
                probabilities = counts[counts > 0] / values.numel()
                entropies[position] = -(probabilities * probabilities.log2()).sum()

        distances = []
        for left in range(tokens.shape[0] - 1):
            jointly_valid = valid[left] & valid[left + 1:]
            comparable = jointly_valid.sum(dim=1)
            usable = comparable > 0
            mismatches = (
                tokens[left].ne(tokens[left + 1:]) & jointly_valid
            ).sum(dim=1)
            distances.append(mismatches[usable].double() / comparable[usable])
        pairwise = torch.cat(distances)

        fully_valid = valid.all(dim=1)
        valid_sequences = tokens[fully_valid]
        _, counts = torch.unique(valid_sequences, dim=0, return_counts=True)
        duplicate_counts = counts[counts > 1]
        return {
            'num_samples': int(tokens.shape[0]),
            'num_positions': int(tokens.shape[1]),
            'invalid_token_positions_excluded': int((~valid).sum()),
            'mean_per_position_entropy_bits': float(torch.nanmean(entropies)),
            'mean_pairwise_normalized_hamming_distance': float(pairwise.mean()),
            'number_of_unordered_pairs': int(pairwise.numel()),
            'fully_valid_sequences': int(fully_valid.sum()),
            'excluded_invalid_sequences': int((~fully_valid).sum()),
            'unique_sequences': int(counts.numel()),
            'duplicate_sequence_groups': int(duplicate_counts.numel()),
            'samples_in_repeated_groups': int(duplicate_counts.sum()),
            'repeated_copies_beyond_first': int((duplicate_counts - 1).sum()),
            'identical_unordered_pairs': int(
                (duplicate_counts * (duplicate_counts - 1) // 2).sum()
            ),
            'maximum_sequence_multiplicity': int(counts.max()) if counts.numel() else 0,
            'per_position_entropy_bits': entropies.tolist(),
            'valid_samples_per_position': valid_per_position.tolist(),
        }

    @staticmethod
    def _latent_diversity(latents: torch.Tensor, valid: torch.Tensor) -> tuple[dict, dict]:
        import torch.nn.functional as F

        retained = valid.all(dim=1) & torch.isfinite(latents).all(dim=(1, 2))
        values = latents[retained].double()
        if len(values) < 2:
            raise ValueError('At least two fully valid sampled speaker sequences are required.')
        position_unit = F.normalize(values, p=2, dim=-1, eps=1e-12)
        sequence_unit = F.normalize(position_unit.flatten(start_dim=1), p=2, dim=1)
        pair_indices = torch.triu_indices(len(sequence_unit), len(sequence_unit), offset=1)
        similarities = (sequence_unit @ sequence_unit.T)[
            pair_indices[0], pair_indices[1]
        ].clamp(-1.0, 1.0)
        distances = 1.0 - similarities
        euclidean = (2.0 * distances).clamp_min(0.0).sqrt()
        rmse = euclidean / sequence_unit.shape[1] ** 0.5
        centered = sequence_unit - sequence_unit.mean(dim=0)
        component_variance = centered.square().mean(dim=0)
        gram = centered @ centered.T / len(sequence_unit)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        proportions = eigenvalues / eigenvalues.sum()
        nonzero = proportions > 0
        effective_rank = torch.exp(
            -(proportions[nonzero] * proportions[nonzero].log()).sum()
        )
        flattened = {
            'source_samples': int(latents.shape[0]),
            'valid_sequences': int(retained.sum()),
            'excluded_invalid_sequences': int((~retained).sum()),
            'positions_per_sequence': int(latents.shape[1]),
            'latent_dimension_per_position': int(latents.shape[2]),
            'flattened_sequence_dimension': int(sequence_unit.shape[1]),
            'unordered_pairs': int(similarities.numel()),
            'mean_pairwise_unit_vector_rmse': float(rmse.mean()),
            'mean_pairwise_cosine_similarity': float(similarities.mean()),
            'mean_pairwise_cosine_distance': float(distances.mean()),
            'mean_pairwise_euclidean_distance': float(euclidean.mean()),
            'total_component_variance': float(component_variance.sum()),
            'mean_component_variance': float(component_variance.mean()),
            'covariance_effective_rank': float(effective_rank),
        }

        aligned_unit = F.normalize(values.float(), p=2, dim=-1, eps=1e-12)
        pair_count = len(aligned_unit) * (len(aligned_unit) - 1) // 2
        vector_sum = aligned_unit.sum(dim=0)
        cosine_by_position = (
            vector_sum.square().sum(dim=-1)
            - aligned_unit.square().sum(dim=-1).sum(dim=0)
        ) / (2.0 * pair_count)
        position_aligned = {
            'num_sequences': int(len(position_unit)),
            'sequence_length': int(position_unit.shape[1]),
            'latent_dim': int(position_unit.shape[2]),
            'number_of_unordered_pairs': int(pair_count),
            'mean_pairwise_cosine_similarity': float(cosine_by_position.mean()),
            'mean_pairwise_cosine_distance': float((1.0 - cosine_by_position).mean()),
            'mean_cosine_similarity_by_position': cosine_by_position.tolist(),
            'mean_cosine_distance_by_position': (1.0 - cosine_by_position).tolist(),
        }
        return flattened, position_aligned

    @staticmethod
    def _campp_diversity(embeddings: np.ndarray, source_count: int) -> dict:
        import torch.nn.functional as F

        values = torch.from_numpy(np.asarray(embeddings)).double()
        valid = torch.isfinite(values).all(dim=1) & (values.norm(dim=1) > 0)
        unit = F.normalize(values[valid], p=2, dim=1, eps=1e-12)
        if len(unit) < 2:
            raise ValueError('At least two valid CAM++ embeddings are required.')
        indices = torch.triu_indices(len(unit), len(unit), offset=1)
        similarities = (unit @ unit.T)[indices[0], indices[1]].clamp(-1.0, 1.0)
        distances = 1.0 - similarities
        euclidean = (2.0 * distances).clamp_min(0.0).sqrt()
        rmse = euclidean / unit.shape[1] ** 0.5
        centered = unit - unit.mean(dim=0)
        component_variance = centered.square().mean(dim=0)
        covariance = centered.T @ centered / len(unit)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        proportions = eigenvalues / eigenvalues.sum()
        nonzero = proportions > 0
        effective_rank = torch.exp(
            -(proportions[nonzero] * proportions[nonzero].log()).sum()
        )
        return {
            'source_samples': int(source_count),
            'excluded_by_extractor': int(source_count - len(values)),
            'archive_embeddings': int(len(values)),
            'valid_embeddings': int(valid.sum()),
            'invalid_embeddings': int((~valid).sum()),
            'embedding_dimension': int(values.shape[1]),
            'unordered_pairs': int(similarities.numel()),
            'mean_pairwise_unit_vector_rmse': float(rmse.mean()),
            'mean_pairwise_cosine_similarity': float(similarities.mean()),
            'mean_pairwise_cosine_distance': float(distances.mean()),
            'mean_pairwise_euclidean_distance': float(euclidean.mean()),
            'total_component_variance': float(component_variance.sum()),
            'mean_component_variance': float(component_variance.mean()),
            'covariance_effective_rank': float(effective_rank),
        }

    @staticmethod
    def _flatten_scalars(report: dict) -> dict[str, float]:
        flattened = {}
        for section, values in report.items():
            if not isinstance(values, dict):
                continue
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    flattened[f'Speaker/{section}/{name}'] = float(value)
        return flattened

    @torch.inference_mode()
    def evaluate(
        self,
        *,
        task_dir: Path,
        samples: list[dict],
        wavs,
        device,
        sampler_tag: str,
        manifest_header: dict,
    ) -> dict[str, float]:
        task_dir = Path(task_dir)
        if len(samples) != len(wavs) or not samples:
            raise ValueError('Joint samples and waveforms must be non-empty and row-aligned.')
        if any(wav is None for wav in wavs):
            raise ValueError('Every joint sample must have a generated waveform.')
        wav_names = [sample.get('gen_wav') for sample in samples]
        if any(not isinstance(name, str) or not name for name in wav_names):
            raise ValueError('Every joint samples.json row must contain gen_wav.')
        if len(set(wav_names)) != len(wav_names):
            raise ValueError('Joint samples.json contains duplicate gen_wav entries.')

        sampled_path = task_dir / 'generated_tokens.pt'
        if not sampled_path.is_file():
            raise FileNotFoundError(
                f'{sampled_path} is required for --speaker; generate with '
                'save_generated_tokens enabled.'
            )
        sampled_archive = torch.load(sampled_path, map_location='cpu', weights_only=False)
        sampled_all = sampled_archive['tokens'].long()
        if len(sampled_all) != len(samples) or int(sampled_archive['num_samples']) != len(samples):
            raise ValueError('generated_tokens.pt is not row-aligned with samples.json.')
        speaker_start, speaker_stop = sampled_archive['segments']['speaker']
        sampled_tokens = (
            sampled_all[:, int(speaker_start):int(speaker_stop)]
            - int(sampled_archive['speaker_offset'])
        ).long()
        vocab_size = int(sampled_archive['speaker_vocab_size'])
        if sampled_tokens.shape != (len(samples), self.SPEAKER_SEQ_LEN):
            raise ValueError(
                f'Expected sampled speaker tokens {(len(samples), self.SPEAKER_SEQ_LEN)}, '
                f'found {tuple(sampled_tokens.shape)}.'
            )
        if int(sampled_archive["speaker_offset"]) != self.SPEAKER_OFFSET:
            raise ValueError(
                f"Expected speaker offset {self.SPEAKER_OFFSET}, found "
                f"{sampled_archive['speaker_offset']}."
            )
        if vocab_size != self.SPEAKER_VOCAB_SIZE:
            raise ValueError(
                f'Expected BiCodec speaker vocabulary {self.SPEAKER_VOCAB_SIZE}, found {vocab_size}.'
            )

        campp_paths = [str(task_dir / name) for name in wav_names]
        campp_embeddings, campp_indices, durations = self._extract_campp(
            wavs, campp_paths, task_dir / 'sv_embeddings.npz'
        )
        realized_tokens = self._extract_bicodec_tokens(wavs, device)
        self._save_bicodec_archive(
            task_dir / 'bicodec_global_tokens.pt', realized_tokens, wav_names,
            task_dir, sampler_tag, manifest_header,
        )

        sampled_valid = (sampled_tokens >= 0) & (sampled_tokens < vocab_size)
        realized_valid = (realized_tokens >= 0) & (realized_tokens < vocab_size)
        matches = sampled_tokens.eq(realized_tokens)
        exact = matches.all(dim=1)
        token_agreement = {
            'mean_per_sample_token_agreement': float(matches.float().mean(dim=1).mean()),
            'mean_per_position_agreement': float(matches.float().mean(dim=0).mean()),
            'exact_sequence_matches': int(exact.sum()),
            'num_samples': int(len(exact)),
            'exact_sequence_agreement': float(exact.float().mean()),
        }

        permutations = self._derangements(
            len(samples), self.shuffle_trials, self.shuffle_seed
        )
        shuffled_token_means = []
        shuffled_exact_counts = []
        for permutation in permutations:
            shuffled_matches = sampled_tokens.eq(realized_tokens[permutation])
            shuffled_token_means.append(shuffled_matches.float().mean())
            shuffled_exact_counts.append(shuffled_matches.all(dim=1).sum())
        shuffled_token_means = torch.stack(shuffled_token_means)
        shuffled_exact_counts = torch.stack(shuffled_exact_counts).float()
        shuffled_agreement = {
            'num_shuffle_trials': self.shuffle_trials,
            'mean_per_sample_token_agreement': float(shuffled_token_means.mean()),
            'mean_per_position_agreement': float(shuffled_token_means.mean()),
            'token_agreement_std_across_trials': float(
                shuffled_token_means.std(unbiased=self.shuffle_trials > 1)
            ),
            'mean_exact_sequence_matches_per_trial': float(shuffled_exact_counts.mean()),
            'exact_sequence_agreement': float(
                shuffled_exact_counts.mean() / len(samples)
            ),
            'fixed_points_across_all_trials': int(
                permutations.eq(torch.arange(len(samples))).sum()
            ),
        }

        sampled_latents, sampled_latent_valid = self._tokens_to_latents(
            sampled_tokens, device
        )
        realized_latents, realized_latent_valid = self._tokens_to_latents(
            realized_tokens, device
        )
        paired_valid = sampled_latent_valid & realized_latent_valid
        paired = self._unit_latent_metrics(
            sampled_latents, realized_latents, paired_valid
        )
        shuffled_position_rmse = []
        shuffled_sequence_rmse = []
        shuffled_position_cosine = []
        shuffled_sequence_cosine = []
        shuffled_valid_counts = []
        for permutation in permutations:
            valid = sampled_latent_valid & realized_latent_valid[permutation]
            values = self._unit_latent_metrics(
                sampled_latents, realized_latents[permutation], valid
            )
            shuffled_position_rmse.append(torch.nanmean(values[0]))
            shuffled_sequence_rmse.append(torch.nanmean(values[1]))
            shuffled_position_cosine.append(torch.nanmean(values[2]))
            shuffled_sequence_cosine.append(torch.nanmean(values[3]))
            shuffled_valid_counts.append(valid.sum())
        shuffled_position_rmse = torch.stack(shuffled_position_rmse)
        shuffled_sequence_rmse = torch.stack(shuffled_sequence_rmse)
        shuffled_position_cosine = torch.stack(shuffled_position_cosine)
        shuffled_sequence_cosine = torch.stack(shuffled_sequence_cosine)
        shuffled_valid_counts = torch.stack(shuffled_valid_counts).float()
        paired_position_rmse = float(torch.nanmean(paired[0]))
        paired_sequence_rmse = float(torch.nanmean(paired[1]))
        shuffled_position_rmse_mean = float(shuffled_position_rmse.mean())
        shuffled_sequence_rmse_mean = float(shuffled_sequence_rmse.mean())
        latent_similarity = {
            'paired_position_unit_vector_rmse': paired_position_rmse,
            'shuffled_position_unit_vector_rmse': shuffled_position_rmse_mean,
            'shuffled_position_unit_vector_rmse_std': float(
                shuffled_position_rmse.std(unbiased=self.shuffle_trials > 1)
            ),
            'position_rmse_relative_reduction': (
                1.0 - paired_position_rmse / shuffled_position_rmse_mean
                if shuffled_position_rmse_mean else float('nan')
            ),
            'paired_sequence_unit_vector_rmse': paired_sequence_rmse,
            'shuffled_sequence_unit_vector_rmse': shuffled_sequence_rmse_mean,
            'shuffled_sequence_unit_vector_rmse_std': float(
                shuffled_sequence_rmse.std(unbiased=self.shuffle_trials > 1)
            ),
            'sequence_rmse_relative_reduction': (
                1.0 - paired_sequence_rmse / shuffled_sequence_rmse_mean
                if shuffled_sequence_rmse_mean else float('nan')
            ),
            'paired_position_cosine_similarity': float(torch.nanmean(paired[2])),
            'shuffled_position_cosine_similarity': float(shuffled_position_cosine.mean()),
            'shuffled_position_cosine_similarity_std': float(
                shuffled_position_cosine.std(unbiased=self.shuffle_trials > 1)
            ),
            'paired_sequence_mean_cosine_similarity': float(torch.nanmean(paired[3])),
            'shuffled_sequence_mean_cosine_similarity': float(shuffled_sequence_cosine.mean()),
            'shuffled_sequence_mean_cosine_similarity_std': float(
                shuffled_sequence_cosine.std(unbiased=self.shuffle_trials > 1)
            ),
            'valid_paired_positions': int(paired_valid.sum()),
            'skipped_paired_positions': int((~paired_valid).sum()),
            'mean_valid_shuffled_positions': float(shuffled_valid_counts.mean()),
            'shuffle_trials': self.shuffle_trials,
        }

        flattened_diversity, aligned_diversity = self._latent_diversity(
            sampled_latents, sampled_latent_valid
        )
        report = {
            'diagnostics': {
                'num_samples': len(samples),
                'sampled_invalid_tokens': int((~sampled_valid).sum()),
                'sampled_rows_with_invalid_tokens': int((~sampled_valid).any(dim=1).sum()),
                'realized_invalid_tokens': int((~realized_valid).sum()),
                'campp_retained_embeddings': int(len(campp_indices)),
                'campp_excluded_short_samples': int(len(samples) - len(campp_indices)),
            },
            'token_agreement': token_agreement,
            'shuffled_token_agreement': shuffled_agreement,
            'latent_similarity': latent_similarity,
            'sampled_token_diversity': self._token_diversity(
                sampled_tokens, vocab_size
            ),
            'sampled_bicodec_latent_diversity': flattened_diversity,
            'sampled_bicodec_position_aligned_diversity': aligned_diversity,
            'generated_campp_diversity': self._campp_diversity(
                campp_embeddings, len(durations)
            ),
            'artifacts': {
                'sampled_tokens': str(sampled_path),
                'realized_tokens': str(task_dir / 'bicodec_global_tokens.pt'),
                'campp_embeddings': str(task_dir / 'sv_embeddings.npz'),
            },
        }
        report_path = task_dir / 'speaker_metrics.json'
        self._atomic_json(report_path, report)
        self._dbg(f'Wrote joint speaker report to {report_path}')
        return self._flatten_scalars(report)
