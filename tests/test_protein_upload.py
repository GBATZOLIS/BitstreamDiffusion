"""Check protein uploads without contacting the Hub."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import upload_to_hf


@pytest.fixture
def release(tmp_path):
    """Create a small release with matching checkpoint and table hashes."""
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    hashes = {}
    for name in upload_to_hf.ENTROPY_TABLES:
        data = name.encode()
        (run / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    profile = {
        "checkpoint": {
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        },
        "results": {"entropy_tables": hashes},
    }
    (run / "entropy_profile.json").write_text(json.dumps(profile))
    return checkpoint


@pytest.fixture(autouse=True)
def no_upload(monkeypatch):
    """Fail if an upload test reaches the real upload function."""
    original = upload_to_hf.upload_manifest

    def fail(*args, **kwargs):
        pytest.fail("This test must not upload files.")

    monkeypatch.setattr(upload_to_hf, "upload_manifest", fail)
    return original


def arguments(checkpoint):
    return [
        "--model",
        "protein",
        "--repo-id",
        "example/protein",
        "--checkpoint",
        str(checkpoint),
    ]


def test_manifest_excludes_data_and_logs(release):
    run = release.parent.parent
    (run / "datasets").mkdir()
    (run / "datasets/sequences.fasta").write_text(">protein\nACDE\n")
    (run / "logs").mkdir()
    (run / "logs/training.json").write_text("{}")
    manifest = upload_to_hf.build_protein_manifest(release)
    destinations = {destination for _, destination in manifest}
    assert destinations == {
        "checkpoints/best.pt",
        "configs/proteins/proteins_400m.py",
        "entropy_profile.json",
        *upload_to_hf.ENTROPY_TABLES,
    }
    assert all(source.is_file() for source, _ in manifest)
    upload_to_hf.validate_protein_manifest(manifest)


def test_protein_dry_run_is_read_only(release, capsys):
    before = release.read_bytes()
    assert upload_to_hf.main(arguments(release) + ["--dry-run"]) == 0
    assert release.read_bytes() == before
    assert "nothing uploaded" in capsys.readouterr().out


@pytest.mark.parametrize("filename", ["checkpoints/best.pt", "entropy_cdf.pt"])
def test_changed_files_block_upload(release, filename, capsys):
    (release.parent.parent / filename).write_bytes(b"changed")
    assert upload_to_hf.main(arguments(release)) == 1
    assert capsys.readouterr().err


def test_missing_files_block_upload(release, capsys):
    (release.parent.parent / "entropy_pdf.pt").unlink()
    assert upload_to_hf.main(arguments(release)) == 1
    assert "required file(s) missing" in capsys.readouterr().err


def test_profile_cannot_add_files_to_upload(release):
    path = release.parent.parent / "entropy_profile.json"
    profile = json.loads(path.read_text())
    profile["results"]["entropy_tables"]["datasets/sequences.fasta"] = "hash"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="four tables"):
        upload_to_hf.validate_protein_manifest(
            upload_to_hf.build_protein_manifest(release)
        )


@pytest.mark.parametrize("profile", [None, {}, {"checkpoint": None}])
def test_incomplete_profile_blocks_upload(release, profile, capsys):
    path = release.parent.parent / "entropy_profile.json"
    path.write_text(json.dumps(profile))
    assert upload_to_hf.main(arguments(release)) == 1
    assert "profile is incomplete" in capsys.readouterr().err


def test_custom_entropy_directory_and_config(release, tmp_path):
    source = release.parent.parent
    entropy_dir = tmp_path / "entropy"
    entropy_dir.mkdir()
    for name in (*upload_to_hf.ENTROPY_TABLES, "entropy_profile.json"):
        (source / name).rename(entropy_dir / name)
    config = tmp_path / "paper.py"
    config.write_text("# Example config.\n")
    manifest = upload_to_hf.build_protein_manifest(
        release, entropy_dir, config
    )
    upload_to_hf.validate_protein_manifest(manifest)
    assert (config, "configs/proteins/proteins_400m.py") in manifest


def test_selected_bundle_reaches_uploader(release, monkeypatch):
    calls = []
    monkeypatch.setattr(
        upload_to_hf,
        "upload_manifest",
        lambda *args: calls.append(args),
    )
    args = arguments(release) + ["--private", "--token", "test-token"]
    assert upload_to_hf.main(args) == 0
    assert len(calls) == 1
    manifest, repo_id, token, private = calls[0]
    assert manifest == upload_to_hf.build_protein_manifest(release)
    assert (repo_id, token, private) == (
        "example/protein",
        "test-token",
        True,
    )


def test_text_manifest_still_has_original_files():
    manifest = upload_to_hf.build_manifest(Path("release"))
    assert len(manifest) == 17
    assert {destination for _, destination in manifest} >= {
        "checkpoints/cobit_s_lm1b_1M_ema.pt",
        "checkpoints/cobit_s_owt_750k_ema.pt",
        "checkpoints/cobit_m_owt_750k_ema.pt",
        "tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.json",
        "tokenizer/tokenizer_gpt2id_bpe16_65536_base1024.meta.json",
    }


def test_hub_commit_contains_only_selected_files(
    release, monkeypatch, no_upload
):
    calls = {}

    class FakeApi:
        def __init__(self, **kwargs):
            calls["auth"] = kwargs

        def create_repo(self, **kwargs):
            calls["repo"] = kwargs

        def create_commit(self, **kwargs):
            calls["commit"] = kwargs

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    manifest = upload_to_hf.build_protein_manifest(release)
    no_upload(manifest, "example/protein", None, True)
    assert calls["auth"] == {"token": None}
    assert calls["repo"] == {
        "repo_id": "example/protein",
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }
    commit = calls["commit"]
    assert commit["repo_id"] == "example/protein"
    assert commit["repo_type"] == "model"
    assert commit["num_threads"] == 1
    assert {
        (Path(operation.path_or_fileobj), operation.path_in_repo)
        for operation in commit["operations"]
    } == set(manifest)
