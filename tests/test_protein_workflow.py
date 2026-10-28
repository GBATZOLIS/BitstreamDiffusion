"""Check the public download and evaluation entry points."""

import argparse
import hashlib
import importlib
import io
from pathlib import Path

import pytest

from configs.proteins.proteins_400m import get_config
from scripts.proteins.download_config import download
from scripts.proteins.download_data import commands
from scripts.proteins.evaluate import command_for


def check_parser(command, monkeypatch):
    module = importlib.import_module(command[2])
    original = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def parse(parser, args=None, namespace=None):
        original(parser, command[3:], namespace)
        raise Parsed()

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(Parsed):
        module.main()


@pytest.mark.parametrize(
    "task", ["entropy", "inverse", "forward", "cogeneration"]
)
def test_evaluation_arguments(task, monkeypatch, tmp_path):
    command = command_for(
        task,
        get_config(),
        Path("configs/proteins/proteins_400m.py"),
        tmp_path / "checkpoints/best.pt",
        tmp_path / "report.json",
        "cpu",
    )
    check_parser(command, monkeypatch)


@pytest.mark.parametrize("component", ["paired", "replay", "cath", "cameo"])
def test_data_arguments(component, monkeypatch):
    for command in commands(component):
        with monkeypatch.context() as patch:
            check_parser(command, patch)


def test_download_checks_hash_before_install(monkeypatch, tmp_path):
    destination = tmp_path / "config.yaml"
    expected = hashlib.sha256(b"correct").hexdigest()
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        download("https://example.org/config.yaml", destination, expected)
    assert not destination.exists()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(b"correct"),
    )
    download("https://example.org/config.yaml", destination, expected)
    assert destination.read_bytes() == b"correct"


def test_download_keeps_existing_file(monkeypatch, tmp_path):
    destination = tmp_path / "config.yaml"
    destination.write_bytes(b"local config")

    def no_network(*args, **kwargs):
        pytest.fail("An existing file must be checked before any download")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    expected = hashlib.sha256(b"local config").hexdigest()
    download("https://example.org/config.yaml", destination, expected)
    with pytest.raises(ValueError, match="different hash"):
        download("https://example.org/config.yaml", destination, "0" * 64)
    assert destination.read_bytes() == b"local config"
