"""Tests convert in-process unless they ask for the pool: a spawned
worker cannot see a test's monkeypatches, and the pool's start-up would
dwarf the tiny parts the suite converts."""
import pytest


@pytest.fixture(autouse=True)
def _in_process_by_default(monkeypatch):
    from stl_to_solid import pipeline
    monkeypatch.setattr(pipeline, 'WORKERS', 0)
