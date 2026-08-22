"""The web app reports which build is running (version + git commit)."""
import importlib
import os

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')


def test_version_endpoint_reports_build(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setenv('STL2PRISM_DATA', str(tmp_path))
    monkeypatch.setenv('STL2PRISM_BUILD_SHA', 'abc1234')
    monkeypatch.setenv('STL2PRISM_BUILD_TIME', '2026-08-18T03:00Z')
    from backend import jobs, main
    importlib.reload(jobs)
    from stl2prism import __version__
    with TestClient(main.app) as c:
        r = c.get('/api/version')
    assert r.status_code == 200
    body = r.json()
    assert body == {'version': __version__, 'commit': 'abc1234',
                    'built': '2026-08-18T03:00Z'}


def test_version_endpoint_defaults_to_unknown(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setenv('STL2PRISM_DATA', str(tmp_path))
    monkeypatch.delenv('STL2PRISM_BUILD_SHA', raising=False)
    monkeypatch.delenv('STL2PRISM_BUILD_TIME', raising=False)
    from backend import main
    with TestClient(main.app) as c:
        body = c.get('/api/version').json()
    assert body['commit'] == 'unknown' and body['built'] == 'unknown'
