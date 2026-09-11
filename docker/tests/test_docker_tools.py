"""Behavior checks for offline delivery and the non-mutating policy health probe."""

import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from websockets.sync.server import serve

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ID = "sha256:" + "a" * 64
INDEX_ID = "sha256:" + "b" * 64


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "docker" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target", [[], ["help"], ["docker-help"]])
@pytest.mark.parametrize("selection", ["environment", "command-line"])
def test_make_help_accepts_legacy_image_setting(monkeypatch, target, selection):
    arguments = []
    if selection == "environment":
        monkeypatch.setenv("DOCKER_IMAGE", "openwam:legacy")
    else:
        arguments = ["DOCKER_IMAGE=openwam:legacy"]
    # Help must not depend on Docker or a working Compose configuration.
    result = subprocess.run(
        ["make", "--no-print-directory", *target, *arguments, "DOCKER=false", "COMPOSE_FILE=/missing.yaml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "make docker-build" in result.stdout
    if target != ["docker-help"]:
        assert "make test " in result.stdout


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    """A tiny CLI stand-in lets bundle round-trips run without a Docker daemon."""
    executable = tmp_path / "docker"
    calls = tmp_path / "calls.jsonl"
    image_source = tmp_path / "image-root"
    for source in load_tool("offline").CONFIG_FILES.values():
        target = image_source / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    # An older image's configuration must win over the exporter's checkout.
    with (image_source / "compose.yaml").open("a") as stream:
        stream.write("\n# configuration frozen in the image\n")
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shutil, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['DOCKER_TEST_CALLS'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "loaded = Path(os.environ['DOCKER_TEST_CALLS'] + '.loaded')\n"
        "tagged = Path(os.environ['DOCKER_TEST_CALLS'] + '.tagged')\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    if sys.argv[-1].startswith('openwam-bundle:') and not (tagged.exists() or loaded.exists()):\n"
        "        sys.exit(1)\n"
        "    config = {'Env': ['TEST=1']} if loaded.exists() else {'Env': ['TEST=1'], 'Cmd': None}\n"
        "    config['Labels'] = {'io.openwam.bundle.schema': os.environ.get('DOCKER_TEST_SCHEMA', '3'),\n"
        "        'org.opencontainers.image.revision': os.environ.get('DOCKER_TEST_REVISION', 'image-commit')}\n"
        "    if Path(str(loaded) + '.saved').exists() and os.environ.get('DOCKER_TEST_RETAG'):\n"
        "        config['Env'] = ['RETAGGED=1']\n"
        "    layers = ['sha256:layer']\n"
        "    if loaded.exists() and os.environ.get('DOCKER_TEST_CORRUPT') == 'config':\n"
        "        config['Env'] = ['TEST=2']\n"
        "    if loaded.exists() and os.environ.get('DOCKER_TEST_CORRUPT') == 'layer':\n"
        "        layers = ['sha256:other']\n"
        "    if sys.argv[-1].startswith('openwam-bundle:') and os.environ.get('DOCKER_TEST_TAG_CONFLICT'):\n"
        "        layers = ['sha256:unrelated']\n"
        "    print(json.dumps([{'Os':'linux', 'Architecture':'amd64',\n"
        f"        'Id': {CONFIG_ID!r} if loaded.exists() else {INDEX_ID!r},\n"
        "        'RepoTags': json.loads(os.environ.get('DOCKER_TEST_TAGS', '[]')),\n"
        "        'Config': config, 'RootFS': {'Layers': layers}}]))\n"
        "elif sys.argv[1:3] == ['image', 'save']:\n"
        "    sys.stdout.buffer.write(b'test image layers')\n"
        "    Path(str(loaded) + '.saved').touch()\n"
        "elif sys.argv[1:3] == ['image', 'load']:\n"
        "    assert Path(sys.argv[-1]).is_file()\n"
        "    loaded.touch()\n"
        "elif sys.argv[1:3] == ['image', 'tag']:\n"
        f"    assert sys.argv[3] == {INDEX_ID!r}\n"
        "    tagged.write_text(sys.argv[4])\n"
        "elif sys.argv[1] == 'create':\n"
        f"    assert sys.argv[-1] == {INDEX_ID!r}\n"
        "    print('temporary-container')\n"
        "elif sys.argv[1] == 'cp':\n"
        "    source = sys.argv[2].split(':/opt/openwam/')[1]\n"
        "    shutil.copy2(Path(os.environ['DOCKER_TEST_SOURCE']) / source, sys.argv[3])\n"
        "elif sys.argv[1] == 'rm':\n"
        "    assert sys.argv[2] == 'temporary-container'\n"
        "else:\n"
        "    sys.exit(2)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("DOCKER_TEST_CALLS", str(calls))
    monkeypatch.setenv("DOCKER_TEST_SOURCE", str(image_source))
    return calls


def run_bundle(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "docker/offline.py"), *map(str, args)], capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "image,expected",
    [
        ("openwam", "openwam:latest"),
        ("openwam:revision", "openwam:revision"),
        ("example.test/team/openwam", "example.test/team/openwam:latest"),
        ("localhost:5000/team/openwam", "localhost:5000/team/openwam:latest"),
        ("localhost:5000/team/openwam:revision", "localhost:5000/team/openwam:revision"),
        ("[::1]:5000/team/openwam", "[::1]:5000/team/openwam:latest"),
        ("openwam@" + INDEX_ID, None),
        ("localhost:5000/team/openwam:release@" + INDEX_ID, None),
        (INDEX_ID, None),
        ("b" * 64, None),
        ("b" * 12, None),
    ],
)
def test_offline_roundtrip_and_image_identity(tmp_path, fake_docker, image, expected):
    destination = tmp_path / "bundle with spaces"
    result = run_bundle("export", image, destination)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((destination / "manifest.json").read_text())
    if expected is None:
        expected = manifest["image"]
        assert expected.startswith("openwam-bundle:sha256-")
        assert len(expected.rsplit("-", 1)[-1]) == 64
    assert gzip.decompress((destination / "image.tar.gz").read_bytes()) == b"test image layers"
    assert "OPENWAM_IMAGE=" + expected + "\n" in (destination / "docker/.env.example").read_text()
    assert "# configuration frozen in the image" in (destination / "compose.yaml").read_text()
    assert manifest["source_image"] == image
    assert manifest["revision"] == "image-commit"
    assert manifest["schema"] == 3
    assert manifest["image"] == expected
    # The copied importer is standalone; it does not depend on this repository.
    result = subprocess.run(
        [sys.executable, str(destination / "docker/offline.py"), "load", str(destination)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in fake_docker.read_text().splitlines()]
    assert ["image", "save", expected] in calls
    assert ["image", "load", "--input", str(destination / "image.tar.gz")] in calls
    assert calls[-1] == ["image", "inspect", expected]
    assert ["create", "--pull=never", "--network", "none", "--entrypoint", "/bin/true", INDEX_ID] in calls
    assert ["rm", "temporary-container"] in calls


def test_digest_delivery_tag_is_stable_and_never_overwrites_another_image(tmp_path, fake_docker, monkeypatch):
    references = ["openwam@" + INDEX_ID, INDEX_ID, "b" * 12]
    tags = []
    for index, reference in enumerate(references):
        destination = tmp_path / str(index)
        result = run_bundle("export", reference, destination)
        assert result.returncode == 0, result.stderr
        tags.append(json.loads((destination / "manifest.json").read_text())["image"])
    assert len(set(tags)) == 1
    calls = fake_docker.read_text()
    assert calls.count('["image", "tag",') == 1
    monkeypatch.setenv("DOCKER_TEST_TAG_CONFLICT", "1")
    destination = tmp_path / "conflict"
    result = run_bundle("export", references[0], destination)
    assert result.returncode != 0
    assert "delivery tag already points to a different image" in result.stderr
    assert not destination.exists()
    assert fake_docker.read_text().count('["image", "tag",') == 1


def test_repository_name_matching_id_prefix_still_selects_latest(tmp_path, fake_docker, monkeypatch):
    image = "b" * 12
    monkeypatch.setenv("DOCKER_TEST_TAGS", json.dumps([image + ":latest", image + ":other"]))
    destination = tmp_path / "bundle"
    result = run_bundle("export", image, destination)
    assert result.returncode == 0, result.stderr
    assert json.loads((destination / "manifest.json").read_text())["image"] == image + ":latest"
    calls = [json.loads(line) for line in fake_docker.read_text().splitlines()]
    assert ["image", "save", image + ":latest"] in calls


def test_schema_two_image_keeps_legacy_bundle_paths(tmp_path, fake_docker, monkeypatch):
    source = tmp_path / "legacy-image"
    current_sources = {
        "compose.yaml": "compose.yaml",
        "compose.host.yaml": "docker/compose.host.yaml",
        "compose.dev.yaml": "docker/compose.dev.yaml",
        ".env.example": "docker/.env.example",
        "docker.md": "docker/README.md",
        "docker/offline.py": "docker/offline.py",
    }
    for name, legacy_path in load_tool("offline").LEGACY_CONFIG_FILES.items():
        target = source / legacy_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / current_sources[name]).read_text())
    # Model the old image's self-contained instructions and overlay selection.
    (source / "assets/openwam_usage_docs/docker.md").write_text("Legacy image instructions\n")
    (source / "docker/.env.example").write_text(
        "OPENWAM_IMAGE=openwam:old\nCOMPOSE_FILE=compose.yaml:compose.host.yaml\n"
    )
    monkeypatch.setenv("DOCKER_TEST_SOURCE", str(source))
    monkeypatch.setenv("DOCKER_TEST_SCHEMA", "2")
    destination = tmp_path / "legacy-bundle"
    result = run_bundle("export", "openwam:old", destination)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["schema"] == 2
    assert set(manifest["sha256"]) == {"image.tar.gz", *current_sources}
    assert (destination / "docker.md").read_text() == "Legacy image instructions\n"
    assert "COMPOSE_FILE=compose.yaml:compose.host.yaml" in (destination / ".env.example").read_text()
    assert not (destination / "docker/README.md").exists()
    result = run_bundle("load", destination)
    assert result.returncode == 0, result.stderr
    # New importers understand schema 2 without converting or modifying it.
    manifest["schema"] = 3
    (destination / "manifest.json").write_text(json.dumps(manifest))
    assert run_bundle("verify", destination).returncode != 0


@pytest.mark.parametrize("filename", list(load_tool("offline").FILES))
def test_corrupt_bundle_never_reaches_docker_load(tmp_path, fake_docker, filename):
    destination = tmp_path / "bundle"
    assert run_bundle("export", "openwam:test", destination).returncode == 0
    (destination / filename).write_bytes(b"damaged during transfer")
    result = run_bundle("load", destination)
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert '"load"' not in fake_docker.read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        (None, []),
        ("schema", []),
        ("schema", {}),
        ("schema", "3"),
        ("schema", 3.0),
        ("sha256", None),
        ("image_identity", {"config": []}),
        ("image_identity", {"config": {"Labels": ["invalid"]}}),
    ],
)
def test_malformed_manifest_reports_error_before_loading(tmp_path, fake_docker, field, value):
    destination = tmp_path / "bundle"
    assert run_bundle("export", "openwam:test", destination).returncode == 0
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    if field is None:
        manifest = value
    else:
        manifest[field] = value
    path.write_text(json.dumps(manifest))
    result = run_bundle("load", destination)
    assert result.returncode != 0
    assert "OpenWAM offline bundle:" in result.stderr
    assert "Traceback" not in result.stderr
    assert '"load"' not in fake_docker.read_text()


def test_export_refuses_to_overwrite_existing_directory(tmp_path, fake_docker):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_text("original")
    assert run_bundle("export", "openwam:test", destination).returncode != 0
    assert sentinel.read_text() == "original"
    assert not fake_docker.exists()


@pytest.mark.parametrize("variable,value", [("DOCKER_TEST_SCHEMA", "1"), ("DOCKER_TEST_REVISION", "unknown")])
def test_export_requires_versioned_image(tmp_path, fake_docker, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    destination = tmp_path / "bundle"
    result = run_bundle("export", "openwam:test", destination)
    assert result.returncode != 0
    assert not destination.exists()
    assert '"create"' not in fake_docker.read_text()


def test_missing_image_configuration_cleans_up_container(tmp_path, fake_docker, monkeypatch):
    source = tmp_path / "empty-image"
    source.mkdir()
    monkeypatch.setenv("DOCKER_TEST_SOURCE", str(source))
    destination = tmp_path / "bundle"
    assert run_bundle("export", "openwam:test", destination).returncode != 0
    assert not destination.exists()
    assert '["rm", "temporary-container"]' in fake_docker.read_text()


def test_retag_during_export_does_not_publish_manifest(tmp_path, fake_docker, monkeypatch):
    monkeypatch.setenv("DOCKER_TEST_RETAG", "1")
    destination = tmp_path / "bundle"
    result = run_bundle("export", "openwam:test", destination)
    assert result.returncode != 0
    assert "image tag changed" in result.stderr
    assert not (destination / "manifest.json").exists()


def test_bundle_revision_must_match_image(tmp_path, fake_docker):
    destination = tmp_path / "bundle"
    assert run_bundle("export", "openwam:test", destination).returncode == 0
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["revision"] = "another-commit"
    path.write_text(json.dumps(manifest))
    result = run_bundle("load", destination)
    assert result.returncode != 0
    assert "bundle revision does not match" in result.stderr
    assert '"load"' not in fake_docker.read_text()


@pytest.mark.parametrize("corrupt", ["config", "layer"])
def test_loaded_image_must_match_layers_and_runtime_config(tmp_path, fake_docker, monkeypatch, corrupt):
    destination = tmp_path / "bundle"
    assert run_bundle("export", "openwam:test", destination).returncode == 0
    monkeypatch.setenv("DOCKER_TEST_CORRUPT", corrupt)
    result = run_bundle("load", destination)
    assert result.returncode != 0
    assert "loaded image layers/config do not match" in result.stderr


@pytest.mark.parametrize("reply", [{"type": "pong"}, {"type": "error"}, ["pong"]])
def test_health_probe_only_sends_ping(reply):
    probe = load_tool("healthcheck")
    received = []

    def handler(websocket):
        received.append(json.loads(websocket.recv()))
        websocket.send(json.dumps(reply))

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
            if reply == {"type": "pong"}:
                probe.check(url, 2)
            else:
                with pytest.raises(ValueError, match="expected policy pong"):
                    probe.check(url, 2)
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert received == [{"type": "ping"}]


@pytest.mark.parametrize("explicit", [False, True])
def test_health_cli_follows_configured_endpoint(monkeypatch, explicit):
    probe = load_tool("healthcheck")
    calls = []
    monkeypatch.setenv("OPENWAM_SERVER_URL", "ws://127.0.0.1:18848")
    monkeypatch.setattr(probe, "check", lambda url, timeout: calls.append((url, timeout)))
    monkeypatch.setattr(sys, "argv", ["healthcheck", *(["--url", "ws://localhost:28848"] if explicit else [])])
    assert probe.main() == 0
    assert calls == [("ws://localhost:28848" if explicit else "ws://127.0.0.1:18848", 30)]
