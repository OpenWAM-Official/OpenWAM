"""Opt-in Docker regressions; run `make docker-integration-check` (no GPU needed)."""

import gzip
import hashlib
import importlib.util
import io
import json
import os
import shlex
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("OPENWAM_DOCKER_TEST_IMAGE")
DOCKER = shlex.split(os.environ.get("OPENWAM_DOCKER_COMMAND", "docker"))


@unittest.skipUnless(IMAGE, "requires make docker-integration-check")
class DockerIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="openwam-docker-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.env = dict(os.environ)
        for key in (
            "OPENWAM_IMAGE",
            "DOCKER_IMAGE",
            "COMPOSE_FILE",
            "COMPOSE_PROFILES",
            "COMPOSE_ENV_FILES",
            "COMPOSE_PROJECT_NAME",
            "OPENWAM_CACHE_DIR",
            "OPENWAM_OUTPUT_DIR",
            "OPENWAM_UID",
            "OPENWAM_GID",
            "OPENWAM_SOURCE_DIR",
            "OPENWAM_WORKSPACE_DIR",
            "MAKEFLAGS",
            "MFLAGS",
            "MAKEOVERRIDES",
        ):
            self.env.pop(key, None)
        self.env.update(OPENWAM_IMAGE=IMAGE, DOCKER=shlex.join(DOCKER))

    def run_command(self, command, **kwargs):
        result = subprocess.run(command, capture_output=True, text=True, env=self.env, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("NWRAP_ERROR", result.stderr)
        return result.stdout.strip()

    def test_make_and_compose_share_dotenv_and_environment_selection(self):
        for name in ("Makefile", "compose.yaml"):
            shutil.copy2(ROOT / name, self.directory / name)
        self.env.pop("OPENWAM_IMAGE")
        (self.directory / ".env").write_text("OPENWAM_IMAGE=openwam:from-dotenv\n")
        for expected in ("openwam:from-dotenv", "openwam:from-environment"):
            make = self.run_command(["make", "--no-print-directory", "docker-image"], cwd=self.directory)
            compose = self.run_command([*DOCKER, "compose", "config", "--images", "serve"], cwd=self.directory)
            self.assertEqual(make, expected)
            self.assertEqual(compose, expected)
            self.env["OPENWAM_IMAGE"] = "openwam:from-environment"
        override = self.run_command(
            ["make", "--no-print-directory", "docker-image", "OPENWAM_IMAGE=openwam:command-line"], cwd=self.directory
        )
        self.assertEqual(override, "openwam:command-line")
        legacy = subprocess.run(
            ["make", "docker-image", "DOCKER_IMAGE=openwam:obsolete"],
            cwd=self.directory,
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(legacy.returncode, 0)
        self.assertIn("was replaced by OPENWAM_IMAGE", legacy.stderr)

    def test_make_and_compose_share_override_files(self):
        for name in ("Makefile", "compose.yaml"):
            shutil.copy2(ROOT / name, self.directory / name)
        for selection in ("automatic", "dotenv", "environment", "make"):
            with self.subTest(selection=selection):
                directory = self.directory / selection
                directory.mkdir()
                for name in ("Makefile", "compose.yaml"):
                    shutil.copy2(ROOT / name, directory / name)
                filename = "compose.override.yaml" if selection == "automatic" else "custom.yaml"
                expected = "openwam:override-" + selection
                (directory / filename).write_text(f"services:\n  serve:\n    image: {expected}\n")
                files = "compose.yaml:" + filename
                self.env.pop("COMPOSE_FILE", None)
                if selection == "dotenv":
                    (directory / ".env").write_text(f"COMPOSE_FILE={files}\n")
                elif selection in ("environment", "make"):
                    self.env["COMPOSE_FILE"] = files
                compose = self.run_command([*DOCKER, "compose", "config", "--images", "serve"], cwd=directory)
                if selection == "make":
                    self.env.pop("COMPOSE_FILE")
                make_args = [f"COMPOSE_FILE={files}"] if selection == "make" else []
                make = self.run_command(["make", "--no-print-directory", "docker-image", *make_args], cwd=directory)
                self.assertEqual(compose, expected)
                self.assertEqual(make, expected)

                # Execute Make's build/check recipes, forwarding Compose to the
                # real CLI while recording expensive build/run calls.
                calls = directory / "calls.jsonl"
                recorder = directory / "docker-recorder"
                recorder.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, subprocess, sys\n"
                    f"docker = {DOCKER!r}\n"
                    "if sys.argv[1] == 'compose':\n"
                    "    sys.exit(subprocess.call([*docker, *sys.argv[1:]]))\n"
                    f"with open({str(calls)!r}, 'a') as log:\n"
                    "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                    "    if sys.argv[1] == '-m':\n"
                    "        log.write(json.dumps(['test-image', os.environ['OPENWAM_DOCKER_TEST_IMAGE']]) + '\\n')\n"
                )
                recorder.chmod(0o755)
                for name in ("compose.dev.yaml", "compose.host.yaml", "compose.worktree.yaml"):
                    shutil.copy2(ROOT / name, directory / name)
                self.run_command(
                    [
                        "make",
                        "docker-build",
                        "docker-check",
                        "docker-export",
                        "docker-integration-check",
                        f"DOCKER={recorder}",
                        f"PYTHON={recorder}",
                        "VCS_REF=probe",
                        *make_args,
                    ],
                    cwd=directory,
                )
                commands = [json.loads(line) for line in calls.read_text().splitlines()]
                build = next(cmd for cmd in commands if cmd[0] == "build")
                self.assertEqual(build[build.index("-t") + 1], expected)
                runs = [cmd for cmd in commands if cmd[0] == "run"]
                self.assertEqual(len(runs), 3)
                for cmd in runs:
                    self.assertIn(expected, cmd)
                export = next(cmd for cmd in commands if cmd[0] == "docker/offline.py")
                self.assertEqual(export[export.index("export") + 1], expected)
                self.assertIn(["test-image", expected], commands)

    def test_development_source_and_outputs_survive_container_exit(self):
        checkout = self.directory / "checkout"
        output = self.directory / "external-outputs"
        cache = self.directory / "external-cache"
        for directory in (checkout / "outputs", checkout / "docker", checkout / "openwam", output, cache):
            directory.mkdir(parents=True)
        shutil.copy2(ROOT / "docker/entrypoint.sh", checkout / "docker/entrypoint.sh")
        (checkout / "openwam/__init__.py").write_text("development_marker = 'live checkout'\n")
        (checkout / "probe.py").write_text(
            "from pathlib import Path\n"
            "import openwam\n"
            "assert openwam.development_marker == 'live checkout'\n"
            "Path('outputs/checkpoint.txt').write_text('training result')\n"
            "assert Path('/outputs/checkpoint.txt').read_text() == 'training result'\n"
            "Path('/outputs/reverse.txt').write_text('same mount')\n"
            "assert Path('outputs/reverse.txt').read_text() == 'same mount'\n"
            "Path('/cache/probe.txt').write_text('persistent cache')\n"
        )
        self.env.update(
            OPENWAM_SOURCE_DIR=str(checkout),
            OPENWAM_OUTPUT_DIR=str(output),
            OPENWAM_CACHE_DIR=str(cache),
            OPENWAM_UID=str(os.getuid()),
            OPENWAM_GID=str(os.getgid()),
        )
        compose = [
            *DOCKER,
            "compose",
            "-p",
            "openwam-check-" + uuid.uuid4().hex[:12],
            "-f",
            str(ROOT / "compose.yaml"),
            "-f",
            str(ROOT / "compose.dev.yaml"),
        ]
        self.addCleanup(self.run_command, [*compose, "down", "--remove-orphans"], cwd=ROOT)
        config = json.loads(self.run_command([*compose, "--profile", "train", "config", "--format", "json"], cwd=ROOT))
        for service in ("serve", "train"):
            volumes = {volume["target"]: volume["source"] for volume in config["services"][service]["volumes"]}
            self.assertEqual(volumes["/opt/openwam"], str(checkout))
            self.assertEqual(volumes["/opt/openwam/outputs"], str(output))
            self.assertEqual(volumes["/outputs"], str(output))
        self.run_command([*compose, "run", "--rm", "-T", "dev", "python", "probe.py"], cwd=ROOT)
        self.assertEqual((output / "checkpoint.txt").read_text(), "training result")
        self.assertEqual((cache / "probe.txt").read_text(), "persistent cache")
        self.assertEqual((output / "checkpoint.txt").stat().st_uid, os.getuid())
        self.assertFalse((checkout / "outputs/checkpoint.txt").exists())

    def test_linked_worktree_git_operations_preserve_host_repository(self):
        workspace = self.directory / "workspace with spaces"
        main = workspace / "main"
        feature = workspace / "feature"
        output = self.directory / "external-output"
        cache = self.directory / "external-cache"
        assets = self.directory / "external-assets"
        for directory in (main / "docker", output, cache, assets):
            directory.mkdir(parents=True)
        shutil.copy2(ROOT / "docker/entrypoint.sh", main / "docker/entrypoint.sh")
        (main / "tracked.txt").write_text("original\n")
        (main / ".gitignore").write_text("outputs/\nassets/\n")
        self.run_command(["git", "init", "-q", str(main)])
        self.run_command(["git", "-C", str(main), "add", "."])
        self.run_command(
            [
                "git",
                "-C",
                str(main),
                "-c",
                "user.name=Probe",
                "-c",
                "user.email=probe@example.invalid",
                "commit",
                "-qm",
                "initial",
            ]
        )
        initial = self.run_command(["git", "-C", str(main), "rev-parse", "HEAD"])
        self.run_command(["git", "-C", str(main), "worktree", "add", "-qb", "feature", str(feature)])
        pointers = [feature / ".git", main / ".git/worktrees/feature/gitdir", main / ".git/worktrees/feature/commondir"]
        original_pointers = {path: path.read_bytes() for path in pointers}
        self.env.update(
            OPENWAM_WORKSPACE_DIR=str(workspace),
            OPENWAM_SOURCE_DIR=str(feature),
            OPENWAM_OUTPUT_DIR=str(output),
            OPENWAM_CACHE_DIR=str(cache),
            OPENWAM_ASSETS_DIR=str(assets),
            OPENWAM_UID=str(os.getuid()),
            OPENWAM_GID=str(os.getgid()),
        )
        override = self.directory / "probe.yaml"
        override.write_text('services:\n  dev:\n    command: [sleep, "300"]\n    network_mode: none\n')
        compose = [*DOCKER, "compose", "-p", "openwam-worktree-" + uuid.uuid4().hex[:12]]
        for filename in (ROOT / "compose.yaml", ROOT / "compose.dev.yaml", ROOT / "compose.worktree.yaml", override):
            compose.extend(["-f", str(filename)])
        self.addCleanup(self.run_command, [*compose, "down", "--remove-orphans"])
        config = json.loads(
            self.run_command([*compose, "--profile", "train", "--profile", "dev", "config", "--format", "json"])
        )
        for service in ("serve", "train", "dev"):
            item = config["services"][service]
            self.assertEqual(item["working_dir"], str(feature))
            mounts = {v["target"]: v["source"] for v in item["volumes"]}
            self.assertEqual(mounts[str(workspace)], str(workspace))
            self.assertEqual(mounts[str(feature / "outputs")], str(output))
            self.assertEqual(mounts["/opt/openwam/outputs"], str(output))
            if service == "train":
                self.assertEqual(mounts[str(feature / "assets")], mounts["/opt/openwam/assets"])
        self.assertEqual(self.run_command([*compose, "run", "--rm", "-T", "dev", "git", "status", "--porcelain"]), "")
        self.run_command([*compose, "up", "-d", "dev"])
        execute = [*compose, "exec", "-T", "dev"]
        self.assertEqual(self.run_command([*execute, "git", "rev-parse", "--show-toplevel"]), str(feature))
        self.assertEqual(
            self.run_command([*execute, "git", "-C", "/opt/openwam", "branch", "--show-current"]), "feature"
        )
        self.run_command(
            [
                *execute,
                "python",
                "-c",
                "from pathlib import Path; Path('tracked.txt').write_text('container edit\\n'); Path('outputs/result.txt').write_text('persistent')",
            ]
        )
        self.assertIn("tracked.txt", self.run_command([*execute, "git", "diff", "--name-only"]))
        self.run_command([*execute, "git", "add", "tracked.txt"])
        self.run_command(
            [
                *execute,
                "git",
                "-c",
                "user.name=Probe",
                "-c",
                "user.email=probe@example.invalid",
                "commit",
                "-qm",
                "container edit",
            ]
        )
        committed = self.run_command([*execute, "git", "rev-parse", "HEAD"])
        self.assertNotEqual(committed, initial)
        self.assertEqual(self.run_command(["git", "-C", str(feature), "rev-parse", "HEAD"]), committed)
        self.assertEqual(self.run_command(["git", "-C", str(main), "rev-parse", "HEAD"]), initial)
        self.assertEqual(
            self.run_command([*execute, "git", "worktree", "prune", "--dry-run", "--verbose", "--expire=now"]), ""
        )
        self.run_command([*compose, "down"])
        self.assertEqual(self.run_command(["git", "-C", str(feature), "status", "--porcelain"]), "")
        self.assertEqual((feature / "tracked.txt").read_text(), "container edit\n")
        self.assertEqual((output / "result.txt").read_text(), "persistent")
        for path, data in original_pointers.items():
            self.assertEqual(path.read_bytes(), data)

    def test_offline_configuration_is_bound_to_image(self):
        spec = importlib.util.spec_from_file_location("offline", ROOT / "docker/offline.py")
        offline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(offline)
        # Check that the real CUDA image includes every export input and revision.
        extracted = self.directory / "real-image-config"
        offline.copy_configuration(offline.inspect_image(IMAGE, DOCKER), extracted, DOCKER)
        for name in offline.CONFIG_FILES:
            self.assertTrue((extracted / name).is_file(), name)

        # A tiny real image exercises save/load without archiving CUDA again.
        context = self.directory / "context"
        for name, source in offline.CONFIG_FILES.items():
            target = context / "payload" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extracted / name, target)
        frozen = context / "payload/compose.yaml"
        frozen.write_text(frozen.read_text() + "\n# frozen image configuration\n")
        (context / "Dockerfile").write_text(
            'FROM scratch\nLABEL io.openwam.bundle.schema="2" '
            'org.opencontainers.image.revision="integration-release"\n'
            'COPY payload /opt/openwam\nCMD ["/bin/true"]\n'
        )
        # A registry port must not be mistaken for a tag. These images stay local.
        repository = "localhost:5000/openwam-bundle-test-" + uuid.uuid4().hex
        tag = repository + ":latest"
        self.run_command([*DOCKER, "build", "--platform", "linux/amd64", "-t", tag, str(context)])
        self.addCleanup(self.run_command, [*DOCKER, "image", "rm", tag])
        # Export must ignore even subsequent changes to the image's build context.
        frozen.write_text("new checkout configuration that must not be exported\n")
        other_tag = repository + ":other"
        self.run_command([*DOCKER, "build", "--platform", "linux/amd64", "-t", other_tag, str(context)])
        self.addCleanup(self.run_command, [*DOCKER, "image", "rm", other_tag])
        other_identity = offline.image_identity(offline.inspect_image(other_tag, DOCKER))

        for selection, image in (("explicit", tag), ("implicit", repository)):
            with self.subTest(selection=selection):
                bundle = self.directory / selection
                self.run_command(
                    [
                        sys.executable,
                        str(ROOT / "docker/offline.py"),
                        "--docker",
                        shlex.join(DOCKER),
                        "export",
                        image,
                        str(bundle),
                    ]
                )
                # Inspect the real archive, not just the exporter's manifest:
                # bare `docker save repository` would include the other version.
                with tarfile.open(bundle / "image.tar.gz", "r:gz") as archive:
                    images = json.load(archive.extractfile("manifest.json"))
                self.assertEqual(len(images), 1)
                self.assertEqual(images[0]["RepoTags"], [tag])
                self.assertIn("# frozen image configuration", (bundle / "compose.yaml").read_text())
                manifest = json.loads((bundle / "manifest.json").read_text())
                self.assertEqual(manifest["revision"], "integration-release")
                self.assertEqual(manifest["image"], tag)
                self.assertIn("OPENWAM_IMAGE=" + tag + "\n", (bundle / ".env.example").read_text())
                self.run_command([*DOCKER, "image", "rm", tag])
                self.run_command(
                    [sys.executable, str(bundle / "docker/offline.py"), "--docker", shlex.join(DOCKER), "load", "."],
                    cwd=bundle,
                )
                self.assertEqual(offline.image_identity(offline.inspect_image(other_tag, DOCKER)), other_identity)

    def test_digest_bundle_loads_without_the_source_registry(self):
        # Serve a tiny real image from a loopback registry. Unlike locally built
        # images, a pull records a repository digest on both Docker image stores.
        spec = importlib.util.spec_from_file_location("offline", ROOT / "docker/offline.py")
        offline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(offline)
        payload = self.directory / "payload"
        for source in offline.CONFIG_FILES.values():
            target = payload / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, target)
        raw_layer = io.BytesIO()
        with tarfile.open(fileobj=raw_layer, mode="w") as archive:
            archive.add(payload, arcname="opt/openwam")

        def digest(data):
            return "sha256:" + hashlib.sha256(data).hexdigest()

        layer = gzip.compress(raw_layer.getvalue())
        config = json.dumps(
            {
                "os": "linux",
                "architecture": "amd64",
                "rootfs": {"type": "layers", "diff_ids": [digest(raw_layer.getvalue())]},
                "config": {
                    "Cmd": ["/bin/true"],
                    "Labels": {
                        offline.SCHEMA_LABEL: "2",
                        offline.REVISION_LABEL: "digest-test-" + uuid.uuid4().hex,
                    },
                },
            }
        ).encode()
        media_type = "application/vnd.docker.distribution.manifest.v2+json"
        manifest = json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": media_type,
                "config": {
                    "mediaType": "application/vnd.docker.container.image.v1+json",
                    "digest": digest(config),
                    "size": len(config),
                },
                "layers": [
                    {
                        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                        "digest": digest(layer),
                        "size": len(layer),
                    }
                ],
            }
        ).encode()
        resources = {
            "/v2/": (b"{}", "application/json"),
            "/v2/openwam/manifests/" + digest(manifest): (manifest, media_type),
            "/v2/openwam/blobs/" + digest(config): (config, "application/octet-stream"),
            "/v2/openwam/blobs/" + digest(layer): (layer, "application/octet-stream"),
        }

        class Registry(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in resources:
                    self.send_error(404)
                    return
                body, content_type = resources[self.path]
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Docker-Content-Digest", digest(body))
                self.send_header("Docker-Distribution-API-Version", "registry/2.0")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            do_HEAD = do_GET

            def log_message(self, *args):
                pass

        # Docker treats loopback registries as insecure (self-signed TLS is
        # accepted), but recent containerd stores still attempt HTTPS first.
        certificates = self.directory / "certificates"
        certificates.mkdir()
        self.run_command(
            [
                *DOCKER,
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--mount",
                f"type=bind,source={certificates},target=/certificates",
                IMAGE,
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=DNS:localhost,IP:127.0.0.1",
                "-keyout",
                "/certificates/key.pem",
                "-out",
                "/certificates/cert.pem",
            ]
        )
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certificates / "cert.pem", certificates / "key.pem")
        with ThreadingHTTPServer(("127.0.0.1", 0), Registry) as registry:
            registry.socket = tls.wrap_socket(registry.socket, server_side=True)
            worker = threading.Thread(target=registry.serve_forever, daemon=True)
            worker.start()
            reference = f"localhost:{registry.server_port}/openwam@{digest(manifest)}"
            try:
                self.run_command([*DOCKER, "pull", "--platform", "linux/amd64", reference])
            finally:
                registry.shutdown()
                worker.join(timeout=5)

        original = offline.inspect_image(reference, DOCKER)
        self.addCleanup(subprocess.run, [*DOCKER, "image", "rm", original["Id"]], capture_output=True)
        bundle = self.directory / "bundle"
        self.run_command(
            [
                sys.executable,
                str(ROOT / "docker/offline.py"),
                "--docker",
                shlex.join(DOCKER),
                "export",
                reference,
                str(bundle),
            ]
        )
        exported = offline.verify_bundle(bundle)
        selected = exported["image"]
        self.assertEqual(exported["source_image"], reference)
        self.assertTrue(selected.startswith("openwam-bundle:sha256-"))
        self.addCleanup(subprocess.run, [*DOCKER, "image", "rm", selected], capture_output=True)
        with tarfile.open(bundle / "image.tar.gz", "r:gz") as archive:
            images = json.load(archive.extractfile("manifest.json"))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["RepoTags"], [selected])
        self.run_command([*DOCKER, "image", "rm", selected, reference])
        self.assertNotEqual(
            subprocess.run([*DOCKER, "image", "inspect", original["Id"]], capture_output=True).returncode, 0
        )
        self.run_command(
            [sys.executable, str(bundle / "docker/offline.py"), "--docker", shlex.join(DOCKER), "load", "."], cwd=bundle
        )
        self.assertEqual(
            offline.image_identity(offline.inspect_image(selected, DOCKER)), offline.image_identity(original)
        )
        shutil.copy2(bundle / ".env.example", bundle / ".env")
        (bundle / ".cache/docker").mkdir(parents=True)
        (bundle / "outputs").mkdir()
        # Resolve the delivered tag through the shipped Compose configuration.
        self.env.pop("OPENWAM_IMAGE", None)
        compose = [*DOCKER, "compose", "-f", str(bundle / "compose.yaml")]
        self.addCleanup(self.run_command, [*compose, "down"], cwd=bundle)
        self.run_command([*compose, "create", "dev"], cwd=bundle)

    def test_validation_configuration_follows_image_gpus_and_host_port(self):
        self.env.update(OPENWAM_IMAGE="openwam:validation-tag", OPENWAM_TRAIN_GPUS="1,3", OPENWAM_PORT="18848")
        compose = [*DOCKER, "compose", "-f", str(ROOT / "compose.yaml")]
        for host_network, endpoint in ((False, "ws://127.0.0.1:8848"), (True, "ws://127.0.0.1:18848")):
            command = [*compose, *(["-f", str(ROOT / "compose.host.yaml")] if host_network else [])]
            config = json.loads(self.run_command([*command, "--profile", "check", "config", "--format", "json"]))
            check = config["services"]["gpu-check"]
            self.assertEqual(check["image"], "openwam:validation-tag")
            self.assertEqual(check["environment"]["CUDA_VISIBLE_DEVICES"], "1,3")
            self.assertEqual(check["network_mode"], "none")
            self.assertEqual({v["target"] for v in check["volumes"]}, {"/cache", "/outputs"})
            serve = config["services"]["serve"]
            self.assertEqual(serve["environment"]["OPENWAM_SERVER_URL"], endpoint)
            # Docker's scheduled probe and manual invocation both use the environment.
            self.assertNotIn("--url", serve["healthcheck"]["test"])

    def test_checkouts_have_independent_container_lifecycles(self):
        projects = []
        for label in ("main", "feature"):
            directory = self.directory / ("openwam-" + label + "-" + uuid.uuid4().hex[:8])
            (directory / ".cache/docker").mkdir(parents=True)
            (directory / "outputs").mkdir()
            shutil.copy2(ROOT / "compose.yaml", directory / "compose.yaml")
            (directory / ".env").write_text(f"OPENWAM_UID={os.getuid()}\nOPENWAM_GID={os.getgid()}\n")
            (directory / "probe.yaml").write_text(
                'services:\n  dev:\n    command: [python, -c, "import time; time.sleep(300)"]\n'
                "    network_mode: none\n    stdin_open: false\n    tty: false\n"
            )
            command = [*DOCKER, "compose", "-f", "compose.yaml", "-f", "probe.yaml"]
            self.addCleanup(self.run_command, [*command, "down", "--remove-orphans"], cwd=directory)
            config = json.loads(self.run_command([*command, "config", "--format", "json"], cwd=directory))
            self.assertEqual(config["name"], directory.name)
            self.run_command([*command, "up", "-d", "dev"], cwd=directory)
            container = self.run_command([*command, "ps", "-q", "dev"], cwd=directory)
            self.assertTrue(container)
            projects.append((directory, command, container))
        first, second = projects
        self.assertNotEqual(first[2], second[2])
        self.assertEqual(self.run_command([*first[1], "ps", "-q", "dev"], cwd=first[0]), first[2])
        self.run_command([*second[1], "down"], cwd=second[0])
        self.assertEqual(self.run_command([*DOCKER, "inspect", first[2], "--format", "{{.State.Running}}"]), "true")

        # Same-basename checkouts use the documented native Compose override.
        for label in ("stable", "experiment"):
            directory = self.directory / label / "OpenWAM"
            directory.mkdir(parents=True)
            shutil.copy2(ROOT / "compose.yaml", directory / "compose.yaml")
            (directory / ".env").write_text(f"COMPOSE_PROJECT_NAME=openwam-{label}\n")
            config = json.loads(self.run_command([*DOCKER, "compose", "config", "--format", "json"], cwd=directory))
            self.assertEqual(config["name"], "openwam-" + label)

    def test_runtime_identity_and_home_survive_run_and_exec(self):
        for uid, gid in ((1000, 1000), (23456, 23457)):
            with self.subTest(uid=uid, gid=gid):
                name = "openwam-user-" + uuid.uuid4().hex[:12]
                self.run_command([*DOCKER, "volume", "create", name])
                self.addCleanup(self.run_command, [*DOCKER, "volume", "rm", name])
                options = [
                    "--network",
                    "none",
                    "--user",
                    f"{uid}:{gid}",
                    "--mount",
                    f"type=volume,source={name},target=/cache",
                ]
                identity = (
                    "import os, pwd, grp, getpass, subprocess\n"
                    "from pathlib import Path\n"
                    f"assert (os.getuid(), os.getgid()) == ({uid}, {gid})\n"
                    "assert subprocess.check_output(['whoami'], text=True).strip() == 'openwam'\n"
                    "assert getpass.getuser() == pwd.getpwuid(os.getuid()).pw_name == 'openwam'\n"
                    "assert grp.getgrgid(os.getgid()).gr_name == 'openwam'\n"
                    "assert Path.home() == Path(pwd.getpwuid(os.getuid()).pw_dir) == Path('/cache/home')\n"
                )
                self.run_command(
                    [
                        *DOCKER,
                        "run",
                        "--rm",
                        *options,
                        IMAGE,
                        "python",
                        "-c",
                        identity
                        + "subprocess.run(['git', 'config', '--global', 'user.name', 'OpenWAM regression'], check=True)\n",
                    ]
                )
                # A fresh container must reuse HOME, and exec must resolve names
                # without relying on environment exports from PID 1.
                self.run_command(
                    [
                        *DOCKER,
                        "run",
                        "-d",
                        "--name",
                        name,
                        *options,
                        IMAGE,
                        "python",
                        "-c",
                        "import time; print('ready', flush=True); time.sleep(300)",
                    ]
                )
                self.addCleanup(self.run_command, [*DOCKER, "rm", "-f", name])
                for _ in range(100):
                    if "ready" in self.run_command([*DOCKER, "logs", name]):
                        break
                    time.sleep(0.1)
                else:
                    self.fail("runtime user setup did not finish")
                self.run_command(
                    [
                        *DOCKER,
                        "exec",
                        name,
                        "python",
                        "-c",
                        identity
                        + "assert subprocess.check_output(['git', 'config', '--global', 'user.name'], text=True).strip() "
                        "== 'OpenWAM regression'\n",
                    ]
                )

    def test_bootstrap_pip_fallback_installs_the_locked_version(self):
        # Local PEP 503 indexes and real wheels make this deterministic/offline:
        # the primary lacks the package; the fallback has both 1.0 and 2.0.
        indexes = self.directory / "indexes"
        package = "openwam_bootstrap_probe"
        for index in ("primary", "fallback"):
            location = indexes / index / "openwam-bootstrap-probe"
            location.mkdir(parents=True)
            links = []
            if index == "fallback":
                for version in ("1.0", "2.0"):
                    filename = f"{package}-{version}-py3-none-any.whl"
                    metadata = f"{package}-{version}.dist-info"
                    with zipfile.ZipFile(location / filename, "w") as wheel:
                        wheel.writestr(f"{package}/__init__.py", f"version = '{version}'\n")
                        wheel.writestr(
                            f"{metadata}/METADATA", f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
                        )
                        wheel.writestr(
                            f"{metadata}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
                        )
                        wheel.writestr(f"{metadata}/RECORD", "")
                    links.append(f'<a href="{filename}">{filename}</a>')
            (location / "index.html").write_text("\n".join(links))
        (indexes / "constraints.txt").write_text("openwam-bootstrap-probe==1.0\n")
        probe = (
            "import os, subprocess\n"
            "subprocess.run(['python', '-m', 'venv', '/tmp/bootstrap-venv'], check=True)\n"
            "env = dict(os.environ, PATH='/tmp/bootstrap-venv/bin:' + os.environ['PATH'], "
            "PIP_INDEX_URL='file:///indexes/primary', PIP_FALLBACK_INDEX_URL='')\n"
            "command = ['bash', '/usr/local/bin/openwam-pip-install', '--no-cache-dir', '--no-deps', "
            "'-c', '/indexes/constraints.txt', 'openwam-bootstrap-probe']\n"
            "missing = subprocess.run(command, env=env, capture_output=True, text=True)\n"
            "assert missing.returncode != 0, missing.stdout\n"
            "assert any(message in missing.stderr for message in ('No matching distribution', 'ResolutionImpossible')), missing.stderr\n"
            "env['PIP_FALLBACK_INDEX_URL'] = 'file:///indexes/fallback'\n"
            "subprocess.run(command, env=env, check=True)\n"
            "subprocess.run(['/tmp/bootstrap-venv/bin/python', '-c', "
            "\"import openwam_bootstrap_probe as p; assert p.version == '1.0'\"], check=True)\n"
        )
        self.run_command(
            [
                *DOCKER,
                "run",
                "--rm",
                "--network",
                "none",
                "--mount",
                f"type=bind,source={indexes},target=/indexes,readonly",
                IMAGE,
                "python",
                "-c",
                probe,
            ]
        )


if __name__ == "__main__":
    unittest.main()
