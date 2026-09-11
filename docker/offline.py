"""Export, verify, and load an OpenWAM image bundle (Python 3.8+, no pip dependencies)."""

import argparse
import gzip
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

SCHEMA = 3
SCHEMA_LABEL = "io.openwam.bundle.schema"
REVISION_LABEL = "org.opencontainers.image.revision"
CONFIG_FILES = {
    "compose.yaml": "compose.yaml",
    "docker/compose.host.yaml": "docker/compose.host.yaml",
    "docker/compose.dev.yaml": "docker/compose.dev.yaml",
    "docker/.env.example": "docker/.env.example",
    "docker/README.md": "docker/README.md",
    "docker/VALIDATION.md": "docker/VALIDATION.md",
    "docker/offline.py": "docker/offline.py",
}
# Older images keep their original paths and ship their original importer.
LEGACY_CONFIG_FILES = {
    "compose.yaml": "compose.yaml",
    "compose.host.yaml": "compose.host.yaml",
    "compose.dev.yaml": "compose.dev.yaml",
    ".env.example": "docker/.env.example",
    "docker.md": "assets/openwam_usage_docs/docker.md",
    "docker/offline.py": "docker/offline.py",
}
BUNDLE_CONFIG_FILES = {2: LEGACY_CONFIG_FILES, SCHEMA: CONFIG_FILES}
FILES = ("image.tar.gz", *CONFIG_FILES)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(image, docker=("docker",)):
    result = subprocess.run([*docker, "image", "inspect", image], check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)[0]
    if (metadata["Os"], metadata["Architecture"]) != ("linux", "amd64"):
        raise ValueError("the OpenWAM bundle requires a linux/amd64 image")
    return metadata


def image_identity(metadata):
    # Classic Docker reports a config digest as Id; containerd reports a
    # manifest/index digest. Compare the same filesystem and runtime settings
    # across both stores instead. Inspect may omit empty/default config fields.
    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items() if item}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return {
        "os": metadata["Os"],
        "architecture": metadata["Architecture"],
        "layers": metadata["RootFS"]["Layers"],
        "config": normalize(metadata["Config"]),
    }


def image_schema(metadata):
    labels = metadata.get("Config", {}).get("Labels") or {}
    if not isinstance(labels, dict):
        raise ValueError("invalid image labels")
    for schema in BUNDLE_CONFIG_FILES:
        if labels.get(SCHEMA_LABEL) == str(schema):
            return schema
    raise ValueError("image does not support bundle schema 2 or 3; rebuild it with make docker-build")


def release_revision(metadata):
    image_schema(metadata)
    labels = metadata.get("Config", {}).get("Labels") or {}
    revision = labels.get(REVISION_LABEL)
    if not isinstance(revision, str) or not revision or revision == "unknown":
        raise ValueError("image has no source revision; build with make docker-build or --build-arg VCS_REF=<commit>")
    return revision


def copy_configuration(metadata, destination, docker=("docker",)):
    """Extract from the inspected immutable image, without starting its process."""
    release_revision(metadata)
    result = subprocess.run(
        [*docker, "create", "--pull=never", "--network", "none", "--entrypoint", "/bin/true", metadata["Id"]],
        check=True,
        capture_output=True,
        text=True,
    )
    container = result.stdout.strip()
    try:
        for name, source in BUNDLE_CONFIG_FILES[image_schema(metadata)].items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([*docker, "cp", container + ":/opt/openwam/" + source, str(target)], check=True)
    finally:
        subprocess.run([*docker, "rm", container], check=True, stdout=subprocess.DEVNULL)


def delivery_tag(metadata, docker=("docker",)):
    """Give digest/ID selections a name that survives save/load across stores."""
    identity = image_identity(metadata)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    tag = "openwam-bundle:sha256-" + digest
    existing = subprocess.run([*docker, "image", "inspect", tag], capture_output=True, text=True)
    if existing.returncode == 0:
        if image_identity(json.loads(existing.stdout)[0]) != identity:
            raise ValueError("delivery tag already points to a different image: " + tag)
    else:
        subprocess.run([*docker, "image", "tag", metadata["Id"], tag], check=True)
    return tag


def export_bundle(image, destination, docker=("docker",)):
    if destination.exists():
        raise ValueError("bundle destination already exists: " + str(destination))
    source_image = image
    metadata = inspect_image(image, docker)
    revision = release_revision(metadata)
    schema = image_schema(metadata)
    files = ("image.tar.gz", *BUNDLE_CONFIG_FILES[schema])
    # Inspect selects :latest for a bare repository; save would include every
    # tag. Digest/ID inputs need a portable name: save/load loses repository
    # digests, and classic and containerd stores can report different image IDs.
    is_image_id = image == metadata["Id"] or (
        metadata["Id"].split(":", 1)[-1].startswith(image) and image + ":latest" not in (metadata.get("RepoTags") or [])
    )
    if is_image_id or "@" in image:
        image = delivery_tag(metadata, docker)
    elif ":" not in image.rsplit("/", 1)[-1]:
        image += ":latest"
    # Configuration and the standalone importer come from the selected image,
    # never from the checkout running this exporter. Extraction needs no GPU.
    with tempfile.TemporaryDirectory(prefix="openwam-config-") as temporary:
        source = Path(temporary)
        copy_configuration(metadata, source, docker)
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source, destination, dirs_exist_ok=True)
    env_path = destination / (".env.example" if schema == 2 else "docker/.env.example")
    env_lines = env_path.read_text().splitlines()
    env_path.write_text(
        "\n".join("OPENWAM_IMAGE=" + image if line.startswith("OPENWAM_IMAGE=") else line for line in env_lines) + "\n"
    )
    print("Saving image (this can take several minutes)...", flush=True)
    # Stream layers through gzip; never hold the image in memory or an extra tar.
    with subprocess.Popen([*docker, "image", "save", image], stdout=subprocess.PIPE) as process:
        with gzip.open(destination / "image.tar.gz", "wb", compresslevel=1) as archive:
            shutil.copyfileobj(process.stdout, archive, length=1024 * 1024)
        if process.wait() != 0:
            raise RuntimeError("docker image save failed; bundle is incomplete")
    if image_identity(inspect_image(image, docker)) != image_identity(metadata):
        raise ValueError("image tag changed while exporting; retry with a stable tag in a new directory")
    manifest = {
        "schema": schema,
        "image": image,
        "source_image": source_image,
        "image_id": metadata["Id"],
        "image_identity": image_identity(metadata),
        "platform": "linux/amd64",
        "revision": revision,
        "sha256": {name: sha256(destination / name) for name in files},
    }
    # A manifest is only published after every file is complete.
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Bundle ready: " + str(destination))


def verify_bundle(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("bundle manifest must be a JSON object")
    schema = manifest.get("schema")
    if type(schema) is not int or schema not in BUNDLE_CONFIG_FILES or manifest.get("platform") != "linux/amd64":
        raise ValueError("unsupported bundle schema or platform; use the importer shipped with that bundle")
    files = ("image.tar.gz", *BUNDLE_CONFIG_FILES[schema])
    checksums = manifest.get("sha256", {})
    if not isinstance(checksums, dict) or set(checksums) != set(files):
        raise ValueError("unexpected bundle file list")
    for name in files:
        if sha256(directory / name) != checksums[name]:
            raise ValueError("SHA256 mismatch: " + name)
    if not isinstance(manifest.get("image"), str) or not isinstance(manifest.get("image_id"), str):
        raise ValueError("missing image identity")
    identity = manifest.get("image_identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("config"), dict):
        raise ValueError("missing image layer/config identity")
    metadata = {"Config": identity["config"]}
    revision = release_revision(metadata)
    if image_schema(metadata) != schema:
        raise ValueError("bundle schema does not match the image schema")
    if manifest.get("revision") != revision:
        raise ValueError("bundle revision does not match the image revision")
    env_lines = (directory / (".env.example" if schema == 2 else "docker/.env.example")).read_text().splitlines()
    selected_images = [line for line in env_lines if line.startswith("OPENWAM_IMAGE=")]
    if selected_images != ["OPENWAM_IMAGE=" + manifest["image"]]:
        raise ValueError("bundle environment does not select the bundled image")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", default="docker", help="Docker CLI command, e.g. 'sudo docker'")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="save a local image and its run configuration")
    export.add_argument("image")
    export.add_argument("directory", type=Path)
    for name in ("verify", "load"):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
    args = parser.parse_args()
    docker = shlex.split(args.docker)
    try:
        if args.command == "export":
            export_bundle(args.image, args.directory, docker)
        else:
            manifest = verify_bundle(args.directory)
            print("Bundle checksums verified.", flush=True)
            if args.command == "load":
                subprocess.run([*docker, "image", "load", "--input", str(args.directory / "image.tar.gz")], check=True)
                if image_identity(inspect_image(manifest["image"], docker)) != manifest["image_identity"]:
                    raise ValueError("loaded image layers/config do not match the bundle")
                print("Loaded " + manifest["image"] + ". Configure .env, then run docker compose up -d serve.")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, "OpenWAM offline bundle: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
