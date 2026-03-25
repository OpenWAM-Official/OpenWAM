"""Gradio viewer for RoboTwin camera angles — video playback, all tasks, all robots.

Reads extracted folders directly; falls back to zip if folder not found.

Features:
    - Browse all tasks x robots x episodes x variants (clean / randomized)
    - View all camera angles simultaneously (single-frame gallery + video grid)
    - Display language instructions per episode
    - Dynamic frame slider adapts to actual episode length

Usage:
    cd src/vam
    python examples/wanvideo/wam/view_cameras.py \
        --src /path/to/robotwin_2_0/dataset \
        --port 7860
"""

import argparse
import glob
import io
import json
import os
import re
import tempfile
import zipfile

import gradio as gr
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

_KNOWN_VARIANTS = ("_clean_50", "_randomized_500")


def _parse_robot_variant(name):
    """Parse 'aloha-agilex_clean_50' -> ('aloha-agilex', 'clean_50')."""
    for v in _KNOWN_VARIANTS:
        if v in name:
            return name.split(v)[0], v.lstrip("_")
    return name, "unknown"


def discover_data(src_dir):
    """Discover all tasks, robots, and variants.

    Looks for both extracted folders and zip archives.  Prefers folders.

    Returns:
        catalog: dict  task -> robot -> variant -> {"type": "dir"|"zip", "path": ...}
        variants: sorted list of variant names
    """
    tasks = sorted(
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith(".")
    )

    catalog = {}
    all_variants = set()

    for task in tasks:
        task_dir = os.path.join(src_dir, task)
        robots = {}

        # 1. Extracted folders (preferred)
        for entry in sorted(os.listdir(task_dir)):
            full = os.path.join(task_dir, entry)
            if not os.path.isdir(full):
                continue
            robot, variant = _parse_robot_variant(entry)
            if variant == "unknown":
                continue
            robots.setdefault(robot, {})[variant] = {"type": "dir", "path": full}
            all_variants.add(variant)

        # 2. Zip fallback (only if no folder exists for that robot/variant)
        for zp in sorted(glob.glob(os.path.join(task_dir, "*.zip"))):
            name = os.path.basename(zp).replace(".zip", "")
            robot, variant = _parse_robot_variant(name)
            if variant == "unknown":
                continue
            if robot in robots and variant in robots[robot]:
                continue  # folder already found
            robots.setdefault(robot, {})[variant] = {"type": "zip", "path": zp}
            all_variants.add(variant)

        if robots:
            catalog[task] = robots

    return catalog, sorted(all_variants)


# ---------------------------------------------------------------------------
# Episode / instruction discovery — works for both dir and zip
# ---------------------------------------------------------------------------

def _data_dir(entry_path):
    """Return the subdirectory containing HDF5 files (data/ or root)."""
    data_sub = os.path.join(entry_path, "data")
    if os.path.isdir(data_sub):
        return data_sub
    return entry_path


def list_episodes(entry):
    """List episodeN names, sorted by number."""
    if entry["type"] == "dir":
        episodes = []
        search_dir = _data_dir(entry["path"])
        for f in os.listdir(search_dir):
            m = re.match(r"^(episode\d+)\.hdf5$", f)
            if m:
                episodes.append(m.group(1))
    else:
        with zipfile.ZipFile(entry["path"], "r") as zf:
            episodes = []
            for n in zf.namelist():
                m = re.match(r"^.*(episode\d+)\.hdf5$", n)
                if m:
                    episodes.append(m.group(1))
    episodes.sort(key=lambda e: int(re.search(r"\d+", e).group()))
    return episodes


def _format_instruction(data):
    """Format instruction JSON into a readable string."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        parts = []
        if "seen" in data and isinstance(data["seen"], list) and data["seen"]:
            parts.append(f"[seen] {data['seen'][0]}")
            if len(data["seen"]) > 1:
                parts.append(f"  ... ({len(data['seen'])} variants total)")
        if "unseen" in data and isinstance(data["unseen"], list) and data["unseen"]:
            parts.append(f"[unseen] {data['unseen'][0]}")
            if len(data["unseen"]) > 1:
                parts.append(f"  ... ({len(data['unseen'])} variants total)")
        if parts:
            return "\n".join(parts)
        if "instruction" in data:
            return str(data["instruction"])
        return json.dumps(data, indent=2)[:500]
    if isinstance(data, list) and data:
        return data[0] if isinstance(data[0], str) else str(data[0])
    return str(data)[:500]


def read_instruction(entry, episode_name):
    """Read language instruction for an episode."""
    try:
        if entry["type"] == "dir":
            json_path = os.path.join(entry["path"], "instructions", f"{episode_name}.json")
            if not os.path.exists(json_path):
                return None
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            target = f"instructions/{episode_name}.json"
            with zipfile.ZipFile(entry["path"], "r") as zf:
                matches = [m for m in zf.namelist() if m.endswith(target)]
                if not matches:
                    return None
                data = json.loads(zf.read(matches[0]))
        return _format_instruction(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HDF5 open — dir (direct) or zip (extract to temp)
# ---------------------------------------------------------------------------

def open_hdf5(entry, episode_name):
    """Open an episode HDF5.

    Returns (h5py.File, tmp_path_or_None).
    For extracted dirs, tmp_path is None (no cleanup needed).
    For zips, tmp_path is the temp file that must be deleted after closing.
    """
    episode_file = f"{episode_name}.hdf5"
    if entry["type"] == "dir":
        hdf5_path = os.path.join(_data_dir(entry["path"]), episode_file)
        if not os.path.exists(hdf5_path):
            return None, None
        return h5py.File(hdf5_path, "r"), None
    else:
        with zipfile.ZipFile(entry["path"], "r") as zf:
            matches = [m for m in zf.namelist() if m.endswith(episode_file)]
            if not matches:
                return None, None
            data = zf.read(matches[0])
        tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        tmp.write(data)
        tmp.close()
        return h5py.File(tmp.name, "r"), tmp.name


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def get_cameras(f):
    """List all camera keys in an HDF5 file."""
    cameras = []
    if "observation" in f:
        for cam in sorted(f["observation"].keys()):
            if f"observation/{cam}/rgb" in f:
                cameras.append(f"observation/{cam}/rgb")
    if "third_view_rgb" in f:
        cameras.append("third_view_rgb")
    return cameras


def cam_label(camera_key):
    """Short display label for a camera key."""
    return camera_key.replace("observation/", "").replace("/rgb", "")


def decode_frame(f, camera_key, frame_idx):
    """Decode a single JPEG frame from HDF5."""
    ds = f[camera_key]
    total = ds.shape[0]
    idx = min(frame_idx, total - 1)
    raw = ds[idx]
    img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    return img, total


# ---------------------------------------------------------------------------
# Video grid
# ---------------------------------------------------------------------------

def make_video_grid(f, cameras, fps=10):
    """Create an H264 mp4 video showing all cameras in a grid."""
    import imageio_ffmpeg

    total_frames = f[cameras[0]].shape[0]
    first_img, _ = decode_frame(f, cameras[0], 0)
    cw, ch = first_img.size

    n_cams = len(cameras)
    cols = min(n_cams, 3)
    rows = (n_cams + cols - 1) // cols
    label_h = 24
    grid_w = cols * cw
    grid_h = rows * (ch + label_h)

    tmp_path = tempfile.mktemp(suffix=".mp4")
    writer = imageio_ffmpeg.write_frames(
        tmp_path, (grid_w, grid_h), fps=fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "fast"],
    )
    writer.send(None)

    for t in range(total_frames):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for i, cam in enumerate(cameras):
            r, c = divmod(i, cols)
            img, _ = decode_frame(f, cam, t)
            arr = np.array(img)
            y0 = r * (ch + label_h)
            x0 = c * cw
            canvas[y0 + label_h:y0 + label_h + ch, x0:x0 + cw] = arr

        canvas_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas_pil)
        for i, cam in enumerate(cameras):
            r, c = divmod(i, cols)
            y0 = r * (ch + label_h)
            x0 = c * cw
            label = cam_label(cam)
            draw.rectangle([x0, y0, x0 + cw, y0 + label_h], fill=(40, 40, 40))
            draw.text((x0 + 8, y0 + 4), f"{label}  [f{t}/{total_frames-1}]",
                      fill=(255, 255, 255))
        writer.send(np.array(canvas_pil))

    writer.close()
    return tmp_path


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(src_dir):
    catalog, variants = discover_data(src_dir)
    if not catalog:
        raise ValueError(f"No data found in {src_dir}")

    task_names = list(catalog.keys())

    # Count how many are extracted vs zip
    n_dir = sum(
        1 for t in catalog.values() for r in t.values()
        for e in r.values() if e["type"] == "dir"
    )
    n_zip = sum(
        1 for t in catalog.values() for r in t.values()
        for e in r.values() if e["type"] == "zip"
    )

    # HDF5 cache (one file at a time)
    _cache = {"f": None, "tmp": None, "key": None}

    def _close_cache():
        if _cache["f"] is not None:
            _cache["f"].close()
            if _cache["tmp"] and os.path.exists(_cache["tmp"]):
                os.unlink(_cache["tmp"])
        _cache.update(f=None, tmp=None, key=None)

    def _get_entry(task, robot, variant):
        """Get catalog entry with fallback."""
        task_robots = catalog.get(task, {})
        robot_variants = task_robots.get(robot, {})
        if variant in robot_variants:
            return robot_variants[variant]
        if robot_variants:
            return next(iter(robot_variants.values()))
        return None

    def _open(task, robot, variant, episode_name):
        entry = _get_entry(task, robot, variant)
        if entry is None:
            return None
        key = (entry["path"], episode_name)
        if _cache["key"] == key and _cache["f"] is not None:
            return _cache["f"]
        _close_cache()
        f, tmp = open_hdf5(entry, episode_name)
        _cache.update(f=f, tmp=tmp, key=key)
        return f

    def get_robots(task):
        return sorted(catalog.get(task, {}).keys())

    def get_variants(task, robot):
        task_robots = catalog.get(task, {})
        robot_variants = task_robots.get(robot, {})
        return sorted(robot_variants.keys())

    def get_episodes(task, robot, variant):
        entry = _get_entry(task, robot, variant)
        if entry is None:
            return ["episode0"]
        return list_episodes(entry)

    def get_instruction(task, robot, variant, episode_name):
        entry = _get_entry(task, robot, variant)
        if entry is None:
            return "(not found)"
        instr = read_instruction(entry, episode_name)
        return instr if instr else "(no instruction found)"

    # ---- Cascading dropdown callbacks ----

    def on_task_change(task):
        robots = get_robots(task)
        robot = robots[0] if robots else ""
        avail_variants = get_variants(task, robot)
        variant = avail_variants[0] if avail_variants else (variants[0] if variants else "")
        episodes = get_episodes(task, robot, variant)
        return (
            gr.update(choices=robots, value=robot),
            gr.update(choices=avail_variants, value=variant),
            gr.update(choices=episodes, value=episodes[0] if episodes else "episode0"),
        )

    def on_robot_change(task, robot):
        avail_variants = get_variants(task, robot)
        variant = avail_variants[0] if avail_variants else (variants[0] if variants else "")
        episodes = get_episodes(task, robot, variant)
        return (
            gr.update(choices=avail_variants, value=variant),
            gr.update(choices=episodes, value=episodes[0] if episodes else "episode0"),
        )

    def on_variant_change(task, robot, variant):
        episodes = get_episodes(task, robot, variant)
        return gr.update(choices=episodes, value=episodes[0] if episodes else "episode0")

    def show_frame_grid(task, robot, variant, episode_name, frame_idx):
        if not task or not robot or not variant or not episode_name:
            return [], "", "(loading...)", gr.update()
        f = _open(task, robot, variant, episode_name)
        if f is None:
            return [], "Failed to open HDF5", "(error)", gr.update()
        cameras = get_cameras(f)
        results = []
        total = 0
        for cam in cameras:
            img, t = decode_frame(f, cam, int(frame_idx))
            total = t
            results.append((img, cam_label(cam)))

        cam_names = [cam_label(c) for c in cameras]
        entry = _get_entry(task, robot, variant)
        src_type = entry["type"] if entry else "?"
        info = (f"frame {int(frame_idx)}/{total-1} | "
                f"{len(cameras)} cameras: {', '.join(cam_names)} | "
                f"{img.size[0]}x{img.size[1]} | source: {src_type}")

        instr = get_instruction(task, robot, variant, episode_name)
        slider_update = gr.update(maximum=max(total - 1, 0))
        return results, info, instr, slider_update

    def generate_video(task, robot, variant, episode_name):
        try:
            entry = _get_entry(task, robot, variant)
            if entry is None:
                return gr.update(value=None)
            f, tmp_hdf5 = open_hdf5(entry, episode_name)
            if f is None:
                return gr.update(value=None)
            cameras = get_cameras(f)
            video_path = make_video_grid(f, cameras, fps=10)
            f.close()
            if tmp_hdf5 and os.path.exists(tmp_hdf5):
                os.unlink(tmp_hdf5)
            return video_path
        except Exception as e:
            print(f"Video generation error: {e}")
            import traceback
            traceback.print_exc()
            return gr.update(value=None)

    # ---- Build UI ----

    init_task = task_names[0]
    init_robots = get_robots(init_task)
    init_robot = init_robots[0] if init_robots else ""
    init_variants = get_variants(init_task, init_robot)
    init_variant = init_variants[0] if init_variants else (variants[0] if variants else "clean_50")
    init_episodes = get_episodes(init_task, init_robot, init_variant)
    init_episode = init_episodes[0] if init_episodes else "episode0"

    with gr.Blocks(title="RoboTwin Camera Viewer") as app:
        gr.Markdown("# RoboTwin Camera Viewer")
        gr.Markdown(
            f"Source: `{src_dir}` | **{len(task_names)} tasks** | "
            f"**{len(variants)} variants**: {', '.join(variants)} | "
            f"**{n_dir} extracted, {n_zip} from zip**\n\n"
            "**Camera coverage per robot:**\n"
            "| Robot | head_camera | front_camera | left_camera | right_camera | third_view_rgb |\n"
            "|-------|:-----------:|:------------:|:-----------:|:------------:|:--------------:|\n"
            "| aloha-agilex | Y | **Y** | Y | Y | - |\n"
            "| ur5 | Y | - | Y | Y | **Y** |\n"
            "| arx-x5 | Y | - | Y | Y | **Y** |\n"
            "| franka | Y | - | Y | Y | **Y** |\n"
            "| piper | Y | - | Y | Y | **Y** |\n"
        )

        with gr.Row():
            task_dd = gr.Dropdown(choices=task_names, value=init_task,
                                  label="Task", filterable=True)
            robot_dd = gr.Dropdown(choices=init_robots,
                                   value=init_robot, label="Robot")
            variant_dd = gr.Dropdown(choices=init_variants,
                                     value=init_variant, label="Variant")
            episode_dd = gr.Dropdown(choices=init_episodes,
                                     value=init_episode, label="Episode",
                                     filterable=True)

        frame_slider = gr.Slider(minimum=0, maximum=200, step=1, value=0,
                                 label="Frame")

        task_dd.change(on_task_change, [task_dd], [robot_dd, variant_dd, episode_dd])
        robot_dd.change(on_robot_change, [task_dd, robot_dd], [variant_dd, episode_dd])
        variant_dd.change(on_variant_change, [task_dd, robot_dd, variant_dd], [episode_dd])

        gr.Markdown("### Language Instruction")
        instr_box = gr.Textbox(label="Instruction", interactive=False, lines=2)

        gr.Markdown("### All cameras — single frame")
        gallery = gr.Gallery(label="Cameras", columns=3, height=350)
        info_box = gr.Textbox(label="Info", interactive=False)

        all_inputs = [task_dd, robot_dd, variant_dd, episode_dd, frame_slider]
        all_outputs = [gallery, info_box, instr_box, frame_slider]

        for inp in all_inputs:
            inp.change(show_frame_grid, all_inputs, all_outputs)

        gr.Markdown("### Video grid (full episode, all cameras)")
        video_btn = gr.Button("Generate Video Grid", variant="primary")
        video_out = gr.Video(label="Camera Grid Video", format="mp4")
        video_btn.click(generate_video, [task_dd, robot_dd, variant_dd, episode_dd], [video_out])

        app.load(show_frame_grid, all_inputs, all_outputs)

    return app


def main():
    parser = argparse.ArgumentParser(description="RoboTwin Camera Viewer")
    parser.add_argument("--src", type=str,
                        default="/path/to/robotwin_2_0/dataset",
                        help="Source directory with RoboTwin data (extracted or zip)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.src)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
