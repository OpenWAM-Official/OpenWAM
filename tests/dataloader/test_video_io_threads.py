"""Decoder threading of decode_video_frames: identical frames for any thread setting, bounded default, env override."""

import types

import numpy as np
import pytest

av = pytest.importorskip("av")

from openwam.dataloader.utils import video_io  # noqa: E402

ENV = "OPENWAM_VIDEO_DECODE_THREADS"


def _write_clip(path, codec, n=40, size=64):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    with av.open(str(path), "w") as out:
        stream = out.add_stream(codec, rate=10)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        # short GOP so that decoding a late window seeks to a mid-stream keyframe (the reader's fast path)
        stream.options = {"preset": "12", "g": "8"} if codec == "libsvtav1" else {"g": "8"}
        for i in range(n):  # moving content, so most frames are inter-coded
            frame = av.VideoFrame.from_ndarray(np.roll(base, 2 * i, axis=1), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@pytest.mark.parametrize("codec", ["libsvtav1", "libx264", "mpeg4"])
def test_thread_settings_decode_identical_frames(tmp_path, monkeypatch, codec):
    try:
        av.codec.Codec(codec, "w")
    except Exception:  # encoder not built into this PyAV/FFmpeg
        pytest.skip(f"encoder {codec} not available")
    path = tmp_path / f"clip_{codec}.mp4"
    _write_clip(path, codec)
    # first window decodes sequentially from frame 0; the late one seeks to a mid-stream keyframe
    for indices in ([0, 4, 8, 12, 16, 20, 24, 28, 32], [21, 23, 25, 27, 29, 31, 33, 35, 37]):
        decoded = {}
        for threads in ("0", "1", "2", "4"):
            monkeypatch.setenv(ENV, threads)
            decoded[threads] = [np.asarray(im) for im in video_io.decode_video_frames(str(path), indices, 32, 32)]
        for threads, frames in decoded.items():
            assert len(frames) == len(indices)
            for got, want in zip(frames, decoded["0"]):
                np.testing.assert_array_equal(got, want, err_msg=f"threads={threads} window={indices[0]}")


def test_default_and_override_thread_count(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert video_io._decode_thread_count() == 2
    monkeypatch.setenv(ENV, "")  # e.g. `export OPENWAM_VIDEO_DECODE_THREADS=`
    assert video_io._decode_thread_count() == 2
    monkeypatch.setenv(ENV, " 3 ")
    assert video_io._decode_thread_count() == 3
    monkeypatch.setenv(ENV, "0")
    assert video_io._decode_thread_count() == 0


@pytest.mark.parametrize("bad", ["-1", "two", "1.5", "\u00b2", "999"])
def test_invalid_thread_count_raises(monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError, match=ENV):
        video_io._decode_thread_count()


class _Stream:
    def __init__(self, codec):
        self.codec_context = types.SimpleNamespace(name=codec)
        self.thread_type = "SLICE"
        self.thread_count = 0


def test_only_libdav1d_streams_are_capped(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    av1, h264 = _Stream("libdav1d"), _Stream("h264")
    video_io._limit_decoder_threads(av1)
    video_io._limit_decoder_threads(h264)
    assert (av1.thread_type, av1.thread_count) == ("SLICE", 2)
    assert (h264.thread_type, h264.thread_count) == ("SLICE", 0)
    monkeypatch.setenv(ENV, "0")
    av1 = _Stream("libdav1d")
    video_io._limit_decoder_threads(av1)
    assert (av1.thread_type, av1.thread_count) == ("SLICE", 0)


@pytest.mark.parametrize("codec,expected", [("libsvtav1", 2), ("mpeg4", 0)])
def test_decode_applies_the_limit_to_the_opened_stream(tmp_path, monkeypatch, codec, expected):
    try:
        av.codec.Codec(codec, "w")
        if codec == "libsvtav1":
            av.codec.Codec("libdav1d", "r")  # the cap targets libdav1d; other AV1 decoders keep defaults
    except Exception:  # encoder/decoder not built into this PyAV/FFmpeg
        pytest.skip(f"{codec} encoding or libdav1d decoding not available")
    path = tmp_path / f"clip_{codec}.mp4"
    _write_clip(path, codec)
    seen = []
    real = video_io._limit_decoder_threads

    def spy(stream):
        real(stream)
        seen.append(stream.thread_count)

    monkeypatch.setattr(video_io, "_limit_decoder_threads", spy)
    monkeypatch.delenv(ENV, raising=False)
    video_io.decode_video_frames(str(path), [27, 31], 16, 16)
    assert seen == [expected]  # AV1 is decoded by libdav1d and capped; MPEG-4 keeps PyAV's default


class _SeekFailsOnce:
    """Container proxy whose first seek raises a PyAV error, like a broken index or an unseekable input."""

    def __init__(self, container):
        self._container = container
        self.seeks = 0

    def __getattr__(self, name):
        return getattr(self._container, name)

    def seek(self, *args, **kwargs):
        self.seeks += 1
        if self.seeks == 1:
            raise av.error.PermissionError(1, "forced seek failure")
        return self._container.seek(*args, **kwargs)


def test_seek_error_falls_back_to_sequential_decode(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    _write_clip(path, "mpeg4")
    indices = [27, 31]
    expected = [np.asarray(im) for im in video_io.decode_video_frames(str(path), indices, 16, 16)]
    proxies = []
    real_open = av.open

    def open_with_failing_seek(*args, **kwargs):
        proxies.append(_SeekFailsOnce(real_open(*args, **kwargs)))
        return proxies[-1]

    monkeypatch.setattr(av, "open", open_with_failing_seek)
    frames = [np.asarray(im) for im in video_io.decode_video_frames(str(path), indices, 16, 16)]
    assert proxies[0].seeks == 1  # the seek raised and the reader decoded sequentially from the start
    for got, want in zip(frames, expected):
        np.testing.assert_array_equal(got, want)
