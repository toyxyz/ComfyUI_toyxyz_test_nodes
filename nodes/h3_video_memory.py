"""Bounded reference decoding and cache-visible, optionally disk-backed IMAGE storage.

No shared ComfyUI code/configuration is modified. Only selected frames are converted
to RGB; output remains a normal float32 CPU tensor, including when file-backed.
"""
import logging
import math
import mmap
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from comfy_api.latest import InputImpl, Types

LOG = logging.getLogger(__name__)
MAPPED_THRESHOLD = 256 * 1024**2
RAM_HEADROOM = 512 * 1024**2
CACHE_DIRECTORY = Path(__file__).resolve().parents[1] / "test" / "cache" / "h3_video_memory"


def check_cancelled():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


def available_ram():
    try:
        from comfy.system_memory import virtual_memory_available
        return int(virtual_memory_available())
    except (ImportError, AttributeError):
        import psutil
        return int(psutil.virtual_memory().available)


def release_old_cache(required):
    """Ask Comfy to evict old results, never active inputs or loaded models."""
    try:
        from comfy.memory_management import extra_ram_release
    except ImportError:
        return
    extra_ram_release(required + RAM_HEADROOM, free_active=False)


class _MappingOwner:
    def __init__(self, size):
        self.file = self.mapping = None
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(CACHE_DIRECTORY).free < size + RAM_HEADROOM:
            raise MemoryError(f"H3 video buffer needs {size / 1024**3:.2f} GiB of temporary disk space; free space in {CACHE_DIRECTORY}.")
        self.file = tempfile.TemporaryFile(prefix="h3_frames_", suffix=".bin", dir=CACHE_DIRECTORY)
        self.mapping = None
        try:
            self.file.truncate(size)
            self.mapping = mmap.mmap(self.file.fileno(), size, access=mmap.ACCESS_WRITE)
        except BaseException:
            if self.file is not None:
                self.file.close()
            raise

    def __del__(self):
        # The owning ndarray is retained by PyTorch's storage, including tensor
        # slices after the original VIDEO/tensor is released. No tensor finalizers.
        try:
            if self.mapping is not None:
                self.mapping.close()
        finally:
            if self.file is not None:
                self.file.close()


class _MappedArray(np.ndarray):
    def __array_finalize__(self, source):
        self._mapping_owner = getattr(source, "_mapping_owner", None)


def allocate_images(shape, *, mapped=None):
    size = math.prod(shape) * 4
    scratch = math.prod(shape[1:]) * 4 * 4
    release_old_cache(min(size, MAPPED_THRESHOLD) + scratch)
    if mapped is None:
        mapped = size >= MAPPED_THRESHOLD or size + scratch + RAM_HEADROOM > available_ram()
    if not mapped:
        try:
            return torch.empty(shape, dtype=torch.float32, device="cpu"), "RAM"
        except (MemoryError, RuntimeError) as exc:
            if not isinstance(exc, MemoryError) and not any(x in str(exc).lower() for x in ("alloc", "memory")):
                raise
            LOG.warning("H3 video: RAM allocation failed; using file-backed storage at the same resolution.")
    try:
        owner = _MappingOwner(size)
        array = np.ndarray(shape, dtype=np.float32, buffer=owner.mapping).view(_MappedArray)
        array._mapping_owner = owner
        return torch.from_numpy(array), "mapped"
    except (OSError, RuntimeError, MemoryError) as exc:
        raise MemoryError(f"H3 could not allocate a {size / 1024**3:.2f} GiB video buffer. Check available RAM, Windows commit/page-file capacity and temporary disk space. {exc}") from exc


class CacheAwareVideo(InputImpl.VideoFromComponents):
    def __init__(self, components, **kwargs):
        super().__init__(components, **kwargs)
        self._cache_components = components

    def _comfy_cache_tensors(self):
        c = self._cache_components
        return c.images, c.alpha, c.audio


def video_from_components(components, **kwargs):
    return CacheAwareVideo(components, **kwargs)


class _FrameConverter:
    """Match Comfy's RGB precision, display rotation and unaligned-width padding."""
    def __init__(self):
        self.graph = None

    def convert(self, frame):
        byte_rgb = frame.format.name in ("yuvj420p", "yuvj422p", "yuvj444p", "rgb24", "rgba", "pal8")
        fmt = "rgb24" if byte_rgb else "gbrpf32le"
        if not byte_rgb and frame.width % 32:
            signature = (frame.width, frame.height, frame.format.name)
            if self.graph is None or self.graph[0] != signature:
                width, height = ((v + 31) // 32 * 32 for v in (frame.width, frame.height))
                graph = av.filter.Graph()
                source = graph.add_buffer(template=frame)
                pad = graph.add("pad", f"{width}:{height}:0:0")
                fill = graph.add("fillborders", f"left=0:right={width-frame.width}:top=0:bottom={height-frame.height}:mode=smear")
                sink = graph.add("buffersink")
                source.link_to(pad); pad.link_to(fill); fill.link_to(sink); graph.configure()
                self.graph = signature, graph, source, sink
            self.graph[2].push(frame)
            array = self.graph[3].pull().to_ndarray(format=fmt)[:frame.height, :frame.width]
        else:
            array = frame.to_ndarray(format=fmt)
        rotation = getattr(frame, "rotation", 0)
        quadrant = int(round(rotation // 90)) % 4 if rotation else 0
        if quadrant:
            array = np.rot90(array, k=quadrant, axes=(0, 1))
        return np.ascontiguousarray(array), byte_rgb


def _decode_audio(path, start, duration):
    """Decode only audio, clipped at sample boundaries; never decode video twice."""
    parts = []
    sample_rate = None
    with av.open(path) as container:
        stream = next((s for s in reversed(container.streams.audio) if s.codec_context is not None), None)
        if stream is None:
            return None
        if start > 0:
            # Compressed audio needs decoder preroll (AAC overlap/filter history).
            # Discard it by timestamp; never retain it in the returned soundtrack.
            container.seek(int(max(0, start - 1.0) / stream.time_base), stream=stream)
        resampler = av.AudioResampler(format="fltp")
        for packet in container.demux(stream):
            check_cancelled()
            try:
                decoded = packet.decode()
            except av.error.InvalidDataError:
                continue
            done = False
            for source in decoded:
                for frame in resampler.resample(source):
                    if frame.pts is None:
                        continue
                    sample_rate = frame.sample_rate
                    time = float(frame.pts * frame.time_base)
                    lo = max(0, int((start - time) * sample_rate))
                    hi = min(frame.samples, max(0, int(round((start + duration - time) * sample_rate))))
                    if hi > lo:
                        parts.append(frame.to_ndarray()[:, lo:hi])
                    if time + frame.samples / sample_rate >= start + duration:
                        done = True
                        break
                if done:
                    break
            if done:
                break
    if not parts:
        return None
    waveform = torch.from_numpy(np.concatenate(parts, axis=1))[:, :round(duration * sample_rate)].unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def load_reference_video(path, start, duration, frame_count, *, mapped=None):
    """Single video decode pass, nearest timestamp sampling at 24fps, no stack/copy batch.

    Keep at most two decoded source frames. CFR ties follow round-to-even as in
    the old loader; VFR sampling uses real presentation timestamps instead of
    assuming that all source frames have the average frame duration.
    """
    converter = _FrameConverter()
    images = None
    storage = "RAM"
    written = decoded_count = 0
    previous = None
    first_time = None
    previous_time = previous_end = 0.0
    previous_index = 0
    bit_depth = 8
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("Reference video has no decodable video stream.")
        stream = container.streams.video[0]
        release_old_cache(stream.width * stream.height * 3 * 4 * 4)
        # Bound decoder queues too; frame threading can retain many full-size frames.
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 2
        fps = float(stream.average_rate or stream.guessed_rate or 24)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Reference video has no valid frame rate.")
        start_pts = int(start / stream.time_base)
        end_pts = int((start + duration) / stream.time_base)
        if start_pts:
            container.seek(start_pts, stream=stream)

        def write_until(frame, limit, *, final_count=None):
            nonlocal images, written, storage
            if final_count is None:
                desired = written
                while desired < frame_count:
                    target = first_time + desired / 24
                    tie = math.isclose(target, limit, abs_tol=1e-9, rel_tol=0)
                    if target > limit and not tie or (tie and previous_index % 2):
                        break
                    desired += 1
            else:
                desired = final_count
            if desired <= written:
                return
            array, byte_rgb = converter.convert(frame)
            if images is None:
                images, storage = allocate_images((frame_count, *array.shape), mapped=mapped)
            elif tuple(images.shape[1:]) != array.shape:
                raise ValueError("Reference video changes display dimensions mid-clip; normalize the source dimensions first.")
            source = torch.from_numpy(array)
            images[written].copy_(source)
            if byte_rgb:
                images[written].div_(255)
            for i in range(written + 1, desired):
                images[i].copy_(images[written])
            written = desired

        done = False
        for packet in container.demux(stream):
            check_cancelled()
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue
            for frame in frames:
                if frame.pts is None or frame.pts < start_pts:
                    continue
                if frame.pts >= end_pts:
                    done = True
                    break
                time = float(frame.pts * frame.time_base)
                if previous is not None:
                    write_until(previous, (previous_time + time) / 2)
                else:
                    first_time = time
                    bit_depth = max((c.bits for c in frame.format.components), default=8)
                previous = frame
                previous_time = time
                previous_index = decoded_count
                frame_duration = float(frame.duration * frame.time_base) if frame.duration else 1 / fps
                previous_end = time + frame_duration
                decoded_count += 1
            if done:
                break
        if previous is None:
            raise ValueError("Reference video contains no decodable frames in the selected interval.")
        available = max(1, round((previous_end - first_time) * 24))
        output_count = frame_count if frame_count <= available + 1 else min(available, frame_count)
        write_until(previous, float("inf"), final_count=output_count)
        images = images[:output_count]
    audio = _decode_audio(path, start, output_count / 24)
    check_cancelled()
    LOG.info("H3 reference video: %d decoded / %d output frames at %dx%d; %s float32 storage, %.2f GiB; no full-batch copies.",
             decoded_count, output_count, images.shape[2], images.shape[1], storage, images.numel()*4/1024**3)
    color_space = InputImpl.VideoFromFile(path).get_color_space()
    return video_from_components(Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(24)),
                                 bit_depth=bit_depth, color_space=color_space)
