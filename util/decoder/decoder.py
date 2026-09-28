# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import io as _pyio
import math
import random
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.io as io


# ---------------------------------------------------------------------------
# PyAV fallback. torchvision's in-memory video reader (_probe_video_from_memory,
# _read_video_from_memory) was removed in newer torchvision releases. These two
# functions keep the same inputs/outputs, and are used only when torchvision
# doesn't provide the originals.
# ---------------------------------------------------------------------------
def _pyav_probe_video_from_memory(video_tensor):
    import av
    from fractions import Fraction

    with av.open(_pyio.BytesIO(video_tensor.numpy().tobytes())) as c:
        s = c.streams.video[0]
        tb = s.time_base
        if s.duration is not None:
            duration = float(s.duration * tb)
        else:
            duration = (c.duration or 0) / 1e6          # container duration is in microseconds
        return SimpleNamespace(
            has_video=True,
            video_timebase=Fraction(tb.numerator, tb.denominator),
            video_duration=duration,                     # seconds
            video_fps=float(s.average_rate),
            has_audio=False,
            audio_timebase=Fraction(0, 1),
            audio_duration=0.0,
            audio_sample_rate=0.0,
        )


def _pyav_read_video_from_memory(
    video_tensor, video_min_dimension=0, video_pts_range=(0, -1), **_unused
):
    """Returns (frames [T, H, W, 3] uint8, None), like torchvision's reader."""
    import av

    start_pts, end_pts = video_pts_range
    frames = []
    with av.open(_pyio.BytesIO(video_tensor.numpy().tobytes())) as c:
        s = c.streams.video[0]
        if start_pts > 0:
            c.seek(int(start_pts), stream=s, backward=True, any_frame=False)
        for f in c.decode(s):
            if f.pts is None or f.pts < start_pts:
                continue
            if end_pts != -1 and f.pts > end_pts:
                break
            if video_min_dimension > 0:                  # resize so the SHORT side = video_min_dimension
                scale = video_min_dimension / min(f.width, f.height)
                f = f.reformat(width=round(f.width * scale), height=round(f.height * scale))
            frames.append(f.to_ndarray(format="rgb24"))
    if not frames:
        return torch.empty(0), None
    return torch.from_numpy(np.stack(frames)), None


_probe_video_from_memory = getattr(io, "_probe_video_from_memory", _pyav_probe_video_from_memory)
_read_video_from_memory = getattr(io, "_read_video_from_memory", _pyav_read_video_from_memory)


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)
    return new_frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, use_offset=False):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    max_spatial_scale=0,
    use_offset=False,
    rigid_decode_all_video=True,
    modalities=("visual",),
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    try:
        assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
        # Convert the bytes to a tensor.
        video_tensor = torch.from_numpy(np.frombuffer(container, dtype=np.uint8))

        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1
        # The video_meta is empty, fetch the meta data from the raw video.
        if len(video_meta) == 0:
            # Tracking the meta info for selective decoding in the future.
            meta = _probe_video_from_memory(video_tensor)
            # Using the information from video_meta to perform selective decoding.
            video_meta["video_timebase"] = meta.video_timebase
            video_meta["video_numerator"] = meta.video_timebase.numerator
            video_meta["video_denominator"] = meta.video_timebase.denominator
            video_meta["has_video"] = meta.has_video
            video_meta["video_duration"] = meta.video_duration
            video_meta["video_fps"] = meta.video_fps
            video_meta["audio_timebas"] = meta.audio_timebase
            video_meta["audio_numerator"] = meta.audio_timebase.numerator
            video_meta["audio_denominator"] = meta.audio_timebase.denominator
            video_meta["has_audio"] = meta.has_audio
            video_meta["audio_duration"] = meta.audio_duration
            video_meta["audio_sample_rate"] = meta.audio_sample_rate

        fps = video_meta["video_fps"]
        if not rigid_decode_all_video:
            if (
                video_meta["has_video"]
                and video_meta["video_denominator"] > 0
                and video_meta["video_duration"] > 0
            ):
                # try selective decoding.
                decode_all_video = False
                clip_size = sampling_rate * num_frames / target_fps * fps
                start_idx, end_idx = get_start_end_idx(
                    fps * video_meta["video_duration"],
                    clip_size,
                    clip_idx,
                    num_clips,
                    use_offset=use_offset,
                )
                # Convert frame index to pts.
                pts_per_frame = video_meta["video_denominator"] / fps
                video_start_pts = int(start_idx * pts_per_frame)
                video_end_pts = int(end_idx * pts_per_frame)

        # Decode the raw video with the tv decoder.
        v_frames, _ = _read_video_from_memory(
            video_tensor,
            seek_frame_margin=1.0,
            read_video_stream="visual" in modalities,
            video_width=0,
            video_height=0,
            video_min_dimension=max_spatial_scale,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase_numerator=video_meta["video_numerator"],
            video_timebase_denominator=video_meta["video_denominator"],
        )

        if v_frames.shape == torch.Size([0]):
            # failed selective decoding
            decode_all_video = True
            video_start_pts, video_end_pts = 0, -1
            v_frames, _ = _read_video_from_memory(
                video_tensor,
                seek_frame_margin=1.0,
                read_video_stream="visual" in modalities,
                video_width=0,
                video_height=0,
                video_min_dimension=max_spatial_scale,
                video_pts_range=(video_start_pts, video_end_pts),
                video_timebase_numerator=video_meta["video_numerator"],
                video_timebase_denominator=video_meta["video_denominator"],
            )
    except Exception as e:
        print("Failed to decode by torchvision with exception: {}".format(e))
        return None

    # Return None if the frames was not decoded successfully.
    if v_frames is None or v_frames.size(0) == 0:
        return None, fps, decode_all_video
    return v_frames, fps, decode_all_video
