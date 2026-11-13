"""Video ingest and the project's single frame reader (plan 3.3 step 1, 3.4, gate G5).

- :func:`probe` -- ffprobe facts with an exact decoded frame count and the file's sha256.
- :func:`normalize` -- one CFR H.264 recipe for every source, verified frame for frame count.
- :func:`iter_frames`, :func:`read_frames`, :func:`frame_count` -- the only way frames are
  read anywhere in gbf: one sequential ffmpeg decode into a pipe, indexed by CFR frame.
- :func:`pixel_sha256` -- hash of an RGB frame (shape header + raw bytes).
- :func:`detect_cuts` -- hard cuts from colour histograms (threshold still to be validated).
- :func:`extract_window` -- lossless all-intra windows, bit-identical to canonical frames.
- :func:`write_video_json` -- the ``video.json`` record (schema ``urn:gbf:schema:video``).

Canonical frame number = (sha256 of the source file, CFR frame index). The CFR index counts
from 0 at the first decoded video frame, on a grid of ``1/fps`` seconds.
"""

from gbf.ingest.cuts import (
    DEFAULT_CUT_THRESHOLD,
    cut_detection_settings,
    cut_scores,
    detect_cuts,
    frame_histogram,
)
from gbf.ingest.errors import (
    FFmpegError,
    FrameRangeError,
    IngestError,
    NormalizationError,
    ProbeError,
    RecordError,
    ToolNotFoundError,
    WindowError,
)
from gbf.ingest.frames import cfr_filter, frame_count, iter_frames, pixel_sha256, read_frames
from gbf.ingest.normalizing import NormalizeResult, normalization_settings, normalize
from gbf.ingest.probing import (
    StreamGeometry,
    VideoProbe,
    expected_cfr_frame_count,
    keyframe_flags,
    probe,
    stream_geometry,
)
from gbf.ingest.rates import format_rate, parse_rate
from gbf.ingest.record import build_video_record, write_video_json
from gbf.ingest.tools import tool_versions
from gbf.ingest.window import WindowRecord, extract_window, window_id

__all__ = [
    "DEFAULT_CUT_THRESHOLD",
    "FFmpegError",
    "FrameRangeError",
    "IngestError",
    "NormalizationError",
    "NormalizeResult",
    "ProbeError",
    "RecordError",
    "StreamGeometry",
    "ToolNotFoundError",
    "VideoProbe",
    "WindowError",
    "WindowRecord",
    "build_video_record",
    "cfr_filter",
    "cut_detection_settings",
    "cut_scores",
    "detect_cuts",
    "expected_cfr_frame_count",
    "extract_window",
    "format_rate",
    "frame_count",
    "frame_histogram",
    "iter_frames",
    "keyframe_flags",
    "normalization_settings",
    "normalize",
    "parse_rate",
    "pixel_sha256",
    "probe",
    "read_frames",
    "stream_geometry",
    "tool_versions",
    "window_id",
    "write_video_json",
]
