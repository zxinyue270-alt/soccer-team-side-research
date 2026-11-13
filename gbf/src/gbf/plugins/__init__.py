"""The plugin protocol: out-of-process plugins exchange files with the core (ADR 0001).

- :mod:`gbf.plugins.protocol` -- job.json, running a plugin, accepting its outputs.
- :mod:`gbf.plugins.licenses` -- SPDX expressions and the permissive allowlist.
- :mod:`gbf.plugins.calib` -- the calib.v1 Arrow schema of calib.parquet and its reader.

The core never imports plugin code; plugins never need to import gbf.
"""

from gbf.plugins.calib import (
    CALIB_FORMAT,
    CALIB_SCHEMA,
    CALIB_STATUSES,
    CalibFrame,
    frame_from_camera,
    read_calib,
    write_calib,
)
from gbf.plugins.errors import (
    CalibFormatError,
    JobError,
    OutputsRejectedError,
    PluginProtocolError,
    PluginRunError,
)
from gbf.plugins.licenses import (
    PERMISSIVE_LICENSES,
    LicenseExpressionError,
    effective_allowlist,
    license_problems,
)
from gbf.plugins.protocol import (
    JOB_FILE,
    RECEIPT_FILE,
    AcceptedOutputs,
    JobInput,
    PluginRun,
    accept_outputs,
    build_job,
    create_job,
    job_sha256,
    plugin_environment,
    run_plugin,
    write_job,
)

__all__ = [
    "CALIB_FORMAT",
    "CALIB_SCHEMA",
    "CALIB_STATUSES",
    "JOB_FILE",
    "PERMISSIVE_LICENSES",
    "RECEIPT_FILE",
    "AcceptedOutputs",
    "CalibFormatError",
    "CalibFrame",
    "JobError",
    "JobInput",
    "LicenseExpressionError",
    "OutputsRejectedError",
    "PluginProtocolError",
    "PluginRun",
    "PluginRunError",
    "accept_outputs",
    "build_job",
    "create_job",
    "effective_allowlist",
    "frame_from_camera",
    "job_sha256",
    "license_problems",
    "plugin_environment",
    "read_calib",
    "run_plugin",
    "write_calib",
    "write_job",
]
