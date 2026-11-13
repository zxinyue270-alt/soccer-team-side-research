"""Lifetime of Windows analysis trees (no optional process-management dependency).

A kill-on-close job object belongs to the server, so a crashed coordinator cannot leave
model workers behind. The job also caps committed memory of the entire tree at 6000 MiB.
On other systems the manager uses a separate process group instead.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from ctypes import wintypes


class WindowsProcessTree:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        if os.name != "nt":
            raise OSError("Windows process trees are only available on Windows")
        size = ctypes.c_size_t

        class Limits(ctypes.Structure):
            _fields_ = [
                ("ProcessTime", ctypes.c_int64),
                ("JobTime", ctypes.c_int64),
                ("Flags", wintypes.DWORD),
                ("MinWorkingSet", size),
                ("MaxWorkingSet", size),
                ("ActiveProcesses", wintypes.DWORD),
                ("Affinity", size),
                ("Priority", wintypes.DWORD),
                ("Scheduling", wintypes.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "ReadOps",
                    "WriteOps",
                    "OtherOps",
                    "ReadBytes",
                    "WriteBytes",
                    "OtherBytes",
                )
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("Basic", Limits),
                ("IO", IO),
                ("ProcessMemory", size),
                ("JobMemory", size),
                ("PeakProcess", size),
                ("PeakJob", size),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.Basic.Flags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
        limits.JobMemory = 6000 * 1024**2
        try:
            if not kernel.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
                if process.poll() is None:
                    raise ctypes.WinError(ctypes.get_last_error())
        except OSError:
            self.close()
            raise

    def terminate(self) -> None:
        if self.handle and not self.kernel.TerminateJobObject(self.handle, 130):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume(self, process: subprocess.Popen[bytes]) -> None:
        """Resume the suspended launcher only after it belongs to this job.

        Python's Windows venv launcher immediately creates another interpreter process;
        assigning an already running launcher would race that child and could miss it.
        """
        resume = ctypes.WinDLL("ntdll").NtResumeProcess
        resume.argtypes = [wintypes.HANDLE]
        resume.restype = ctypes.c_long
        status = resume(int(process._handle))
        if status != 0:
            raise OSError(f"Cannot resume analysis process (NTSTATUS {status:#x})")

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
