import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

@dataclass
class BatchRecord:
    batch_idx:      int
    batch_size:     int
    seq_len:        int
    latency_ms:     float
    throughput_sps: float
    mflops:         float
    peak_memory_mb: float


class ScalabilityProfiler:
    """
    Context manager that wraps a single epoch and collects per-batch
    latency, throughput, FLOPs, and peak GPU memory.

    Parameters
    ----------
    model_name : str
        Label written to every CSV row (e.g. "CausalBatch", "DCC_LSTM").
    device : torch.device | None
        Inferred from CUDA availability if None.
    trace_batches : int
        Number of batches to collect a full torch.profiler trace for.
        Set to 0 to skip trace generation (faster).
    warmup_batches : int
        Batches at the start of the epoch excluded from statistics.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        trace_batches: int = 3,
        warmup_batches: int = 2,
    ):
        self.model_name     = model_name
        self.device         = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trace_batches  = trace_batches
        self.warmup_batches = warmup_batches

        self.records: list[BatchRecord] = []
        self._epoch: int = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._print_summary()

    @contextmanager
    def profile_forward(self, batch_idx: int, inputs: torch.Tensor):
        """
        Context manager wrapping the forward pass.

        Example
        -------
        with profiler.profile_forward(i, inputs):
            output, hidden = network(inputs, hidden)
        """
        is_warmup   = batch_idx < self.warmup_batches
        do_trace    = (not is_warmup) and (batch_idx < self.warmup_batches + self.trace_batches)
        B           = inputs.shape[0]
        T           = inputs.shape[1] if inputs.dim() >= 2 else 1

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()

        if do_trace:
            activities = [ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)

            with profile(
                activities=activities,
                profile_memory=True,
                with_flops=True,
                record_shapes=True,
            ) as torch_prof:
                with record_function("forward"):
                    t0 = self._start_timer()
                    yield                           # <-- forward pass runs here
                    latency_ms = self._stop_timer(t0)

            mflops = self._extract_flops(torch_prof)
        else:
            t0 = self._start_timer()
            yield                                   # <-- forward pass runs here
            latency_ms = self._stop_timer(t0)
            mflops = 0.0

        if is_warmup:
            return

        peak_mb = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda" else 0.0
        )

        self.records.append(BatchRecord(
            batch_idx      = batch_idx,
            batch_size     = B,
            seq_len        = T,
            latency_ms     = round(latency_ms, 4),
            throughput_sps = round(B / (latency_ms / 1000), 2),
            mflops         = round(mflops, 4),
            peak_memory_mb = round(peak_mb, 3),
        ))

    def next_epoch(self):
        """Call between epochs to flush records and increment counter."""
        self._print_summary()
        self.records.clear()
        self._epoch += 1

    def _print_summary(self):
        if not self.records:
            return
        lats  = np.array([r.latency_ms     for r in self.records])
        tputs = np.array([r.throughput_sps for r in self.records])
        mems  = np.array([r.peak_memory_mb for r in self.records])
        flops = np.array([r.mflops         for r in self.records])

        print(f"\n── {self.model_name}  epoch {self._epoch} ──────────────────────────")
        print(f"  Latency  (ms)  mean={lats.mean():.2f}  std={lats.std():.2f}"
              f"  p95={np.percentile(lats,95):.2f}  p99={np.percentile(lats,99):.2f}")
        print(f"  Throughput     mean={tputs.mean():.1f} sps  std={tputs.std():.1f} sps")
        if flops.max() > 0:
            print(f"  MFLOPs         mean={flops.mean():.2f}")
        if mems.max() > 0:
            print(f"  Peak VRAM (MB) mean={mems.mean():.1f}  max={mems.max():.1f}")
        print("────────────────────────────────────────────────────────────\n")

    def _start_timer(self):
        if self.device.type == "cuda":
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def _stop_timer(self, start) -> float:
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        return (time.perf_counter() - start) * 1000

    @staticmethod
    def _extract_flops(torch_prof) -> float:
        return sum(
            e.flops for e in torch_prof.key_averages() if e.flops is not None
        ) / 1e6