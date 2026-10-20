"""vLLM worker extension for the per-layer check.

Reuses the profiler's scheduler-bypass ``Shot`` so the batch shape is
exactly the one asked for, but records every GPU kernel with
``torch.profiler`` instead of ``layerwise_profile``. That hook needs
eager mode (it attributes kernels to nn.Modules from Python), while
this check exists to look inside CUDA-graph runs, where no Python runs
on replay and only the kernels themselves can be seen.
"""

from __future__ import annotations

from typing import Any

from profiler.core.hooks.batch import Shot, assemble_scheduler_output


class Extension:
    """Worker-side entry point; ``self.model_runner`` is injected by vLLM."""

    def run_shot(self, shot_dict: dict[str, Any], tag: str,
                 iterations: int = 3, flush_l2_mb: int = 0) -> dict[str, Any]:
        """Run one shot ``iterations`` times and return its kernels.

        Returns ``{"forward_ms": [...], "kernels": [...]}``. Each kernel is
        ``{"name", "start_us", "dur_us"}`` in launch order, across all timed
        forwards; the host splits them per forward by counting.

        ``flush_l2_mb > 0`` overwrites a buffer that large before every timed
        forward, so nothing the previous forward read is still in L2. The
        fill kernel lands in each forward's kernel list ahead of the first
        attention, i.e. in the first piece, never in an attention.
        """
        import torch
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity, profile

        shot = Shot.hydrate(shot_dict)

        def _forward():
            batch, _ = assemble_scheduler_output(shot, self.model_runner)
            out = self.model_runner.execute_model(batch)
            if out is None:
                self.model_runner.sample_tokens(None)

        # Two warmups: the first may still be compiling or capturing a graph
        # for this batch size.
        _forward()
        _forward()
        torch.cuda.synchronize()

        flush = (torch.empty(flush_l2_mb << 20, dtype=torch.uint8, device="cuda")
                 if flush_l2_mb > 0 else None)

        forward_ms: list[float] = []
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iterations):
                if flush is not None:
                    flush.fill_(1)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.nvtx.range_push(tag)
                start.record()
                _forward()
                end.record()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                forward_ms.append(start.elapsed_time(end))

        kernels = [
            {"name": e.name,
             "start_us": e.time_range.start,
             "dur_us": e.time_range.elapsed_us()}
            for e in prof.events()
            if e.device_type == DeviceType.CUDA
        ]
        kernels.sort(key=lambda k: k["start_us"])
        return {"forward_ms": forward_ms, "kernels": kernels}
