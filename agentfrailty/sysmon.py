"""
System monitoring for Apple Silicon benchmark runs.

The whole point of this module: on an 8 GB machine, a run that spills into swap
produces throughput numbers that look real but measure macOS's swap subsystem,
not the inference runtime. Those runs must be detected and quarantined, not
averaged in.

Also captures thermal throttling state, which matters on fanless hardware
(MacBook Air) in a way it does not on actively-cooled Macs.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, asdict, field
from typing import Optional


# vm_stat reports in pages; page size is 16384 on Apple Silicon but read it
# rather than assuming.
_VM_STAT_LINE = re.compile(r'^"?([A-Za-z][A-Za-z0-9 _\-()]*?)"?:\s+(\d+)\.?$')
_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


@dataclass
class MemSnapshot:
    """A point-in-time view of memory and swap state."""

    page_size: int = 16384
    pages_free: int = 0
    pages_active: int = 0
    pages_inactive: int = 0
    pages_wired: int = 0
    pages_compressor: int = 0
    pages_purgeable: int = 0
    pages_speculative: int = 0
    swapins: int = 0
    swapouts: int = 0
    swap_used_bytes: int = 0
    swap_total_bytes: int = 0
    cpu_speed_limit: Optional[int] = None  # percent; <100 means throttled
    timestamp: float = field(default_factory=time.time)

    @property
    def free_bytes(self) -> int:
        return self.pages_free * self.page_size

    @property
    def wired_bytes(self) -> int:
        return self.pages_wired * self.page_size

    @property
    def compressor_bytes(self) -> int:
        return self.pages_compressor * self.page_size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["free_bytes"] = self.free_bytes
        d["wired_bytes"] = self.wired_bytes
        d["compressor_bytes"] = self.compressor_bytes
        return d


def _parse_vm_stat(text: str) -> dict:
    """Parse `vm_stat` output into {key: pages} plus page_size."""
    result: dict = {}
    m = _PAGE_SIZE_RE.search(text)
    if m:
        result["page_size"] = int(m.group(1))
    for line in text.splitlines():
        m = _VM_STAT_LINE.match(line.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
            result[key] = int(m.group(2))
    return result


def _parse_swapusage(text: str) -> tuple[int, int]:
    """Parse `sysctl vm.swapusage` -> (used_bytes, total_bytes)."""

    def _to_bytes(tok: str) -> int:
        m = re.match(r"([\d.]+)([KMG]?)", tok)
        if not m:
            return 0
        val = float(m.group(1))
        mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m.group(2)]
        return int(val * mult)

    total = used = 0
    mt = re.search(r"total\s*=\s*([\d.]+[KMG]?)", text)
    mu = re.search(r"used\s*=\s*([\d.]+[KMG]?)", text)
    if mt:
        total = _to_bytes(mt.group(1))
    if mu:
        used = _to_bytes(mu.group(1))
    return used, total


def _parse_therm(text: str) -> Optional[int]:
    """
    Parse `pmset -g therm` -> CPU_Speed_Limit percent, or None.

    KNOWN LIMITATION: on Apple Silicon this usually reports nothing. The field
    is an Intel-era interface; `powermetrics` has the real data but requires
    sudo, which a benchmark people are meant to reproduce cannot assume.

    So on M-series, `cpu_speed_limit` is None and RunGuard's `throttled` flag is
    always False. Do not read that as "no throttling occurred" -- it means
    "not measured".

    Throttling on this hardware is instead detected BEHAVIOURALLY, at analysis
    time: every row carries a timestamp and decode_tps, so regressing tok/s
    against elapsed-time-within-a-cell shows sustained-decode drift directly.
    That is the better measurement anyway -- it captures the effect rather than
    a proxy for it, and needs no privileges.
    """
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def snapshot() -> MemSnapshot:
    """Capture current memory, swap and thermal state. macOS only."""
    vm = _parse_vm_stat(_run(["vm_stat"]))
    used, total = _parse_swapusage(_run(["sysctl", "vm.swapusage"]))
    limit = _parse_therm(_run(["pmset", "-g", "therm"]))

    return MemSnapshot(
        page_size=vm.get("page_size", 16384),
        pages_free=vm.get("pages_free", 0),
        pages_active=vm.get("pages_active", 0),
        pages_inactive=vm.get("pages_inactive", 0),
        pages_wired=vm.get("pages_wired_down", 0),
        pages_compressor=vm.get("pages_occupied_by_compressor", 0)
        or vm.get("pages_used_by_compressor", 0),
        pages_purgeable=vm.get("pages_purgeable", 0),
        pages_speculative=vm.get("pages_speculative", 0),
        swapins=vm.get("swapins", 0),
        swapouts=vm.get("swapouts", 0),
        swap_used_bytes=used,
        swap_total_bytes=total,
        cpu_speed_limit=limit,
    )


@dataclass
class RunHealth:
    """Verdict on whether a run's timing numbers are trustworthy."""

    swapins_delta: int = 0
    swapouts_delta: int = 0
    swap_used_delta_bytes: int = 0
    throttled: bool = False
    min_cpu_speed_limit: Optional[int] = None
    clean: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RunGuard:
    """
    Context manager wrapping a single timed generation.

    Compares memory/swap state before and after. If the run caused meaningful
    swap activity or the CPU was thermally throttled, the resulting timings are
    marked unclean so downstream analysis can exclude them.

    Quality scores from an unclean run are still valid -- swapping makes a model
    slow, not wrong. Only the *speed* numbers are compromised. The schema keeps
    them so this distinction stays visible in the data.
    """

    def __init__(self, swap_page_tolerance: int = 64, poll_thermal: bool = True):
        # A handful of swapped pages is background noise from other processes.
        # Anything above tolerance means our run moved memory to disk.
        self.swap_page_tolerance = swap_page_tolerance
        self.poll_thermal = poll_thermal
        self.before: Optional[MemSnapshot] = None
        self.after: Optional[MemSnapshot] = None
        self.health = RunHealth()

    def __enter__(self) -> "RunGuard":
        self.before = snapshot()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.after = snapshot()
        b, a = self.before, self.after
        assert b is not None and a is not None

        h = self.health
        h.swapins_delta = max(0, a.swapins - b.swapins)
        h.swapouts_delta = max(0, a.swapouts - b.swapouts)
        h.swap_used_delta_bytes = a.swap_used_bytes - b.swap_used_bytes

        limits = [x for x in (b.cpu_speed_limit, a.cpu_speed_limit) if x is not None]
        h.min_cpu_speed_limit = min(limits) if limits else None
        h.throttled = h.min_cpu_speed_limit is not None and h.min_cpu_speed_limit < 100

        if h.swapins_delta > self.swap_page_tolerance:
            h.clean = False
            h.reasons.append(f"swapins_delta={h.swapins_delta}")
        if h.swapouts_delta > self.swap_page_tolerance:
            h.clean = False
            h.reasons.append(f"swapouts_delta={h.swapouts_delta}")
        if h.throttled:
            h.clean = False
            h.reasons.append(f"cpu_speed_limit={h.min_cpu_speed_limit}")

        return False  # never suppress exceptions


def available_bytes_estimate() -> int:
    """
    Headroom estimate: the page classes macOS can hand to a new allocation
    without swapping.

      free        -- unused
      inactive    -- clean pages, evictable immediately
      purgeable   -- caches the OS will drop on request
      speculative -- read-ahead, dropped under pressure

    Excludes `active` (in use) and `wired` (kernel, unreclaimable). Still
    conservative: macOS will compress and swap to make almost anything "fit",
    badly. This estimates what fits *well*, which is the only thing worth
    benchmarking on.
    """
    s = snapshot()
    pages = (
        s.pages_free + s.pages_inactive + s.pages_purgeable + s.pages_speculative
    )
    return pages * s.page_size


def cooldown(seconds: float, require_unthrottled: bool = True, max_wait: float = 300.0):
    """
    Sleep between runs so thermal state does not leak from one run to the next.

    On a fanless Air this is not optional: back-to-back runs measure the heat of
    the previous run as much as the current model.
    """
    time.sleep(seconds)
    if not require_unthrottled:
        return
    waited = 0.0
    while waited < max_wait:
        limit = snapshot().cpu_speed_limit
        if limit is None or limit >= 100:
            return
        time.sleep(5.0)
        waited += 5.0
