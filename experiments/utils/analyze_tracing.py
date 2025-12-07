#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 PyTorch profiler 导出的 Chrome trace JSON：
- 只看某个 GPU device 的 kernel 事件
- 利用 args["Collective name"] 或 kernel 名字里的 nccl/allreduce/sendrecv 等关键字
  区分通信和计算 kernel
- 支持按 ProfilerStep#N 裁剪时间窗口
- 计算通信/计算时间 & overlap ratio
- 估计通信量（按 In/Out msg nelems 和 dtype 估计本 GPU 上的 payload 字节数）

注意：
- 现在大部分 NCCL kernel 的 args 里没有 "Collective name"/"In msg nelems"/"dtype"，
  这些信息通常存在于 CPU 侧的 record_param_comms 事件里，通过 correlation id 关联。
- 本脚本会先在所有事件中构建 correlation -> 通信元数据 的映射，再用它来估计通信量。
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


# ---------- 基础工具 ----------

def load_events(path: str) -> List[Dict[str, Any]]:
    """加载 trace json，返回 traceEvents 列表"""
    with open(path, "r") as f:
        data = json.load(f)

    # 兼容几种可能的结构
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unrecognized trace json format: no 'traceEvents' field.")


def get_step_window(events: List[Dict[str, Any]],
                    step_idx: int) -> Tuple[float, float]:
    """
    在 traceEvents 中找到 ProfilerStep#step_idx 的时间区间 [ts, ts+dur]
    没找到就返回 (None, None)
    """
    target_name = f"ProfilerStep#{step_idx}"
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("name") != target_name:
            continue
        if not ev.get("cat", "").startswith("gpu"):
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        return ts, ts + dur
    return None, None


def filter_by_window(events: List[Dict[str, Any]],
                     t0: float,
                     t1: float) -> List[Dict[str, Any]]:
    """只保留与 [t0, t1] 区间有交集的事件"""
    out = []
    for ev in events:
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        te = ts + dur
        if te <= t0 or ts >= t1:
            continue
        out.append(ev)
    return out


def get_gpu_kernel_events(events: List[Dict[str, Any]],
                          device: int) -> List[Dict[str, Any]]:
    """
    从 traceEvents 中挑出指定 GPU device 的 kernel 事件。
    返回的每个元素统一成：
      {
        "name": str,
        "start": float,
        "end": float,
        "dur": float,
        "stream": int,
        "args": dict,
      }
    """
    gpu_events: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "kernel":
            continue
        args = ev.get("args", {})
        if args.get("device") != device:
            continue

        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        te = ts + dur
        gpu_events.append(
            {
                "name": ev.get("name", ""),
                "start": ts,
                "end": te,
                "dur": dur,
                "stream": args.get("stream"),
                "args": args,
            }
        )
    return gpu_events


def split_comm_compute(events: List[Dict[str, Any]]
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    区分 通信 / 计算 kernel：

    规则：
    1) 如果 args 里有 "Collective name"，一定视为通信；
    2) 否则，如果 kernel 名字里包含以下关键字之一，也视为通信：
         nccl, allreduce, all_reduce, allgather, all_gather,
         reduce_scatter, reducescatter, broadcast, sendrecv, send_recv
    """
    COMM_KEYWORDS = (
        "nccl",
        "allreduce", "all_reduce",
        "allgather", "all_gather",
        "reduce_scatter", "reducescatter",
        "broadcast",
        "sendrecv", "send_recv",
    )

    comm: List[Dict[str, Any]] = []
    comp: List[Dict[str, Any]] = []

    for e in events:
        args = e.get("args", {})
        name = e.get("name", "").lower()

        is_comm = False

        # 情况 1：有 Collective name（某些版本 / 设置下可能存在）
        if "Collective name" in args:
            is_comm = True
        else:
            # 情况 2：靠 kernel 名字匹配 NCCL / collectives
            if any(kw in name for kw in COMM_KEYWORDS):
                is_comm = True

        if is_comm:
            comm.append(e)
        else:
            comp.append(e)

    return comm, comp


# ---------- 通信量估计相关工具 ----------

DTYPE_BYTES: Dict[str, int] = {
    # int / uint / bool
    "Bool": 1,
    "Byte": 1, "UInt8": 1, "Int8": 1, "QUInt8": 1, "QInt8": 1,
    "Int16": 2, "Short": 2, "QInt16": 2,
    "Int32": 4, "Int": 4,
    "Int64": 8, "Long": 8,
    # float family
    "Half": 2, "Float16": 2, "BFloat16": 2,
    "Float32": 4, "Float": 4,
    "Float64": 8, "Double": 8,
    # 有时会出现小写
    "half": 2, "float16": 2, "bfloat16": 2,
    "float32": 4, "float": 4,
    "float64": 8, "double": 8,
}


def _to_int_or_none(v: Any) -> int:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def build_comm_meta(events: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    在所有事件中构建 correlation -> 通信元数据 的映射。

    典型用法：
    - CPU 事件（如 record_param_comms）里带有：
        "Collective name", "In msg nelems", "Out msg nelems", "dtype", "correlation"
    - GPU kernel 事件（ncclDevKernel_*）里有 "correlation"，但没有上述字段。

    我们：
    - 遍历所有事件，找出那些带 msg 大小 / dtype / Collective name 的事件；
    - 用 args["correlation"] / "External id" / "Correlation ID" 作为 key；
    - 把这些 args 合并存到 meta 字典里。
    """
    meta: Dict[int, Dict[str, Any]] = {}

    for ev in events:
        if ev.get("ph") != "X":
            continue
        args = ev.get("args", {})
        # 尝试从多种字段里拿 correlation，这里统一成 int
        corr = _to_int_or_none(
            args.get("correlation")
            or args.get("Correlation ID")
            or args.get("CorrelationId")
            or args.get("External id")
            or args.get("external_id")
        )
        if corr is None:
            continue

        has_nelems = ("In msg nelems" in args) or ("Out msg nelems" in args)
        has_dtype = ("dtype" in args) or ("data_type" in args) or ("DType" in args)
        has_coll = ("Collective name" in args)

        if not (has_nelems or has_dtype or has_coll):
            # 这个事件没有我们关心的通信元数据，跳过
            continue

        if corr not in meta:
            meta[corr] = dict(args)
        else:
            # 简单 merge：只补充缺失字段
            for k, v in args.items():
                if k not in meta[corr]:
                    meta[corr][k] = v

    return meta


def estimate_comm_volume(comm_events: List[Dict[str, Any]],
                         corr_meta: Dict[int, Dict[str, Any]]
                         ) -> Tuple[int, Dict[str, int]]:
    """
    估计通信量（本 GPU 的 payload 字节数）

    逻辑：
    - 对每个通信 kernel（GPU 事件）：
        1) 先查它自己的 args 里是否有 "In msg nelems"/"Out msg nelems"/"dtype"；
        2) 如果没有，则用 args["correlation"]/External id 到 corr_meta 里查对应的 CPU 侧元数据；
        3) 将两份 args 合并后，再用：
             nelems = max(In msg nelems, Out msg nelems)
             bytes  = nelems * dtype_size

    返回：
      total_bytes: 所有通信 kernel 累积字节数
      per_coll:    按 Collective name 聚合的字节数（没有则归类为 "unknown"）

    注意：
    - 如果既没有 msg nelems 也没有 dtype，则该 kernel 无法估计，直接跳过。
    """
    from collections import defaultdict

    total_bytes = 0
    per_coll: Dict[str, int] = defaultdict(int)

    for e in comm_events:
        args = e.get("args", {}) or {}

        # 先从 GPU 事件 args 里拿 correlation
        corr = _to_int_or_none(
            args.get("correlation")
            or args.get("Correlation ID")
            or args.get("CorrelationId")
            or args.get("External id")
            or args.get("external_id")
        )

        # 基础元数据：先用 CPU 侧的，再用 GPU 侧覆盖
        merged_args: Dict[str, Any] = {}
        if corr is not None and corr in corr_meta:
            merged_args.update(corr_meta[corr])
        merged_args.update(args)

        coll = merged_args.get("Collective name", "unknown")

        ne_in = _to_int_or_none(merged_args.get("In msg nelems"))
        ne_out = _to_int_or_none(merged_args.get("Out msg nelems"))

        if ne_in is None and ne_out is None:
            # 没有 msg 大小信息，跳过这个 kernel
            continue

        nelems = max(ne_in or 0, ne_out or 0)
        if nelems <= 0:
            continue

        # dtype 可能存在多种 key 下
        dtype = (
            merged_args.get("dtype")
            or merged_args.get("data_type")
            or merged_args.get("DType")
        )
        if dtype is None:
            # 没 dtype，无法估算字节数，跳过
            continue

        bpe = DTYPE_BYTES.get(str(dtype), None)
        if bpe is None:
            # 未知类型，跳过
            continue

        this_bytes = nelems * bpe
        total_bytes += this_bytes
        per_coll[coll] += this_bytes

    return total_bytes, dict(per_coll)


# ---------- 区间运算：覆盖时间 & overlap ----------

def merge_intervals(intervals: List[Tuple[float, float]]
                    ) -> List[Tuple[float, float]]:
    """
    intervals: [(start, end), ...]
    返回合并后的不相交区间
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def measure_covered_time(intervals: List[Tuple[float, float]]) -> float:
    """区间并集长度"""
    merged = merge_intervals(intervals)
    return sum(e - s for s, e in merged)


def overlap_time(comm_intervals: List[Tuple[float, float]],
                 comp_intervals: List[Tuple[float, float]]) -> float:
    """
    计算“通信 & 计算都在运行”的总时长（区间交集长度）。
    用线扫算法：维护两个活动计数器。
    """
    events: List[Tuple[float, int, str]] = []
    for s, e in comm_intervals:
        events.append((s, 1, "comm"))
        events.append((e, -1, "comm"))
    for s, e in comp_intervals:
        events.append((s, 1, "comp"))
        events.append((e, -1, "comp"))

    if not events:
        return 0.0

    events.sort()
    comm_cnt = 0
    comp_cnt = 0
    prev_t = events[0][0]
    overlap = 0.0

    for t, delta, kind in events:
        # 先结算 [prev_t, t) 区间
        if comm_cnt > 0 and comp_cnt > 0:
            overlap += t - prev_t

        if kind == "comm":
            comm_cnt += delta
        else:
            comp_cnt += delta

        prev_t = t

    return overlap


# ---------- 主逻辑 ----------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch chrome trace json: "
                    "split GPU kernels into comm/compute, compute overlap ratio, "
                    "and estimate communication volume."
    )
    parser.add_argument("trace", help="Path to chrome trace json "
                                      "(from torch.profiler.export_chrome_trace).")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device id to analyze (default: 0).")
    parser.add_argument("--step", type=int, default=None,
                        help="ProfilerStep index to focus on, e.g. 10 for ProfilerStep#10. "
                             "If not set, use full trace.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print some extra info, e.g. kernel counts per stream.")
    args = parser.parse_args()

    path = args.trace
    if not os.path.exists(path):
        raise SystemExit(f"Trace file not found: {path}")

    print(f"[INFO] Loading trace from: {path}")
    events = load_events(path)
    print(f"[INFO] Total events in trace: {len(events)}")

    # 可选：按 ProfilerStep#N 裁剪时间窗口
    if args.step is not None:
        t0, t1 = get_step_window(events, args.step)
        if t0 is None:
            print(f"[WARN] ProfilerStep#{args.step} not found, "
                  f"use full trace instead.")
        else:
            step_dur_us = t1 - t0
            print(f"[INFO] Restricting to ProfilerStep#{args.step}: "
                  f"[{t0:.0f}, {t1:.0f}] (duration={step_dur_us/1000.0:.3f} ms)")
            events = filter_by_window(events, t0, t1)

    # 构建 correlation -> 通信元数据 映射（基于当前时间窗口内的所有事件）
    corr_meta = build_comm_meta(events)

    # 只保留指定 device 的 GPU kernel
    gpu_events = get_gpu_kernel_events(events, device=args.device)
    if not gpu_events:
        raise SystemExit(f"[ERROR] No GPU kernel events found for device={args.device}")

    print(f"[INFO] GPU device={args.device}, kernel events={len(gpu_events)}")

    # 区分通信 / 计算 kernel
    comm_events, comp_events = split_comm_compute(gpu_events)
    print(f"[INFO] Classified GPU kernels: comm={len(comm_events)}, "
          f"compute={len(comp_events)}")

    # 每类的区间列表
    comm_intervals = [(e["start"], e["end"]) for e in comm_events]
    comp_intervals = [(e["start"], e["end"]) for e in comp_events]

    # 覆盖时间（做了区间合并）
    comm_time_us = measure_covered_time(comm_intervals)
    comp_time_us = measure_covered_time(comp_intervals)
    ovlp_time_us = overlap_time(comm_intervals, comp_intervals)

    # 以 "GPU 活动窗口" 作为参考（comm ∪ comp 的区间并集）
    all_intervals = comm_intervals + comp_intervals
    total_active_us = measure_covered_time(all_intervals)

    # 换算成 ms
    us_to_ms = 1.0 / 1000.0
    comm_ms = comm_time_us * us_to_ms
    comp_ms = comp_time_us * us_to_ms
    ovlp_ms = ovlp_time_us * us_to_ms
    total_ms = total_active_us * us_to_ms

    # overlap ratio
    overlap_ratio_comm = (ovlp_time_us / comm_time_us) if comm_time_us > 0 else 0.0
    overlap_ratio_total = (ovlp_time_us / total_active_us) if total_active_us > 0 else 0.0

    # 通信量估计
    total_bytes, per_coll = estimate_comm_volume(comm_events, corr_meta)
    total_gb = total_bytes / (1024.0 ** 3) if total_bytes > 0 else 0.0

    print("\n========== Summary ==========")
    if args.step is not None:
        print(f"Step:             ProfilerStep#{args.step}")
    print(f"GPU device:       {args.device}")
    print(f"#kernels (comm):  {len(comm_events)}")
    print(f"#kernels (comp):  {len(comp_events)}")

    print("\n--- Time (ms, merged intervals) ---")
    print(f"Comm time:        {comm_ms:.3f} ms")
    print(f"Compute time:     {comp_ms:.3f} ms")
    print(f"Overlap time:     {ovlp_ms:.3f} ms")
    print(f"GPU active time:  {total_ms:.3f} ms")

    print("\n--- Ratios ---")
    print(f"Overlap / Comm:   {overlap_ratio_comm * 100:.2f} %  "
          f"(comm 时间中被计算覆盖的比例)")
    print(f"Overlap / Active: {overlap_ratio_total * 100:.2f} %  "
          f"(GPU 运行时间中，算子 & 通信同时活跃的比例)")

    print("\n--- Comm volume (approx, this GPU) ---")
    print(f"Total volume:     {total_gb:.6f} GB  "
          f"({total_bytes / (1024.0 ** 2):.3f} MiB)")
    if per_coll:
        print("By collective:")
        for coll, b in sorted(per_coll.items(), key=lambda x: -x[1]):
            gb = b / (1024.0 ** 3)
            print(f"  {coll:20s}: {gb:.6f} GB  ({b / (1024.0 ** 2):.3f} MiB)")

    if args.verbose:
        # 小 debug：按 stream 看一下分布
        from collections import Counter
        stream_counter_comm = Counter(e["stream"] for e in comm_events)
        stream_counter_comp = Counter(e["stream"] for e in comp_events)

        print("\n--- Per-stream kernel counts (comm) ---")
        for s, cnt in sorted(stream_counter_comm.items()):
            print(f"  stream {s}: {cnt}")

        print("\n--- Per-stream kernel counts (compute) ---")
        for s, cnt in sorted(stream_counter_comp.items()):
            print(f"  stream {s}: {cnt}")


if __name__ == "__main__":
    main()
