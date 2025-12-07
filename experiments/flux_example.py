#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
import json
import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Set

import torch
import torch.distributed as dist
import diffusers  # noqa: F401

from torch.profiler import profile, ProfilerActivity, schedule

# from xfuser.config.diffusers import has_valid_diffusers_version, get_minimum_diffusers_version

# if not has_valid_diffusers_version("flux"):
#     minimum_diffusers_version = get_minimum_diffusers_version("flux")
#     raise ImportError(f"Please install diffusers>={minimum_diffusers_version} to use Flux.")

from transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
)
from xfuser.compact.main import CompactConfig, compact_init, compact_reset, compact_hello
from xfuser.compact.utils import COMPACT_COMPRESS_TYPE
from xfuser.compact.patchpara.df_utils import PatchConfig
from xfuser.prof import Profiler, prof_summary
# ----------------------------
# Memory + step-time profiling
# ----------------------------
def _bytes_to_gb(x: int) -> float:
    return float(x) / 1e9


def _module_param_buffer_bytes(module: torch.nn.Module) -> int:
    p = 0
    for t in module.parameters(recurse=True):
        if t is None:
            continue
        p += int(t.numel() * t.element_size())
    for b in module.buffers(recurse=True):
        if b is None:
            continue
        p += int(b.numel() * b.element_size())
    return p


def _collect_unique_modules(pipe, extra_modules: List[torch.nn.Module] = None) -> List[torch.nn.Module]:
    """Try best to find all nn.Modules reachable from the pipeline without double counting."""
    seen: Set[int] = set()
    mods: List[torch.nn.Module] = []

    def add(m):
        if isinstance(m, torch.nn.Module):
            mid = id(m)
            if mid not in seen:
                seen.add(mid)
                mods.append(m)

    # diffusers pipelines usually have `components` dict
    if hasattr(pipe, "components") and isinstance(getattr(pipe, "components"), dict):
        for v in pipe.components.values():
            add(v)

    # common attribute names
    for name in ["transformer", "unet", "vae", "text_encoder", "text_encoder_2"]:
        if hasattr(pipe, name):
            add(getattr(pipe, name))

    # conservative scan over attributes (avoid heavy objects)
    for name in dir(pipe):
        if name.startswith("_"):
            continue
        try:
            v = getattr(pipe, name)
        except Exception:
            continue
        add(v)

    if extra_modules:
        for m in extra_modules:
            add(m)

    return mods


@dataclass
class StepProfile:
    step_end_events: List[torch.cuda.Event] = field(default_factory=list)
    alloc_bytes: List[int] = field(default_factory=list)        # current allocated at step end
    peak_alloc_bytes: List[int] = field(default_factory=list)   # peak allocated so far at step end (during denoising)

    def record_step_end(self, device: str):
        # 这个回调只在 DiT 去噪 loop 的每个时间步结束时调用
        e = torch.cuda.Event(enable_timing=True)
        e.record()  # record on current stream
        self.step_end_events.append(e)
        self.alloc_bytes.append(int(torch.cuda.memory_allocated(device=device)))
        self.peak_alloc_bytes.append(int(torch.cuda.max_memory_allocated(device=device)))

    def finalize_step_ms(self, start_event: torch.cuda.Event) -> List[float]:
        if not self.step_end_events:
            return []
        torch.cuda.synchronize()
        ms: List[float] = []
        prev = start_event
        for e in self.step_end_events:
            ms.append(float(prev.elapsed_time(e)))
            prev = e
        return ms


def _reduce_scalar(value: float, op: dist.ReduceOp) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    wg = get_world_group()
    dev = torch.device(f"cuda:{wg.local_rank}")
    t = torch.tensor([value], device=dev, dtype=torch.float32)
    dist.all_reduce(t, op=op)
    return float(t.item())


def _load_prompts_from_file(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"prompts_file not found: {path}")
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            prompts.append(line)
    if not prompts:
        raise ValueError(f"No valid prompts (non-empty, non-comment lines) found in {path}")
    return prompts


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")

    # -------- 新增：torch.profiler 相关 CLI --------
    parser.add_argument(
        "--torch_profiler",
        action="store_true",
        help="Enable torch.profiler and export Chrome trace JSON per rank.",
    )
    parser.add_argument(
        "--torch_profiler_dir",
        type=str,
        default=os.path.join(os.getenv("RESULT_ROOT", "./results"), "traces"),
        help="Directory to save torch.profiler Chrome trace JSON.",
    )
    parser.add_argument(
        "--torch_profiler_step",
        type=int,
        default=-1,
        help=(
            "0-based denoising step index to capture with torch.profiler. "
            "-1 means profile the whole denoising loop."
        ),
    )
    # -------- 新增：从文件批量读取 prompt --------
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help=(
            "Path to a text file containing one prompt per line. "
            "Non-empty, non-comment lines will be used. "
            "If set, overrides --prompt and runs one sampling per line "
            "without re-loading the model."
        ),
    )

    parser.add_argument(
        "--method_tag",
        type=str,
        default="ulysses",
        help="Method tag for result directories.",
    )

    # -------- xFuser 原有参数 --------
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    DTYPE = torch.half
    engine_config.runtime_config.dtype = DTYPE

    # RUN_ID 作为起始 offset（方便兼容老的脚本，或多次实验拼接）
    run_id_env = os.getenv("RUN_ID")
    try:
        base_run_id = int(run_id_env) if run_id_env is not None else 0
    except ValueError:
        base_run_id = 0

    # 结果根目录（由 run.sh 设置为 results/<method>/<ngpus>）
    base_results = os.getenv("RESULT_ROOT", "./results")

    wg = get_world_group()
    ws = wg.world_size
    local_rank = wg.local_rank
    device = f"cuda:{local_rank}"

    # Ensure results sub-dirs exist (avoid races)
    if wg.rank == 0:
        os.makedirs(base_results, exist_ok=True)
        os.makedirs(os.path.join(base_results, "imgs"), exist_ok=True)
        os.makedirs(os.path.join(base_results, "mem"), exist_ok=True)
        os.makedirs(os.path.join(base_results, "comm"), exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # -------- load text encoder 2 (支持本地目录) --------
    model_root = engine_config.model_config.model
    if os.path.isdir(model_root):
        te2_path = os.path.join(model_root.rstrip("/"), "text_encoder_2")
        local_files_only = True
    else:
        te2_path = model_root
        local_files_only = False

    text_encoder_2 = T5EncoderModel.from_pretrained(
        te2_path,
        torch_dtype=DTYPE,
        local_files_only=local_files_only,
    )
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments.configs import get_config
    compact_config = get_config("Flux", args.method_tag)
    compact_init(compact_config)
    if compact_config.enabled: # IMPORTANT: Compact should be disabled when using pipefusion
        assert args.pipefusion_parallel_degree == 1, "Compact should be disabled when using pipefusion"
    torch.distributed.barrier()
    if getattr(args, "use_fp8_t5_encoder", False):
        from optimum.quanto import freeze, qfloat8, quantize

        logging.info(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

    cache_args = {
        "use_teacache": engine_args.use_teacache,
        "use_fbcache": engine_args.use_fbcache,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": input_config.num_inference_steps,
    }

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=DTYPE,
        text_encoder_2=text_encoder_2,
    )

    if getattr(args, "enable_sequential_cpu_offload", False):
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(device)
    from xfuser.collector.collector import Collector, init
    collector = Collector(
        save_dir="./results/collector_lowrank8_iter16", 
        target_steps=None,
        target_layers=None,
        enabled=False,
        rank=local_rank
    )
    init(collector)
    # --------- parameter memory (theoretical) ---------
    modules = _collect_unique_modules(pipe, extra_modules=[text_encoder_2])
    param_bytes = 0
    for m in modules:
        param_bytes += _module_param_buffer_bytes(m)
    param_gb = _bytes_to_gb(param_bytes)

    # 提前 sync 一下，确保权重都搬完
    torch.cuda.synchronize(device=device)

    # -------- 准备 prompt 列表（支持 --prompts_file） --------
    prompts_file = getattr(args, "prompts_file", None)
    prompts: List[str] = []

    if prompts_file is not None:
        prompts = _load_prompts_from_file(prompts_file)
    else:
        base_prompt = input_config.prompt
        if base_prompt is None:
            raise ValueError("No prompt provided. Use --prompt or --prompts_file.")
        if isinstance(base_prompt, str):
            prompts = [base_prompt]
        elif isinstance(base_prompt, (list, tuple)):
            prompts = [p for p in base_prompt if p]
        else:
            raise TypeError(f"Unsupported prompt type for input_config.prompt: {type(base_prompt)}")

    if wg.rank == 0:
        print(f"[INFO] Total prompts to run: {len(prompts)}, base_run_id={base_run_id}")

    # -------- callback 能力检测 --------
    cb_mode = "none"
    sig = inspect.signature(pipe.__call__)
    has_callback_on_step_end = "callback_on_step_end" in sig.parameters
    has_callback = "callback" in sig.parameters
    if has_callback_on_step_end:
        cb_mode = "callback_on_step_end"
    elif has_callback:
        cb_mode = "callback"

    # -------- parallel_info (用于文件命名，包括 profiler trace) --------
    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )

    # -------- torch.profiler 配置（所有 run 共享） --------
    use_profiler = bool(getattr(args, "torch_profiler", False))
    profiler_dir = getattr(args, "torch_profiler_dir", os.path.join(base_results, "traces"))
    prof_step = int(getattr(args, "torch_profiler_step", -1))

    if use_profiler:
        os.makedirs(profiler_dir, exist_ok=True)
        if wg.rank == 0:
            logging.info(
                "torch_profiler enabled, traces will be saved to %s (step=%d)",
                profiler_dir,
                prof_step,
            )

    # -------- 主循环：对每个 prompt 跑一次推理 / profile --------
    for idx, prompt in enumerate(prompts):
        global_run_id = base_run_id + idx
        run_id_str = f"{global_run_id:04d}"
        run_suffix = f"_run{run_id_str}"

        # 让 input_config.prompt 跟当前 run 对齐（写 profile json 用）
        input_config.prompt = prompt

        if wg.rank == 0:
            print("=" * 80)
            print(f"[RUN {run_id_str}] prompt: {prompt}")
            print("=" * 80)

        # -------- per-run 的 StepProfile 和 CUDA 事件 --------
        step_prof = StepProfile()
        start_cuda = torch.cuda.Event(enable_timing=True)
        start_cuda.record()

        # 每个 run 单独重置 peak 统计，并记录去噪前的 baseline 显存
        torch.cuda.synchronize(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
        base_alloc = int(torch.cuda.memory_allocated(device=device))

        # -------- 组装 call_kwargs（每个 run 单独的随机种子生成器） --------
        call_kwargs: Dict[str, Any] = dict(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            max_sequence_length=256,
            guidance_scale=input_config.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(input_config.seed),
        )

        # 注意：time.time() 这套包含了 VAE，我们只用来 debug，不进 json 指标
        start_wall = time.time()

        if use_profiler:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            profile_kwargs: Dict[str, Any] = dict(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )

            # prof_step >= 0 时，只截取某一个时间步的 trace
            if prof_step >= 0:
                profile_kwargs["schedule"] = schedule(
                    wait=prof_step,  # 跳过前 wait 个 prof.step()
                    warmup=0,
                    active=1,        # 只记录一个 step
                    repeat=1,
                )

            trace_path = os.path.join(
                profiler_dir,
                f"trace_{parallel_info}{run_suffix}_rank{wg.rank}_of{ws}.json",
            )

            with profile(**profile_kwargs) as prof:
                # 在 with 里面绑定回调，这样 callback 里能访问 prof.step()
                if cb_mode == "callback_on_step_end" and has_callback_on_step_end:

                    def _cb_on_step_end(pipe_, step, timestep, callback_kwargs):
                        step_prof.record_step_end(device=device)
                        prof.step()  # 每个去噪 step 结束算一次 iteration
                        return callback_kwargs

                    call_kwargs["callback_on_step_end"] = _cb_on_step_end

                elif cb_mode == "callback" and has_callback:

                    def _cb(step, timestep, latents):
                        step_prof.record_step_end(device=device)
                        prof.step()

                    call_kwargs["callback"] = _cb
                output = pipe(**call_kwargs)
                if hasattr(prof, 'profiler') and prof.profiler is not None:
                    prof.export_chrome_trace(trace_path)
                else:
                    print(f"[DEBUG] Profiler not initialized, skipping trace export for rank {wg.rank}")

            # 兼容修补：把非法 `"Process Group Description": ,` 改成合法 JSON
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    txt = f.read()
                fixed_txt, n_sub = re.subn(
                    r'"Process Group Description":\s*,',
                    '"Process Group Description": "",',
                    txt,
                )
                if n_sub > 0:
                    with open(trace_path, "w", encoding="utf-8") as f:
                        f.write(fixed_txt)
            except Exception as e:
                print(f"[WARN] failed to post-process torch.profiler trace: {e}")

            if wg.rank == 0:
                print(f"[torch.profiler] Chrome trace written to: {trace_path}")
        else:
            # 不启用 torch.profiler，只做显存 + step 时间统计
            if cb_mode == "callback_on_step_end" and has_callback_on_step_end:

                def _cb_on_step_end(pipe_, step, timestep, callback_kwargs):
                    step_prof.record_step_end(device=device)
                    return callback_kwargs

                call_kwargs["callback_on_step_end"] = _cb_on_step_end

            elif cb_mode == "callback" and has_callback:

                def _cb(step, timestep, latents):
                    step_prof.record_step_end(device=device)

                call_kwargs["callback"] = _cb

            output = pipe(**call_kwargs)

        end_wall = time.time()
        torch.cuda.synchronize(device=device)

        wall_elapsed = end_wall - start_wall  # 整个 pipeline 时间（DiT + VAE）

        steps = int(input_config.num_inference_steps)

        # step times（只覆盖 DiT 去噪 loop）
        step_ms = step_prof.finalize_step_ms(start_cuda)
        if len(step_ms) != steps and steps > 0:
            # 极端 fallback：没拿到 callback，就用 wall_elapsed 均摊
            step_ms = [wall_elapsed * 1000.0 / steps] * steps
            cb_mode_eff = f"{cb_mode}(fallback_avg)"
        else:
            cb_mode_eff = cb_mode

        # DiT 去噪总时间（秒）= 所有 step 时间之和
        total_time_s = sum(step_ms) / 1000.0 if step_ms else 0.0

        # -------- 只看 DiT 去噪过程的峰值显存 --------
        if step_prof.peak_alloc_bytes:
            peak_alloc_denoise = max(step_prof.peak_alloc_bytes)
        else:
            peak_alloc_denoise = base_alloc

        activation_peak_bytes = max(0, peak_alloc_denoise - base_alloc)

        # 标量转 GB
        peak_alloc_gb = _bytes_to_gb(peak_alloc_denoise)          # 去噪过程峰值显存 (参数+激活)
        act_peak_gb = _bytes_to_gb(activation_peak_bytes)         # 去噪过程激活峰值（减去基线）

        # 这里把 avg_step_activation_gb 定义为：单步所需的激活峰值（因为每步复用同一套容量）
        avg_step_activation_gb = act_peak_gb

        # 可选：跨 rank 做 mean/max（仅用于打印）
        if dist.is_available() and dist.is_initialized():
            param_gb_mean = _reduce_scalar(param_gb, dist.ReduceOp.SUM) / ws
            param_gb_max = _reduce_scalar(param_gb, dist.ReduceOp.MAX)
            act_peak_gb_mean = _reduce_scalar(act_peak_gb, dist.ReduceOp.SUM) / ws
            act_peak_gb_max = _reduce_scalar(act_peak_gb, dist.ReduceOp.MAX)
        else:
            param_gb_mean = param_gb_max = param_gb
            act_peak_gb_mean = act_peak_gb_max = act_peak_gb

        # save image(s)（这部分会用到 VAE，但我们已经把 profiling 截在去噪 loop）
        if input_config.output_type == "pil":
            imgs_dir = os.path.join(base_results, "imgs")
            os.makedirs(imgs_dir, exist_ok=True)

            dp_group_index = get_data_parallel_rank()
            num_dp_groups = get_data_parallel_world_size()
            dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups

            # method tag 用 parallel_info + run_suffix，方便和 trace / mem 对应
            method_tag = f"{parallel_info}{run_suffix}"

            if pipe.is_dp_last_group():
                for i, image in enumerate(output.images):
                    # 计算全局 index，避免多卡重复命名
                    global_idx = dp_group_index * dp_batch_size + i
                    image_name = f"{method_tag}_{global_idx}.png"
                    image_path = os.path.join(imgs_dir, image_name)
                    image.save(image_path)
                    print(f"image {i} saved to {image_path}")

        # print on last rank（打印里同时给 DiT 时间 & 整体时间方便你 sanity check）
        if wg.rank == ws - 1:
            step_ms_mean = sum(step_ms) / len(step_ms) if step_ms else 0.0
            step_ms_sorted = sorted(step_ms) if step_ms else []
            step_ms_p50 = step_ms_sorted[len(step_ms_sorted) // 2] if step_ms else 0.0
            step_ms_p90 = (
                step_ms_sorted[int(len(step_ms_sorted) * 0.9) - 1]
                if len(step_ms_sorted) >= 10
                else (max(step_ms_sorted) if step_ms_sorted else 0.0)
            )
            print(
                f"[PROFILE][run={run_id_str}] cb_mode={cb_mode_eff}, steps={steps}\n"
                f"  WallTime(total pipeline, s)={wall_elapsed:.2f}\n"
                f"  DenoiseTime(only DiT, s)={total_time_s:.2f}\n"
                f"  ParamMem(theoretical)={param_gb:.2f} GB (mean={param_gb_mean:.2f}, max={param_gb_max:.2f})\n"
                f"  BaseAlloc(before_denoise)={_bytes_to_gb(base_alloc):.2f} GB\n"
                f"  PeakAlloc_denoise={peak_alloc_gb:.2f} GB\n"
                f"  ActivationPeak_denoise={act_peak_gb:.2f} GB (mean={act_peak_gb_mean:.2f}, max={act_peak_gb_max:.2f})\n"
                f"  StepActivationPeak(≈per-step activation)={avg_step_activation_gb:.2f} GB\n"
                f"  StepTime(ms, DiT only): mean={step_ms_mean:.2f}, p50={step_ms_p50:.2f}, p90={step_ms_p90:.2f}"
            )

        # -------- gather per-rank summary & write ONE json file (每个 run 一份) --------
        rank_payload: Dict[str, Any] = {
            "rank": wg.rank,
            "param_mem_gb_theoretical": param_gb,
            "peak_alloc_gb": peak_alloc_gb,
            "avg_step_activation_gb": avg_step_activation_gb,
            "total_time_s": total_time_s,
            "step_time_ms": step_ms,
        }

        if dist.is_available() and dist.is_initialized():
            gathered_payloads: List[Dict[str, Any]] = [None for _ in range(ws)]
            dist.all_gather_object(gathered_payloads, rank_payload)
        else:
            gathered_payloads = [rank_payload]

        if wg.rank == 0:
            mem_dir = os.path.join(base_results, "mem")
            os.makedirs(mem_dir, exist_ok=True)

            profile_path = os.path.join(
                mem_dir,
                f"profile_{parallel_info}{run_suffix}.json",
            )
            combined: Dict[str, Any] = {
                "parallel_info": parallel_info,
                "world_size": ws,
                "run_id": run_id_str,
                "prompt": prompt,
                "ranks": gathered_payloads,
            }
            try:
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(combined, f, indent=2)
            except Exception as e:
                print(f"[WARN] failed to write profile json: {e}")

    # -------- 所有 prompt 跑完，销毁分布式环境 --------
   # get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
