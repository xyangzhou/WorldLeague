"""WorldLeague command line. Run: python -m worldleague --help."""

import argparse
from .plugins import HarnessDriver, load_plugin
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def device(gpu, output):
    from .resources import CONFIG_ROOT

    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID", OMNIGIBSON_GPU_ID="0"
    )
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault(
        "OMNIGIBSON_APPDATA_PATH",
        str(Path(tempfile.gettempdir()) / f"worldleague-{os.getpid()}"),
    )
    os.environ.setdefault("VK_ICD_FILENAMES", str(CONFIG_ROOT / "esi_vulkan_egl.json"))
    os.environ.setdefault(
        "XDG_CACHE_HOME", os.environ["OMNIGIBSON_APPDATA_PATH"] + "/xdg"
    )
    path = output / "gpu.json"
    path.write_text(json.dumps(dict(renderer_index=gpu, uuid=uuid)))
    return str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="WorldLeague physical agency benchmark"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        help="Benchmark checkout or downloaded dataset directory",
    )
    cmd = parser.add_subparsers(dest="command", required=True)
    cmd.add_parser("tasks", help="List all 60 tasks with track and primary dimension")
    scores = cmd.add_parser("profile", help="Summarize one system over tasks and seeds")
    scores.add_argument("directory", type=Path)
    analysis = cmd.add_parser("analyze", help="Compare results for one or more systems")
    analysis.add_argument("--system", action="append", required=True, metavar="NAME=DIRECTORY")
    analysis.add_argument("--tariffs", type=Path, help="Historical USD per million token rates keyed by system name")
    analysis.add_argument("--seed", type=int, default=0)
    analysis.add_argument("--compare-seed", type=int)
    analysis.add_argument("--pair", nargs=2, metavar=("FIRST", "SECOND"), help="Compare matched tasks between two named conditions")
    for name in ("scene", "eval", "continuous"):
        p = cmd.add_parser(name)
        if name in ("eval", "continuous"):
            p.add_argument("--record-trajectory", action="store_true", help="Save evaluator-side RGB and physical evidence at decision or action-chunk boundaries")
        if name == "continuous":
            p.add_argument("model", help="Robot and action interface preset")
            provider = p.add_mutually_exclusive_group(required=True)
            provider.add_argument("--url", help="WebSocket policy endpoint")
            provider.add_argument("--plugin", help="module:factory returning a predictor")
        if name == "eval":
            p.add_argument("--world", choices=["run"], default="run")
            p.add_argument(
                "--action-mode",
                choices=["parameterized", "fixed-repeat"],
                default="parameterized",
            )
            interface = p.add_mutually_exclusive_group(required=True)
            interface.add_argument(
                "--config", type=Path, help="WorldLeague VLM preset name or path to a policy YAML"
            )
            interface.add_argument(
                "--policy", help="Python module:factory returning an act/answer policy"
            )
            interface.add_argument(
                "--harness", help="module:play implementing the public episode generator"
            )
        if name == "scene":
            p.add_argument("--port", type=int, default=8615)
        p.add_argument("task")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--gpu", type=int, required=True)
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.benchmark:
        os.environ["WORLDLEAGUE_BENCHMARK"] = str(args.benchmark.resolve())
    from .tasks.loader import catalogue

    if args.command == "analyze":
        from .evaluation.analysis import analyze, paired_tasks
        systems = {}
        for entry in args.system:
            name, sep, location = entry.partition("=")
            if not sep or not name or not Path(location).is_dir() or name in systems:
                parser.error("--system requires a unique name and an existing directory: NAME=DIRECTORY")
            systems[name] = Path(location)
        result = analyze(systems, args.seed, args.compare_seed, json.loads(args.tariffs.read_text()) if args.tariffs else None)
        if args.pair:
            if any(name not in systems for name in args.pair):
                parser.error("--pair accepts names supplied with --system")
            left, right = args.pair
            result["paired_conditions"] = {"first":left, "second":right, "tasks":paired_tasks(
                result["systems"][left]["tasks"], result["systems"][right]["tasks"])}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    if args.command == "tasks":
        for row in catalogue():
            print(
                row["alias"],
                row["track"],
                row["dimension"],
                row["dimension_name"],
                sep="\t",
            )
        return 0
    if args.command == "profile":
        from .score import profile

        print(
            json.dumps(
                profile(
                    json.loads(f.read_text())
                    for f in args.directory.rglob("result.json")
                ),
                indent=2,
            )
        )
        return 0
    launch_time = time.monotonic()
    session_config = None
    if args.command == "eval":
        session_config = args.config
        if args.harness == "worldharness:play" and os.environ.get("WORLDHARNESS_CONFIG"):
            session_config = Path(os.environ["WORLDHARNESS_CONFIG"])
    if args.command in ("scene", "eval"):
        from importlib.util import find_spec
        if find_spec("worldharness") is None:
            parser.error("VLM evaluation requires WorldHarness. Install the WorldHarness checkout in this environment.")
    if session_config:
        from .resources import evaluation_config
        try:
            session_config = evaluation_config(session_config)
        except FileNotFoundError as error:
            parser.error(str(error))
        if args.config:
            args.config = session_config
    args.output.mkdir(parents=True, exist_ok=False)
    if session_config:
        import yaml
        config = yaml.safe_load(session_config.read_text())
        if isinstance(config.get("policy"), dict) and config["policy"].get("type") == "worldharness":
            from .evaluation.r1_session import evaluate as evaluate_session
            return evaluate_session(args, config)
    gpu_info = device(args.gpu, args.output)
    if args.command == "continuous":
        from .continuous import plan
        from .runtime.commands.run_continuous_episode import main as run

        path = args.output / "plan.json"
        path.write_text(
            json.dumps(
                {**plan(args.model, args.task, args.url, args.output, args.seed, plugin=args.plugin),
                 "record_trajectory": args.record_trajectory}, indent=2
            )
        )
        return run(["--plan", str(path), "--gpu-info", gpu_info], launch_time=launch_time)
    from .runtime.commands.run_esi_robot import main as run

    native = [
        "--task-file",
        args.task,
        "--scene-seed",
        str(args.seed),
        "--gpu-info",
        gpu_info,
        "--output-dir",
        str(args.output),
    ]
    if args.command == "scene":
        return run(native + ["--port", str(args.port)])
    from .worldharness import evaluate

    play = load_plugin(args.harness) if args.harness else None
    policy = None
    if args.config:
        import yaml
        from .policies.chat_completions import ChatPolicy

        config = yaml.safe_load(args.config.read_text())
        (args.output / "policy.json").write_text(
            json.dumps(
                {
                    k: config[k]
                    for k in ("model", "reasoning_effort", "temperature", "max_tokens")
                    if k in config
                },
                indent=2,
            )
        )
        policy = ChatPolicy(config, action_mode=args.action_mode, output=args.output)
    elif args.policy:
        policy = load_plugin(args.policy)()

    def episode(session, controller, executor):
        episode_start = time.monotonic()
        driver = None
        if play is not None:
            driver = HarnessDriver(play, action_mode=args.action_mode,
                                   max_decisions=session.loaded_task.definition.defaults.max_steps)
        try:
            result = evaluate(
                session, controller, executor, driver if driver is not None else policy,
                args.output, world=args.world, action_mode=args.action_mode, record_trajectory=args.record_trajectory,
            )
            episode_end = time.monotonic()
        finally:
            if driver is not None:
                driver.close()
            elif args.config:
                policy.close()
        result["timings"].update(scene_load_s=session.scene_load_s, setup_s=episode_start-launch_time,
                                 episode_s=episode_end-episode_start, evaluation_total_s=episode_end-launch_time)
        result.update(task=session.loaded_task.alias, seed=args.seed, wall_s=time.monotonic()-launch_time)
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        return 0

    return run(native, episode=episode)
