"""Prepare local PandaOmron assets. Run python -m worldleague.prepare_robot --help."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from .cli import device
from .resources import DATASET_ROOT, CONFIG_ROOT
from .runtime.device import install_gpu_selection


def native_asset(output, mesh_directory, scratch):
    from isaacsim import SimulationApp

    source = output / "source"
    shutil.copytree(DATASET_ROOT / "robots/pandaomron", source)
    (source / "assets").symlink_to(mesh_directory.resolve(), target_is_directory=True)
    app = SimulationApp({"headless": True, "width": 256, "height": 256})
    try:
        import omni.kit.commands
        from isaacsim.core.utils.extensions import enable_extension
        from .environments.pandaomron_asset import prepare_asset

        enable_extension("isaacsim.asset.importer.mjcf")
        app.update()
        ok, config = omni.kit.commands.execute("MJCFCreateImportConfig")
        if not ok:
            raise RuntimeError("MJCF importer initialization failed")
        for field, value in {
            "fix_base": True,
            "import_inertia_tensor": True,
            "self_collision": True,
            "import_sites": True,
            "visualize_collision_geoms": False,
            "make_default_prim": True,
        }.items():
            getattr(config, "set_" + field)(value)
        raw = scratch / "pandaomron_raw.usd"
        ok, response = omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=str(source / "pandaomron.xml"),
            import_config=config,
            prim_path="/PandaOmron",
            dest_path=str(raw),
        )
        if not ok:
            raise RuntimeError(str(response))
        for _ in range(4):
            app.update()
        prepare_asset(raw, source, output / "native")
    finally:
        app.close()


def compound_asset(output):
    from omnigibson import simulator
    from .runtime.device import isolate_texture_cache

    isolate_texture_cache(simulator, os.environ["OMNIGIBSON_APPDATA_PATH"])
    import omnigibson as og
    from .continuous import resolve_robot_assets
    from .environments.robots import pandaomron  # noqa: F401 - register the native robot class
    from .environments.compound_tray_asset import build_compound_asset
    from .environments.robot_config import native_robot_config
    from .environments.mobile_tray import support_objects

    presets = CONFIG_ROOT / "continuous/robots"
    robot = resolve_robot_assets(
        yaml.safe_load((presets / "pandaomron.yaml").read_text())
    )
    tray = yaml.safe_load(
        (presets / "pandaomron_mobile_tray_compound.yaml").read_text()
    )
    support = tray["eval"]["mobile_tray_support"]["config"].copy()
    support.pop("attachment_mode")
    support.pop("compound_manifest", None)
    plate = {
        "type": "DatasetObject",
        "name": "collection_plate",
        "category": "tray",
        "model": "gsxbym",
        "fixed_base": False,
        "visual_only": False,
        "position": [103.0, 100.0, 2.0],
    }
    env = og.Environment(
        configs={
            "env": {
                "action_frequency": 20,
                "rendering_frequency": 20,
                "physics_frequency": 120,
            },
            "scene": {"type": "Scene"},
            "robots": [native_robot_config(robot, {})],
            "task": {"type": "DummyTask"},
            "objects": [*support_objects(support), plate],
        }
    )
    try:
        env.reset()
        receipt = build_compound_asset(
            env.robots[0], env, support, robot["asset_path"], output / "compound"
        )
        (output / "compound/manifest.json").write_text(json.dumps(receipt, indent=2))
    finally:
        og.sim.close()
        og.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["native", "compound"])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument(
        "--robosuite-assets",
        type=Path,
        help="robosuite/models/assets directory for the native build",
    )
    args = parser.parse_args()
    if args.component == "native" and args.robosuite_assets is None:
        parser.error("the native build requires --robosuite-assets")
    output = args.robot_assets.resolve() / "pandaomron"
    output.mkdir(parents=True, exist_ok=True)
    os.environ["WORLDLEAGUE_ROBOT_ASSETS"] = str(output.parent)
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        info = json.loads(Path(device(args.gpu, scratch)).read_text())
        install_gpu_selection(info)
        if args.component == "native":
            native_asset(output, args.robosuite_assets, scratch)
        else:
            compound_asset(output)


if __name__ == "__main__":
    main()
