"""Select native CUDA and Vulkan devices. Used by worldleague scene."""
import os
from pathlib import Path
def install_gpu_selection(gpu):
    """Set renderer Vulkan ordinal separately from visible CUDA ordinal zero."""
    import isaacsim

    original = isaacsim.SimulationApp

    def isolated_app(config=None, *args, **kwargs):
        config = dict(config or {})
        config.update(active_gpu=gpu["renderer_index"], physics_gpu=0, multi_gpu=False)
        config["extra_args"] = [*config.get("extra_args", []),
            f"--/renderer/activeGpu={gpu['renderer_index']}",
            "--/renderer/multiGpu/enabled=false",
            f"--/renderer/multiGpu/activeGpus={gpu['renderer_index']}"]
        if os.environ.get("OMNIGIBSON_APPDATA_PATH"):
            appdata = Path(os.environ["OMNIGIBSON_APPDATA_PATH"])
            appdata.mkdir(parents=True, exist_ok=True)
            config["extra_args"].extend([
                f"--/app/tokens/cache={appdata / 'kit_cache'}",
                f"--/app/tokens/data={appdata / 'kit_data'}",
                f"--/app/tokens/logs={appdata / 'kit_logs'}",
                "--/plugins/carb.tasking.plugin/threadCount=8",
                "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                "--/exts/omni.physx/numThreads=4"])
        return original(config, *args, **kwargs)

    isaacsim.SimulationApp = isolated_app


def isolate_texture_cache(simulator, appdata):
    """Redirect native startup cleanup to this worker's cache, preserving peers."""
    from types import SimpleNamespace

    original = simulator.shutil
    shared = (Path.home() / ".cache/ov/texturecache").resolve()
    private = Path(appdata).resolve() / "xdg_cache/ov/texturecache"

    def remove(path, *args, **kwargs):
        selected = private if Path(path).resolve() == shared else path
        return original.rmtree(selected, *args, **kwargs)

    simulator.shutil = SimpleNamespace(**{**vars(original), "rmtree": remove})
