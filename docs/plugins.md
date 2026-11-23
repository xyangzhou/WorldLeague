# Connect a harness, VLA or WAM

Install your Python module in the environment that runs `worldleague`, or add its directory to `PYTHONPATH`. Choose the interface that matches the commands your system produces. Both use the benchmark alias and seed supplied on the command line.

## Install and run

Use the OmniGibson simulator environment with its licensed BEHAVIOR assets. From the WorldLeague checkout, install the simulation extras and set the path to the downloaded benchmark. WorldLeague includes the scene builders, robot configurations and continuous policy presets.

```sh
python -m pip install -e '.[simulation]'
export WORLDLEAGUE_BENCHMARK=/path/to/benchmark
export OMNIGIBSON_DATA_PATH=/path/to/behavior
worldleague tasks
worldleague continuous pi0.5 T04 --url ws://127.0.0.1:8000 --gpu 0 --output runs/pi05
```

Start your policy server with its checkpoint before running a continuous episode. For a PandaOmron policy such as OpenWAM, prepare its native and compound assets with the matching robosuite meshes.

```sh
export WORLDLEAGUE_ROBOT_ASSETS="$HOME/.cache/worldleague/robots"
python -m worldleague.prepare_robot native --gpu 0 --robot-assets "$WORLDLEAGUE_ROBOT_ASSETS" --robosuite-assets /path/to/robosuite/models/assets
python -m worldleague.prepare_robot compound --gpu 0 --robot-assets "$WORLDLEAGUE_ROBOT_ASSETS"
worldleague continuous OpenWAM-Alpha-RoboCasa365 T04 --url ws://127.0.0.1:8001 --gpu 0 --output runs/openwam
```

For VLMs, install WorldHarness in the same environment. Set `OPENAI_API_KEY` for `luna-medium` or `sol-medium`, or `AI_GATEWAY_API_KEY` for `deepseek-low` and its two-decision `deepseek-low-smoke` preset. The CLI accepts a preset name or a custom YAML path.

```sh
python -m pip install -e '../WorldHarness[simulation]'
worldleague eval T04 --config luna-medium --gpu 0 --output runs/luna
```

## An external R1 harness

Export one generator function. It receives the first public turn, yields one decision, then receives the next turn as the value of the yield. Keep your model client, planner and memory in generator locals. The runner drives physics, holds the decision budget and captures the final state before it asks for QA.

This complete smoke harness takes one short movement and finishes T04. Save it as `my_harness.py`.

```python
"""Run with worldleague eval T04 --harness my_harness:play --gpu 0 --output runs/harness."""

def play(turn):
    turn = yield {"action": "BASE_FWD", "distance_m": 0.05, "note": "Started exploring."}
    while turn.questions is None:
        turn = yield {"action": "DONE"}
    return {"answers": {question["id"]: "Yes" for question in turn.questions}}
```

```bash
PYTHONPATH="$PWD:${PYTHONPATH:-}" worldleague eval T04 --harness my_harness:play --gpu 0 --seed 0 --output runs/harness
```

- Optionally report provider latency and token usage with `turn.record_request(elapsed_seconds, usage)` once per request. The runner supplies the phase and action step.
- Read the instruction from `turn.observation.task_text`, mounted images from `turn.observation.rgb_images()`, and feedback from `turn.observation.last_action_feedback`. The observation also exposes the initial overview, snapshot ID and action history. `turn.history` contains public prelude frames on tasks with a recorded demonstration. Each frame includes a PNG in `png_base64`.
- `turn.actions` gives the executor's primitive names. `turn.action_mode` and `turn.max_decisions` give the evaluation protocol. In parameterized mode supply `distance_m` for translation or `angle_deg` for rotation when the command takes a magnitude. In fixed-repeat mode supply `repeat` of 1, 2, 4 or 8. A yielded decision counts once even if it requests several repetitions.
- A decision may include `arrived_rooms`, `observed_objects`, `phase_1_complete`, an interaction `answer` where the task requests one, and a `note`. The runner returns saved notes in `turn.notes` in order, retaining up to 128 characters per decision.
- Yield `{"action": "DONE"}` to finish interaction or advance a two-phase survey. On T38, yield `{"action":"DONE","survey_complete":true,"answer":null}` after inspecting all nine items and read the delayed request on the next turn. An incomplete survey or invalid action consumes one decision. Its explanation appears in `turn.feedback` (a policy object can implement `set_feedback(message)`). Check `turn.questions` after **every** yield, including after an ordinary action. Budget exhaustion also produces a terminal turn. Return one JSON-compatible `{"answers": {...}}` mapping on that turn. Each question supplies its ID, text, schema and time.
- Physics advances while the harness chooses an action. Each R1 decision counts once, including any historical lookups within it. The built-in chat client can recall an RGB view sent on an earlier decision by its one-based decision number and camera name. `--policy module:factory` is also available for clients that already implement `act`, `answer` and, on prelude tasks, `set_history`.

## A WAM or VLA

For a new policy, select `R1Pro` or `PandaOmron` to send executable actions directly. To reproduce a trained policy interface, select its named preset. Export a zero-argument factory that loads your checkpoint or opens your model client once and returns a callable. Its entire per-request contract is `predict(observation) -> actions`. If the callable has a `close()` method, the runner calls it during cleanup.

This smoke plugin emits a single zero command for the PandaOmron interface. Save it as `my_wam.py` and replace the predictor body with your inference and action decoder.

```python
"""Run with worldleague continuous PandaOmron T04 --plugin my_wam:load --gpu 0 --output runs/wam."""
import numpy as np

def load():
    def predict(observation):
        return np.zeros((1, 12), dtype=np.float32)
    return predict
```

```bash
PYTHONPATH="$PWD:${PYTHONPATH:-}" worldleague continuous PandaOmron T04 --plugin my_wam:load --gpu 0 --seed 0 --output runs/wam
```

| Preset | Public arrays and prompt | Executable action width | Max chunk / execution horizon |
| --- | --- | ---: | ---: |
| R1Pro | Same public joint state, RGB views and prompt as pi0.5 | 23 | 32 / 32, direct |
| PandaOmron | 1 PandaOmron history frame | 12 | 32 / 32, direct |
| pi0.5 | `state` (23,), `images` with head and both wrists, 224 × 224 RGB, `prompt` | 23 | 32 / 20 |
| GR00T-N1.5-RoboCasa365 | 1 PandaOmron history frame | 12 | 16 / 16 |
| Xiaomi-XR-1-RoboCasa365 | 4 PandaOmron history frames | 12 | 16 / 16 |
| Awomo-0.5I-RoboCasa365 | 5 PandaOmron history frames | 12 | 16 / 16 |
| OpenWAM-Alpha-RoboCasa365 | 1 PandaOmron history frame | 12 | 32 / 32 |

PandaOmron supplies a string `prompt`, `state` of shape (history, 14), `images` keyed by `agentview_left`, `agentview_right` and `wrist` in time-major RGB order, plus `history_control_steps` and `history_simulation_times_s`. The 12 executable channels are six arm controls, gripper close, three base commands, trunk and control mode. Decode any checkpoint-specific output, padding or predicted video into this action space inside the plugin. R1Pro uses three base, four trunk, seven left-arm, one left-gripper, seven right-arm and one right-gripper channels. The named pi0.5 preset applies its existing chunk resampling and gripper mapping after prediction.

Return a finite real numeric NumPy array or list of shape (steps, width), or (width,) for one action. The runner executes up to the preset horizon and remaining control budget before requesting another prediction. Image conversion, observation history, control timing, robot setup, chunk processing and physical scoring are shared with the WebSocket path. On T38 the public delivery request is appended to the prompt once the native evaluator accepts the survey. On T38 all interfaces allow at most half the episode budget (rounded down) for the survey. The last permitted decision or control can qualify. If the survey is still incomplete, the episode ends and the scorer uses the recorded physical evidence. A predicted action chunk stops at the survey deadline so the policy receives the delivery request before its next prediction. Native physics continues while VLA or WAM inference is pending. These hold steps are logged separately and do not consume the policy control budget. T52 has no published continuous setup.

### A model in another environment

Use `--url ws://HOST:PORT` in place of `--plugin`. The server sends a binary MessagePack metadata mapping on connect. GR00T sends a one-way `{"reset": true}` client message before inference. Each inference message contains the same public fields as the Python predictor and receives `{"actions": array}`. Numeric arrays use MessagePack mappings with binary keys `__ndarray__` (true), `dtype` (NumPy dtype string), `shape` and `data` (C-order bytes). Per-preset reset modes are recorded in the policy YAML. The server owns model-specific preprocessing and decoding. A custom transport can also be wrapped in the Python factory.
