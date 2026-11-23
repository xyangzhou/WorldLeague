# Contributing to WorldLeague

WorldLeague loads external policies and harnesses by their Python import path. You can develop the integration in your own package and install it in the environment that runs `worldleague`. For a model served from another environment, the continuous runner also accepts a WebSocket endpoint.

| What you want to add | Where to start |
| --- | --- |
| A VLM endpoint using an existing interface | Copy a matching [WorldHarness preset](../configs/vlm) and edit its model, endpoint and API key variable. Select it with `--config my-model.yaml` on `worldleague eval`. A flat [Chat Completions example](../configs/chat.example.yaml) is also available. |
| An R1 policy with its own model client | Pass `--policy my_package:load` to `worldleague eval`. The zero-argument factory returns an object with `act(observation, actions, notes)` and `answer(observation, questions, notes)`. Implement `set_history(frames)` for tasks with a recorded prelude. |
| An R1 harness with its own decision loop | Pass `--harness my_package:play` to `worldleague eval`. The generator keeps its model client and memory across turns. [Runnable harness example](plugins.md#an-external-r1-harness) |
| A continuous VLA or WAM | Pass `--plugin my_package:load` or `--url ws://HOST:PORT` to `worldleague continuous`. The local factory returns a predictor that takes one public observation and returns an action or action chunk. [Predictor example and action layouts](plugins.md#a-wam-or-vla) |

## Add an R1 harness

On each interaction turn, `play(turn)` receives the public observation, available actions, retained notes and any feedback. Yield one decision. The value sent back into the generator is the next turn. Check `turn.questions` after every yield because the budget can expire after an ordinary motion. When `turn.questions is not None`, return an `{"answers": {...}}` mapping keyed by question ID. WorldLeague handles the simulation clock, budget and scoring. [Turn definition and driver](../src/worldleague/plugins.py)

To try the example in the plugin guide, save its generator as `my_harness.py`. From that directory, run the following in the simulator environment. The two files to inspect first are `decision_events.jsonl` for the action outcomes and `result.json` for the episode scores.

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" worldleague eval T04 --harness my_harness:play \
  --seed 0 --gpu 0 --output runs/my-harness/T04
```

## Add a continuous model or interface

If your model already produces executable commands for PandaOmron or R1Pro, select that robot on the command line. The direct interfaces accept 12 and 23 action values per step, respectively. Put checkpoint-specific preprocessing and decoding in your predictor. Python plugins and WebSocket clients receive observations through the same [observation adapter](../src/worldleague/policies/continuous_websocket.py).

For a reusable named preset, follow [the OpenWAM model mapping](../configs/continuous/models/OpenWAM-Alpha-RoboCasa365.yaml). Its `T04` entry selects a policy by name and the first setup in the dataset’s `continuous/T04.yaml` (`setup: 1`). The setup selects the robot and control rate. The [policy YAML](../configs/continuous/policies/OpenWAM-Alpha-RoboCasa365.yaml) specifies the observation format, chunk length and execution horizon. If the interface needs a different sensor encoding or action layout, implement that conversion and document the camera names, array shapes, units and control rate.

## Add a robot

For a new embodiment, start with the [PandaOmron configuration](../configs/continuous/robots/pandaomron.yaml) to see how controllers, camera roles and assets are declared. Add the simulator class name to the [robot type mapping](../src/worldleague/environments/robot_config.py). If your robot uses a custom class, import it during [scene initialization](../src/worldleague/runtime/sim/esi_omnigibson.py) so OmniGibson can construct it.

Connect the robot’s measured state and executable commands to the [continuous environment](../src/worldleague/environments/continuous.py). Add a matching task setup to the benchmark dataset and select it from the model mapping. These files are resolved together by the [episode planner](../src/worldleague/continuous.py).

When opening a pull request, a command for one supported task and a brief description of the expected observation and action make the integration easier to try. For a new robot, include how to obtain its assets.
