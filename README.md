# policies

policies trains neural network policies with PyTorch and exports them as ONNX artifacts that run
on NumPy and ONNX Runtime alone. A YAML spec declares each policy's named input and output parts,
its bookkeeping and its model, all built from registries that plugins extend.

## Install

```bash
uv sync --extra cpu --extra gym     # training on CPU, with Gymnasium
uv sync --extra cu126 --extra gym   # training with CUDA 12.6
uv sync                             # inference only: NumPy and ONNX Runtime, no PyTorch
```

| Extra         | Adds                                                             |
| ------------- | ---------------------------------------------------------------- |
| `cpu`         | PyTorch (CPU wheels) and ONNX export: models, training, export   |
| `cu126`       | The same with PyTorch for CUDA 12.6. Conflicts with `cpu`        |
| `hub`         | `huggingface_hub`, for `hf://` artifact registries               |
| `gym`         | Gymnasium, for the vector environment adapter                    |
| `tensorboard` | TensorBoard, for the `tensorboard` hook                          |

From another project, depend on the core for inference and on an extra for training:

```toml
[project]
dependencies = ["policies"]

[dependency-groups]
dev = ["policies[cpu,gym]"]

[tool.uv.sources]
policies = { git = "https://github.com/Pequi-RCSS3D/policies" }
```

Dependents must configure the PyTorch index themselves, as the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) shows.

## Quickstart

A run is one YAML file:

<!-- test: cartpole.yaml -->
```yaml
policy:
  name: cartpole
  inputs:
    parts:
      - {name: observation, source: observation, shape: [4], normalize: true}
  outputs:
    parts:
      - {name: action, kind: {type: categorical, choices: [left, right]}}
  model:
    block: {type: mlp, hidden: [64, 64]}
learner:
  rollout: {length: 16}
trainer:
  iterations: 10  # a smoke run, raise it to train for real
```

Train and export:

<!-- test: train -->
```python
import gymnasium as gym
from gymnasium.vector import AutoresetMode

from policies.training.envs.gymnasium import GymnasiumEnvironment
from policies.training.loop import Trainer

vector = gym.make_vec(
    "CartPole-v1",
    16,
    vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
)
env = GymnasiumEnvironment(vector, seed=0)
trainer = Trainer.from_yaml("cartpole.yaml", env, "runs/cartpole")
trainer.run()  # resumes from the latest checkpoint in the run directory
manifest = trainer.export()  # runs/cartpole/export
```

Run the artifact without PyTorch:

<!-- test: deploy -->
```python
import gymnasium as gym

from policies.deploy.inference import Policy

policy = Policy.load("runs/cartpole/export")
env = gym.make("CartPole-v1")
observation, _ = env.reset(seed=0)
policy.reset()
done = False
while not done:
    action = policy.step({"observation": observation})["action"]  # index into `choices`
    observation, reward, terminated, truncated, _ = env.step(int(action))
    done = terminated or truncated
```

The same config in Python, for `Trainer(config, env, run_directory)`:

<!-- test: config -->
```python
from policies.learning import LearnerConfig, RolloutConfig
from policies.spec import PolicySpec
from policies.training.loop import RunConfig, TrainerConfig

observation = {"name": "observation", "source": "observation", "shape": [4], "normalize": True}
action = {"name": "action", "kind": {"type": "categorical", "choices": ["left", "right"]}}
spec = PolicySpec.model_validate(
    {
        "name": "cartpole",
        "inputs": {"parts": [observation]},
        "outputs": {"parts": [action]},
        "model": {"block": {"type": "mlp", "hidden": [64, 64]}},
    }
)
config = RunConfig(
    policy=spec,
    learner=LearnerConfig(rollout=RolloutConfig(length=16)),
    trainer=TrainerConfig(iterations=10),
)
```

## Policy spec

`spec.fingerprint()` is the SHA-256 of the spec's canonical form, with every default filled in.
It identifies the contract between training and deployment (inputs, outputs, bookkeeping and
architecture), not the weights. `pm components` lists every installed component with its fields
and defaults.

### Inputs

```yaml
inputs:
  parts:
    - {name: positions, source: history.joint_position, shape: [3, 12], normalize: true}
    - name: velocity
      source: joint_velocity
      shape: [12]
      transforms: [{type: scale, factor: 0.05}]
```

| Field        | Meaning                                                                       |
| ------------ | ----------------------------------------------------------------------------- |
| `name`       | Lowercase identifier                                                          |
| `source`     | An observation key, or a bookkeeping slot such as `history.joint_position`    |
| `shape`      | Per sample, batch dimensions come in front. `[]` is a scalar                  |
| `dtype`      | Any real array API dtype, `float32` by default                                |
| `transforms` | Applied in order, on any array namespace                                      |
| `normalize`  | Standardize by running mean and variance, updated while training and exported |
| `privileged` | Only for `learner.critic_inputs`. Policy inputs must be observable            |

Transforms: `scale {factor}`, `offset {value}`, `clip {low, high}` and `cast {dtype}`.

### Outputs

```yaml
outputs:
  parts:
    - name: joints
      kind: {type: continuous, size: 12, low: -1, high: 1}
      distribution: {initial_std: 0.3}
    - {name: skill, kind: {type: categorical, choices: [walk, kick, stand]}}
```

| Kind                | Config                           | Decodes to                | Default distribution |
| ------------------- | -------------------------------- | ------------------------- | -------------------- |
| `continuous`        | `size` or `shape`, `low`, `high` | Values, clipped to bounds | `normal`             |
| `scalar`            | `low`, `high`                    | One value, clipped        | `normal`             |
| `binary`            | `size` or `shape`                | Booleans                  | `bernoulli`          |
| `categorical`       | `choices`                        | Index of the choice       | `categorical`        |
| `multi_categorical` | `choices`, a list per element    | Index per element         | `multi_categorical`  |

Distributions: `normal` (learned state-independent log standard deviation, `initial_std`),
`normal_state_std` (the model outputs the standard deviation too, `initial_std`, `min_std`),
`bernoulli`, `categorical` and `multi_categorical`. A `distribution` without `type`, or with the
default's, updates the default. Another `type` replaces it. Training samples, inference takes the
mode.

### Bookkeeping

Trackers expose state as slots that inputs read like observations, and reset with the episode.
They run on PyTorch while training and on NumPy at inference. Sources are observation keys or
output parts.

| Tracker    | Config                                   | Slot               | Value                                                                            |
| ---------- | ---------------------------------------- | ------------------ | -------------------------------------------------------------------------------- |
| `history`  | `source, length`                         | `history.<source>` | The last `length` values, oldest first                                           |
| `last`     | `source`                                 | `last.<source>`    | The output part's previous action, zeros after a reset                           |
| `timer`    | `name, dt`                               | `timer.<name>`     | Steps since the reset times `dt`, shape `[1]`                                    |
| `legendre` | `source, window, order, dt, name, reset` | `legendre.<name>`  | Last `window` seconds as `order` Legendre coefficients (LMU), `[*source, order]` |

`legendre` gives memory over a time window without recurrent gradients: a fixed linear
recurrence, `reset: zero` (default) or `hold`, which fills the window with the next input.

### Model

Without `type`, the model is `parts`: one encoder per input part, concatenated in input order
into one block, which feeds one head per output part. Other architectures come from the `models`
registry as `model: {type: my_package.model, ...}`.

```yaml
model:
  encoder: {type: flatten}                       # default for parts not in `encoders`
  encoders: {positions: {type: mlp, hidden: [64]}}
  block: {type: transformer, memory: 4, causal: true}
  head: {type: linear}                           # default for parts not in `heads`
  heads: {skill: {type: mlp, hidden: [64]}}
```

| Encoder   | Config               | Does                                                   |
| --------- | -------------------- | ------------------------------------------------------ |
| `flatten` |                      | Flattens the part                                      |
| `mlp`     | `hidden, activation` | Flattens, then an MLP                                  |
| `block`   | `block`              | Flattens, then any block, for example a per-part `gru` |

| Head     | Config                                    |
| -------- | ----------------------------------------- |
| `linear` | `initial_scale: 0.01`                     |
| `mlp`    | `hidden: [64], activation, initial_scale` |

| Block         | Config                                                                    | State                        |
| ------------- | ------------------------------------------------------------------------- | ---------------------------- |
| `mlp`         | `hidden: [256, 128], activation: elu`                                     | None                         |
| `gru`, `lstm` | `hidden_size: 128, layers: 1, after: [], activation: elu`                 | `h`, and `c` for `lstm`      |
| `transformer` | `width: 64, heads: 4, layers: 2, feedforward: 128, activation: gelu`      | `memory`, `filled` if memory |
|               | `tokens: parts, memory: 0, causal: false, positions: true, pooling: mean` |                              |
| `mixture`     | `experts: 4, expert: {type: mlp}, gate: [], top_k: null, activation: elu` | Each expert's                |
| `sequential`  | `blocks: [...]`, each fed the previous one's output                       | Each block's                 |

- `after`: MLP sizes applied to a recurrent or transformer block's output.
- `tokens`: `parts` (one token per encoded input part) or `input` (one token).
- `memory`: past steps kept as recurrent state to attend to. `causal` hides later steps.
- `positions`: learned embeddings per step and token.
- `pooling`: `mean`, `max`, `first`, `last` or `summary` (a learned extra token).
- `mixture` gates its experts with a softmax, and `top_k` mixes only the best per sample. It
  records the auxiliary loss `balance` (Switch Transformer load balancing).

Recurrent models train on whole `[time, batch]` environment sequences, others on shuffled steps.
`gru`, `lstm`, a transformer with `memory`, `mixture` and `sequential` run a sequence natively,
other blocks step through it.

Activations: `elu`, `relu`, `gelu`, `silu`, `tanh`.

## Training

### Learner

```yaml
learner:
  objectives:
    ppo: {entropy_weight: 0.005}
    smooth: {type: action_smoothness, weight: 0.001}
    expert: {type: distillation, teacher: walker, weight: 0.5}
  teachers:
    walker: runs/walker/export
  optimizer: {type: adamw, lr: 3.0e-4}
  schedule: {type: adaptive_kl, target_kl: 0.01}
  rollout: {length: 24, epochs: 5, minibatches: 4, gamma: 0.99, gae_lambda: 0.95}
```

| Field             | Default            | Meaning                                                            |
| ----------------- | ------------------ | ------------------------------------------------------------------ |
| `objectives`      | `{ppo: {}}`        | Instance name to config. `type` defaults to the instance name      |
| `optimizer`       | `{type: adamw}`    | `adam` or `adamw`, with `lr`, `weight_decay`, `betas`              |
| `schedule`        | `{type: constant}` | Learning rate schedule                                             |
| `rollout`         | See above          | Steps per update, epochs, minibatches, discount and GAE lambda     |
| `initial_std`     | `1.0`              | Default for distributions that learn a standard deviation          |
| `max_grad_norm`   | `1.0`              | Gradient clipping                                                  |
| `normalizer_clip` | `5.0`              | Normalized inputs are clipped to this many standard deviations     |
| `auxiliary`       | `{balance: 0.01}`  | Weights of losses models record, by the last segment of their name |
| `teachers`        | `{}`               | Teacher name to artifact path or registry reference                |
| `critic_inputs`   | The policy inputs  | May include privileged parts                                       |
| `critic_model`    | `{type: parts}`    | Any architecture                                                   |

| Objective           | Settings                                         | Does                                                        |
| ------------------- | ------------------------------------------------ | ----------------------------------------------------------- |
| `ppo`               | `clip, value_weight, value_clip, entropy_weight` | Clipped surrogate and value losses, entropy bonus           |
| `imitation`         | `targets: {output part: observation key}`        | Clones targets the environment provides                     |
| `distillation`      | `teacher`                                        | DAgger: the student acts, the teacher labels                |
| `kl_to_reference`   |                                                  | KL to a frozen copy of the actor, taken at the first update |
| `action_smoothness` |                                                  | Penalizes changing actions between steps                    |

Every objective takes a `weight`. The critic exists only when an objective needs values, as
`ppo` does. An optimizer or schedule without `type` is the default one. Both validate with the
config.

Schedules: `constant`, `cosine {iterations, min_lr}` and `adaptive_kl {target_kl, factor,
min_lr, max_lr, metric}`, which follows the `kl` of the one `ppo` objective unless `metric` is
set.

A teacher is an artifact path or a registry reference `{root, id}` or `{root, name, tag}`, run
batched on ONNX Runtime with its own bookkeeping. `Trainer(..., teachers={"walker": callable})`
also takes any callable `(observation, reset) -> outputs` in action space.

`Learner.fit` clones recorded data offline through each part's imitation loss:

```python
from policies.data import InMemoryDataset
from policies.deploy.export import export_artifact
from policies.learning import Learner, LearnerConfig

learner = Learner(spec, LearnerConfig(objectives={}), num_envs=1)  # no objectives, no critic
dataset = InMemoryDataset(inputs, targets, batch_size=256)  # `[steps, ...]` arrays per part
metrics = learner.fit(dataset, epochs=10)
export_artifact(learner, "exports/cloned")
```

- Inputs are keyed by input part, before normalization.
- Targets are keyed by output part, in action space: means for continuous parts, indices for
  categorical ones.
- Recurrent models need `reset` flags and a `sequence_length`.
- Export needs the shapes of the observations trackers read. `fit` records those that
  untransformed parts copy. For the rest, export
  `ExportSource(learner.actor, {"joint_position": (12,)})`.

### Trainer

| Field              | Default                            | Meaning                                                |
| ------------------ | ---------------------------------- | ------------------------------------------------------ |
| `iterations`       | Required                           | Updates to run                                         |
| `checkpoint_every` | `100`                              | Iterations between checkpoints                         |
| `keep_checkpoints` | `3`                                | Older checkpoints are deleted                          |
| `seed`             | `0`                                | Seeds Python, NumPy, PyTorch and the wrappers          |
| `device`           | `auto`                             | CUDA when available, else the CPU, or any torch device |
| `hooks`            | `[{type: console}, {type: jsonl}]` | Event receivers                                        |
| `wrappers`         | `[]`                               | Environment wrappers, the first wraps the environment  |
| `tunable`          | `[optimizer.lr]`                   | Paths that may change while running                    |
| `bounds`           | `{}`                               | `{path: [low, high]}` for tunable numbers              |

An environment is batched and auto-resetting:

```python
class Environment(Protocol):
    num_envs: int

    def reset(self) -> dict[str, Tensor]: ...
    def step(
        self, outputs: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, dict[str, Any]]: ...
```

`step` returns observations, rewards, dones and an `info` that may carry `truncated` (to
bootstrap time limits) and `final_observation` (the observation before the reset). Outputs
arrive decoded, on the observations' device. `GymnasiumEnvironment` takes `SAME_STEP` or
`DISABLED` autoreset. Dict spaces map to parts by key, other spaces become the parts
`observation` and `action`.

| Wrapper          | Config                                     | Simulates                                |
| ---------------- | ------------------------------------------ | ---------------------------------------- |
| `action_delay`   | `cycles`                                   | Latency: outputs apply late              |
| `dropped_cycles` | `rate, seed`                               | Late messages: outputs repeat            |
| `hold`           | `parts, every, phase, random_phase, seed`  | Slower sensors: parts refresh less often |
| `noise`          | `scales: {part: standard deviation}, seed` | Imprecise sensors                        |
| `quantize`       | `steps: {part: step}, rounding`            | Limited sensor resolution                |

Custom wrappers subclass `Wrapper` or `ObservationWrapper` from `policies.training.envs`.

A run directory holds `config.yaml`, `checkpoints/`, `control.jsonl`, the hooks' output (such as
`events.jsonl`) and, after `trainer.export()`, `export/`.

- `trainer.run(checkpoint=None)` resumes from `checkpoint`, else from the run's latest.
- Values in the checkpoint, such as a learning rate tuned live, override the YAML.
- Control messages newer than the checkpoint, including those sent while the run was down, apply
  at the first iteration.
- `Trainer(..., agent=...)` runs any `Agent` instead of the default `Learner`.

`Learner.from_checkpoint(path, num_envs=1, device="cpu")` rebuilds a learner, spec and config
included, with `training` off: it acts with each part's mode and records nothing. Checkpoints are
pickles, so load only trusted ones.

<!-- test: checkpoint -->
```python
from policies.learning import Learner

learner = Learner.from_checkpoint(trainer.checkpoints[-1], num_envs=16)
outputs = learner.act(env.reset())
```

### Live control

```bash
pm set runs/cartpole optimizer.lr=1e-4 objectives.ppo.entropy_weight=0.01
pm pause runs/cartpole
pm resume runs/cartpole
pm checkpoint runs/cartpole
pm stop runs/cartpole
```

Messages go through `control.jsonl` and apply between iterations. The target must contain
`config.yaml`. `pm set` parses values as JSON and checks paths and bounds before sending. Tunable
paths are `optimizer.lr` and `objectives.<instance>.<setting>`. Only those in `trainer.tunable`
change, within `trainer.bounds` and keeping their type. Refused changes become `error` events.

Custom commands are functions in `trainer.commands`, sent with `pm send RUN NAME` or:

```python
from pathlib import Path

from policies.training.control import send_command

trainer.commands["snapshot"] = lambda: trainer.export(Path("runs/cartpole/snapshot"))
send_command(Path("runs/cartpole"), "snapshot")
```

### Events and hooks

| Event        | Data                                               |
| ------------ | -------------------------------------------------- |
| `start`      | `resumed`                                          |
| `update`     | The update's metrics, such as `ppo/kl` and `lr`    |
| `iteration`  | `seconds` and, once episodes end, `episode_return` |
| `evaluation` | What an evaluation function returns                |
| `checkpoint` | `path`                                             |
| `control`    | `changes` as `{path: [old, new]}`, or `command`    |
| `error`      | `error`, or `control` for a refused message        |
| `end`        | `stopped`                                          |

Hooks: `console {every: 10, max_metrics: 6}`, `jsonl {path: events.jsonl, every: 1}`,
`tensorboard {every}` (needs the `tensorboard` extra) and `wandb {project, entity, name, every}`
(needs `wandb`). `every` thins out `iteration` and `update` events. A hook that raises is
disabled with a warning.

```python
trainer.hooks.add("print", lambda event: print(event.kind, event.step, event.data), every=10)
trainer.add_evaluation(lambda trainer: {"score": score(trainer)}, every=50)
trainer.emit("milestone", {"reached": 100})  # custom kinds are plain strings
```

`trainer.evaluation` keeps the latest value of each evaluation key, and `trainer.export()`
records them as `evaluation/<key>` metrics.

The library never configures logging. Until the application does, its loggers show INFO and
above on stderr. `policies.core.logs.configure(level="INFO", json=False)` sets up structlog, as
`pm` does. Levels are names in any case or numbers. Times are UTC: `HH:MM:SSZ` on the console,
ISO in JSON logs and manifests, Unix seconds in `events.jsonl`.

## Deployment

### Export and run

`trainer.export(directory=None, parents=None)` and
`export_artifact(source, directory, *, metrics, parents, run)` write an artifact directory and
return its `Manifest`. `metrics` are scores to record. `parents` are the ids of the artifacts it
derives from, by default a learner's artifact teachers. The artifact keeps only the observations
its bookkeeping reads. Without a `Learner`, wrap an actor as
`ExportSource(actor, observation_shapes, observation_dtypes)`.

| File            | Holds                                                                       |
| --------------- | --------------------------------------------------------------------------- |
| `policy.onnx`   | Input parts and state in, raw outputs and next state out, metadata embedded |
| `weights.pt`    | The actor's settings and state, to rebuild it exactly                       |
| `manifest.json` | Id, name, fingerprint, spec, requirements, metrics, parents, files and run  |

```python
policy = Policy.load("runs/cartpole/export", threads=1)
policy.reset()                          # new episode: bookkeeping and recurrent state
outputs = policy.step(observation)      # one sample, keyed like training observations
step = policy.act(observation)          # PolicyStep(outputs, actions, extras)
raw, state = policy.run(parts, state)   # a batch of input parts, no bookkeeping
```

`act` also returns the modes before decoding and the extras of kinds that have them.
`policy.observation_keys` names exactly the observations `step` reads. `policy.kinds` and
`policy.kind_types` give each output part's kind and its registered name.

`validate_artifact(path)`, or `pm validate PATH`, checks the metadata, the manifest and, with
`weights.pt` and PyTorch, that ONNX Runtime matches PyTorch. The returned `Validation` says
whether parity ran in `parity`, and why not in `skipped`.

### Artifact registry

```python
from policies.deploy.registry import ArtifactRegistry

registry = ArtifactRegistry("hf://my-org/policies")  # or a local path, s3://, any fsspec URL
manifest = registry.register("runs/cartpole/export")
registry.tag(manifest.id, "competition")
local = registry.fetch_tagged("cartpole", "competition", "artifacts")  # artifacts/<id>
registry.untag("cartpole", "competition")
```

- Artifacts are immutable under `<root>/<id>/`, with ids `<name>-<first 12 fingerprint digits>`.
  Registering identical content again does nothing.
- Tags in `<root>/tags.json` map a tag and a policy name to an id, and can move.
- `hf://` roots need the `hub` extra and write one commit per change. Other protocols go through
  fsspec and need its backend, such as `s3fs`.
- `requirements` records the version of every distribution whose components the spec uses.
  Loading names the plugin to install when a component is unknown. Other versions only warn,
  since the fingerprint guards behavior.

### The `pm` command

| Command                                        | Does                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `pm set RUN key=value ...`                     | Changes tunable values of a run                              |
| `pm pause`, `resume`, `checkpoint`, `stop RUN` | Sends a command to a run                                     |
| `pm send RUN NAME`                             | Sends a custom command to a run                              |
| `pm validate PATH`                             | Checks an artifact and says whether parity ran               |
| `pm list [--name NAME] [--tag TAG]`            | Lists artifacts with their tags and metrics                  |
| `pm show ID`                                   | Shows a manifest                                             |
| `pm diff A B`                                  | Spec, requirement, metric and parent paths that differ       |
| `pm fetch ID\|NAME:TAG DEST`                   | Copies an artifact to `DEST/<id>`, unless already there      |
| `pm tag ID TAG`                                | Points a tag at an artifact                                  |
| `pm untag NAME TAG`                            | Removes a tag from a policy name                             |
| `pm components [REGISTRY]`                     | Registered components, their origin and config fields        |

Registry commands take `--root`, or the `POLICIES_REGISTRY` environment variable. `diff` matches
list items, such as parts, by name.

## Extending

A component is a config class (a subclass of `policies.core.config.Config`) and a factory called
as `factory(config, **context)`:

| Registry        | Import from                        | Context                                | Builds                                                      |
| --------------- | ---------------------------------- | -------------------------------------- | ----------------------------------------------------------- |
| `transforms`    | `policies.io.inputs`               | None                                   | `transform(x, xp) -> y`                                     |
| `kinds`         | `policies.io.outputs`              | `initial_std` while training           | `Kind`: `shape`, `distribution`, `decode(action, xp)`       |
| `distributions` | `policies.io.outputs`              | `shape`, `initial_std` while training  | `Distribution`                                              |
| `trackers`      | `policies.io.bookkeeping`          | `context`, a `TrackerContext`          | `Tracker`: `slots`, `sources`, `reset`, `observe`, `commit` |
| `models`        | `policies.spec.model`              | `in_shapes`, `out_sizes`               | `Model`                                                     |
| `encoders`      | `policies.spec.model`              | `shape`                                | `Encoder`                                                   |
| `blocks`        | `policies.spec.model`              | `in_sizes`                             | `Block`                                                     |
| `heads`         | `policies.spec.model`              | `in_size`, `out_size`                  | `Head`, or any module                                       |
| `activations`   | `policies.spec.model`              | None                                   | `nn.Module`                                                 |
| `objectives`    | `policies.learning`                | `teachers`, `device`                   | `Objective`, or a `BaseObjective` subclass                  |
| `optimizers`    | `policies.learning`                | `params`                               | `torch.optim.Optimizer`                                     |
| `schedules`     | `policies.learning`                | `base_lr`, `objectives`                | `Schedule`: `step(lr, iteration, metrics)`                  |
| `wrappers`      | `policies.training.envs`           | `env`, `seed`                          | `Environment`, usually a `Wrapper`                          |
| `hooks`         | `policies.training.monitoring`     | `run_directory`, `trainer`, `spec`     | `Hook`: `every`, `__call__(event)`                          |
| `storages`      | `policies.deploy.registry.storage` | Backend options. Named by URL protocol | `Storage`                                                   |

- Each contract is a protocol or base class beside its registry and documents shapes and state.
  `Model`, `Block`, `Encoder` and `Head` are in `policies.models`, which also re-exports their
  registries.
- Accept `**_` for context you do not use.
- Namespace names, as in `my_package.mirror`.
- A kind may have `extras(action, xp)`, named per-step values beside the decoded one, such as
  controller gains, which `Policy.act` returns.
- A tracker's `sources` must name every observation and output it reads, since artifacts record
  only those observations.

A transform, usable as `transforms: [{type: my_package.mirror, axis: 0}]`:

<!-- test: transform -->
```python
from typing import Any

from policies.core.config import Config
from policies.io.inputs import Transform, transforms


class MirrorConfig(Config):
    axis: int = -1


@transforms.register("my_package.mirror", MirrorConfig)
def mirror(config: MirrorConfig, **_: Any) -> Transform:
    return lambda x, xp: xp.flip(x, axis=config.axis)
```

A block, usable as `model: {block: {type: my_package.residual, hidden: 128}}`:

<!-- test: block -->
```python
from torch import Tensor, nn

from policies.core.config import Config
from policies.models import Block, State, blocks


class ResidualConfig(Config):
    hidden: int = 64


@blocks.register("my_package.residual", ResidualConfig)
class Residual(Block):
    def __init__(self, config: ResidualConfig, in_sizes: tuple[int, ...]) -> None:
        super().__init__()
        size = sum(in_sizes)
        self.net = nn.Sequential(
            nn.Linear(size, config.hidden), nn.ELU(), nn.Linear(config.hidden, size)
        )
        self.out_size = size
        self.state_shapes = {}  # stateless

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        return x + self.net(x), {}
```

Models are `forward(parts, state) -> (raw outputs, next state)`, with recurrent state passed
explicitly. Export traces `forward`. Training runs `forward_sequence(x, state, reset)`, which
steps `forward` by default. Override it to run a `[time, batch]` sequence natively. It must match
stepping `forward`, with `reset` zeroing the state before a step.

- `register_lazy(name, config_class, "module:attribute")` imports the factory on first build, so
  configs validate without its dependencies, as the built-in torch components do.
- `replace=True` overrides a name on purpose.
- `with registry.scoped():` undoes registrations on exit, for tests.
- `policies.models.record(self, name, value)` records an auxiliary loss, weighted by
  `learner.auxiliary`.

### Plugins

```toml
[project.entry-points."policies.plugins"]
my_package = "my_package.components"

[project.entry-points."policies.registries"]
formations = "my_package.formations:formations"
```

`policies.plugins` names a module that registers components on import. Plugins load on the
first lookup of an unknown name, or on listing, so specs use plugin components without imports.
Installed plugins cannot replace names.

`policies.registries` lists registries a plugin defines itself, such as
`Registry[T]("formation")`, so `pm components` finds them. Annotate config fields as
`Annotated[dict[str, Any], formations]` to validate `{type: ...}` configs canonically and count
them in artifact requirements.

rcss3d, by the Pequi Mecânico RoboCup SS3D team, is the reference domain plugin. It adds the
RoboCup Soccer Simulation 3D robots as output kinds and transforms, the server protocol and team
packaging.

## Development

```bash
uv sync --extra cpu
uv run pre-commit install
uv run --locked pytest -q
```

`tests/test_readme.py` runs the examples marked in this README. See the organization's
[CONTRIBUTING.md](https://github.com/Pequi-RCSS3D/.github/blob/main/CONTRIBUTING.md) for how to
work here.
