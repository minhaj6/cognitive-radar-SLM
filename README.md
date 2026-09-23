[![arXiv](https://img.shields.io/badge/arXiv-2608.11596-b31b1b.svg)](https://arxiv.org/abs/2608.11596)

# Cognitive Radar

An SLM-driven radar agent. A local language model (via [Ollama](https://ollama.com))
reasons about the array-processing scenario, then calls Python DSP tools
(beamforming, DOA estimation, analysis) to produce beampattern plots and
numerical results.

```
+------------+  natural language  +----------------------+  function calls  +----------------------+
|  Module C  | -----------------> |  Module B: Reasoner  | ---------------> |  Module A: Physics   |
|   (CLI)    | <----------------- |  (Ollama, tool use)  | <--------------- |  (NumPy / SciPy)     |
+------------+  explanation +     +----------------------+  numerical       +----------------------+
                plot filenames                              results + plots
```
## Citation

The paper associated with this code-base has been accepted at IEEE MLSP 2026 Conference 🎉! 

If you find this repository or our [paper](https://arxiv.org/abs/2608.11596) useful, please consider citing it:
```
@article{ahmad2026small,
  title={Small Language Model enabled Autonomous agent for Language-Conditioned Cognitive Radar},
  author={Ahmad, Minhaj Uddin and Zaman, Zakia and Sun, Shunqiao and Rahman, Mizanur},
  journal={arXiv preprint arXiv:2608.11596},
  year={2026}
}
```

## Layout

```
term-project/
+-- config.yaml              # model + array default configurations
+-- requirements.txt
+-- src/
|   +-- cli.py               # Module C: CLI entrypoint (run via `python -m src.cli ...`)
|   +-- agent/               # Module B: Ollama wrapper, system prompt, tool specs
|   +-- physics/             # Module A: simulator, DSP (beamforming, DOA, etc.), analysis
+-- tests/                   # tests using `pytest`
+-- output/                  # runtime logs
    +-- run-YYYYMMDD-HHMMSS/
        +-- plots/*.png      # beampatterns
        +-- logs/
            +-- query.txt        # verbatim user command
            +-- transcript.json  # full chain-of-thought + tool calls
```

## Prerequisites

1. **Python 3.12** (a `.python-version` file pins this).
2. **uv** (fast Python package/env manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Plain `python -m venv` + `pip` also works; see the fallback below.
3. **Ollama** running locally with a tool-capable model pulled:
   ```bash
   ollama serve &                 # start the daemon if not already running
   ollama pull gemma4:latest      # or any other tool-capable model
   ```
   Verify the model advertises the `tools` capability: `ollama show <model>` --
   the "Capabilities" section must list `tools`.

## Setup

From the `term-project/` directory:

```bash
# 1. Create the venv (Python 3.12). Uses uv (recommended).
uv venv --python 3.12 .venv

# 2. Install dependencies.
uv pip install -r requirements.txt

# 3. Activate the env for interactive work.
source .venv/bin/activate
```

**If you don't like `uv` for some reason**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the project

Always run as a module from the project root so package imports resolve:

```bash
source .venv/bin/activate

# Verify the Ollama host and model are reachable.
python -m src.cli --ping "ignored when --ping is set"

# Scenario 1: The Sidelobe Negotiator.
python -m src.cli "I need to scan towards 30 degree, but there is clutter everywhere. I need really
low sidelobes, at least -30dB."

# Scenario 2: The Adaptive Survivor.
python -m src.cli \
  "We are tracking a target at 10 degree. Suddenly, a high-power jammer appears at -20 degree"

# Scenario 3: The Low-Snapshot Detective.
python -m src.cli \
  "We only have 10 snapshots of data. Find the directions of arrival for two closely spaced targets."
```

### Flags

| Flag | Meaning |
|------|---------|
| `--show` | After the run finishes, open every generated figure in a GUI window (blocks until closed). Without this flag, plots are saved to disk only. |
| `--model NAME` | Override the Ollama model (default comes from `config.yaml`). |
| `--host URL`   | Override the Ollama host URL. |
| `--config PATH` | Alternate config file. |
| `--max-steps N` | Cap the tool-calling loop (default 8). |
| `--ping` | Check Ollama reachability and exit. |

### Output

Each run creates `output/run-YYYYMMDD-HHMMSS/` containing:

* `plots/*.png` -- every beampattern or pseudo-spectrum the agent generated.
* `logs/query.txt` -- the verbatim user command.
* `logs/transcript.json` -- full message history: assistant reasoning, tool
  calls with arguments, tool observations, final reply. This is the
  chain-of-thought artifact required for grading.

## Swapping the LLM

There are two ways of doing this. 

1. **Edit `config.yaml`** -- change `model.name` to any Ollama model that lists
   `tools` in its capabilities (e.g., `llama3.1:8b`, `qwen2.5:7b-instruct`,
   future `gemma` revisions).
2. **Pass `--model`** at the command line: `python -m src.cli --model llama3.1:8b "..."`.

If the new model is hosted remotely, also set `--host http://other-host:11434`
or update `model.host` in the config.


## How to extend this project
                                                                                                                                                      
There are three moving parts that have to line up -- (1) TOOL_SPECS, (2) the handlers in build_tool_registry, and (3) the loop in LLMAgent.run.                                                                                               
  
                                                         
1. Physics (src/physics) implement the tool.
                                                
2. Handler (src/agent/tools.py). Write a handler for the new tool. Write a TOOL_SPEC description of the tool for the agent.

3. If the new tool changes which algorithm should be preferred in some scenario, add a rule to src/agent/prompts.py
