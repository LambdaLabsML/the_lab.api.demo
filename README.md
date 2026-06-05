# Gemma Plays a Tetris-like Puzzle — The Lab Demo

A demo that shows how [the_lab.api](https://github.com/LambdaLabsML/the_lab.api)
works by letting Claude autonomously optimize the Gemma 4 model family to play
the puzzle through better prompting and inference configuration.
Here is what it can look like when you run it — the same Gemma 4 model at five
stages of the optimization:

<img width="988" height="634" alt="Same Gemma 4 model at five stages of the optimization, with scores 0, 4, 7, 9, and 16. Piece placement improves at each stage." src="https://github.com/user-attachments/assets/daccc7dc-796b-4e66-8246-e5633cf3c169" />

*Same model, sharper behavior — Gemma 4 at scores 0, 4, 7, 9, and 16.*

---

## What this is

The benchmark runs a Gemma 4 model (via vLLM on a local H100) as a puzzle
player. Each piece placement is one LLM call: the model receives an ASCII
board, the current piece, ghost/drop position, column heights, and its
previous reasoning. It returns an ordered action list (`LEFT`, `RIGHT`,
`ROT_LEFT`, `ROT_RIGHT`, `HARD_DROP`). The game runs in **wait-for-turn
mode**: the clock is virtual and only advances when the model sends actions.

Over a 2.5-day reference run, the score climbed like this:

<img width="943" height="639" alt="Animated chart showing 468 experiments over time, with best score climbing from 0 to 16. Key milestones are annotated: survival rules, fallback config, cheat-sheet tables, per-piece nudges, and the final user.txt line." src="https://github.com/user-attachments/assets/c8a39b9a-39ac-4b37-af46-757d5b9ac816" />

*468 experiments, 90 ideas — each score jump tied to a specific change Claude discovered and tested.*

**Why sandboxing matters.** The whole point of this demo is to observe how
far a frontier LLM can push a smaller model (Gemma) through prompting and
inference tricks alone. A non-LLM solution would trivially solve the puzzle by
hard-coding an optimal placement algorithm. Without sandboxing, Claude tends
to do exactly that within a few ideas: it notices that the game is not
performing well (both in score and in latency) and quickly figures out that
it can outsource move search to code. The sandbox enforces that the core game
logic stays read-only, so the only levers left are the ones that actually
exercise the Gemma model's reasoning.

Each new idea the agent explores becomes a git branch. The agent can edit
and commit changes to:
- `prompts/system.txt` (system prompt)
- `prompts/user.txt` (per-turn user message template)
- `prompts/action_names.json` (action token names)
- `prompts/fallback.regex` (regex fallback parser)
- `.env` (model selection, vLLM flags)
- `launch_gemma.sh` (vLLM serving config)

Core game files (`tetris_server.py`, `tetris_client.py`,
`run_tetris_experiment.py`, `run_experiment.sh`) are read-only, enforced via
a git pre-commit hook and the sandbox file rules.

---

## Prerequisites

- [the_lab.api](https://github.com/LambdaLabsML/the_lab.api) installed:
  ```bash
  pipx install git+https://github.com/LambdaLabsML/the_lab.api.git
  ```
- A local H100 (or other GPU) — `install.sh` handles the vLLM / torch /
  triton installation. Or use SSH access to a Slurm cluster (see
  [Optional: remote Slurm queue](#optional-remote-slurm-queue) below)
- Gemma 4 model weights on HuggingFace (downloaded automatically by vLLM on
  first run)

---

## Setup

### 1. Clone the project

```bash
git clone https://github.com/LambdaLabsML/the_lab.api.demo.git
cd the_lab.api.demo
```

### 2. Run `the-lab init`

`the-lab init` sets up the project: it creates `.the_lab/PROMPT.md` from a
template, installs the MCP bridge that lets Claude talk to the lab API, writes
`.claude/settings.json` with the right permissions, updates `.gitignore`, and
installs a git pre-commit hook that blocks commits to any file listed in
`.the_lab/blocked_files.txt`.

```bash
the-lab init
```

Toward the end, `the-lab init` asks you to describe your research goal so
that Claude can pre-fill `PROMPT.md` by analyzing the repo:

```
  ? Describe your research goal so Claude can pre-fill PROMPT.md.
    Leave blank to skip and edit the file yourself.
    >
```

Type in the goal below (copy it verbatim), then press Enter. Claude will
read the repo and fill in the prompt file; you can review and adjust it
afterward.

```
Maximize the score that the Gemma model achieves while playing Tetris for
30 minutes (any variant: E2B, E4B, 26B-A4B, 12B or 31B).

Also add to the prompt to "remember the following things":
1) which files are editable
2) to never give up - there's always more to test. If stuck make a
   hypothesis, then test with an analysis script on the logs, then see
   how this can be improved
3) Tetris is only a benchmark for the LLM's reasoning capabilities -
   instead of us precomputing optimal moves in-code we want to see how
   far we can push the Gemma model family
4) sandboxing is enabled to protect our experiment from (accidental)
   cheating
5) use the-lab wait as a background task to be notified of changes
6) You can push multiple jobs to the queue - up to N jobs can run in
   parallel on the remote slurm machine (fill in N from
   .the_lab/queue.json).
```

After Claude finishes, `the-lab init` prints:

```
  ✓ Claude pre-filled PROMPT.md — review and adjust as needed

Next steps:

  1. Review PROMPT.md
  2. Start the server: the-lab .
  3. Launch an agent: the-lab-agent loop
```

The generated prompt is ready to use as-is. In our reference run we left it
unchanged (Claude Opus 4.8 produced a solid starting prompt from the goal
above). You can review and tweak it, but it is not required.

### 3. Configure the sandbox

Sandboxing prevents the agent from writing to the core game files. It also
blocks the agent from swapping out the pinned Jinja chat template (which
would otherwise let it embed move-search logic into the template itself).

**First, set a disable password.** Without one, the agent can turn the
sandbox off through the API. Start the dashboard (`the-lab .`, see step 4),
open the **Sandbox** pane, and click "Set disable password". This writes a
bcrypt hash into `config.json` so the agent cannot disable the sandbox even
if it tries.

Then add the blocked files to the sandbox read-only list. In the **Sandbox**
pane, add the following paths under "Read-only files":

```
tetris_server.py
tetris_client.py
run_tetris_experiment.py
run_experiment.sh
.gitignore
.mcp.json
.the_lab/queue.json
.the_lab/instance.id
.the_lab/sandbox/runtime.json
.the_lab/PROMPT.md
.the_lab/blocked_files.txt
```

Alternatively, write the config directly (the UI and file are in sync):

```bash
mkdir -p .the_lab/sandbox
cat > .the_lab/sandbox/config.json << 'EOF'
{
  "enabled": true,
  "allowlist": [],
  "denylist": [],
  "file_rw": [],
  "file_ro": [
    "tetris_server.py",
    "tetris_client.py",
    "run_tetris_experiment.py",
    "run_experiment.sh",
    ".gitignore",
    ".mcp.json",
    ".the_lab/queue.json",
    ".the_lab/instance.id",
    ".the_lab/sandbox/runtime.json",
    ".the_lab/PROMPT.md",
    ".the_lab/blocked_files.txt"
  ]
}
EOF
```

If you use Slurm (optional, see below), also set the resource queue now
before starting the dashboard, since `queue.json` is in the read-only list
and cannot be edited once the sandbox is enabled.

### 4. Start the dashboard

```bash
the-lab .
```

Open `http://localhost:8000`. If you need TLS or authentication:

```bash
the-lab . --https                             # self-signed TLS
```

```bash
export THE_LAB_USER=alice THE_LAB_PASSWORD=secret
the-lab .                                     # enforce HTTP Basic Auth
```

Key dashboard panes:

| Pane | What it shows |
|------|---------------|
| Metrics | Live metric charts (total\_score, max\_score, restarts, latency) |
| Table | Sortable comparison across all experiments |
| Graph | Idea DAG — branches, status, score improvements |
| Queue | Running and queued experiments |
| Sandbox | Enable/disable sandbox; read-only and hidden file rules |
| Prompts | Edit PROMPT.md; copy the agent launch command |

To inspect a finished experiment's puzzle GIF replay: click the experiment
row in the **Table** pane, then click **Show output** in the detail panel
that slides open. The replay renders inline as an animated GIF.

### 5. Launch the agent

The **Prompts** pane shows the exact launch command for the current session.
Copy it from there, or use:

```bash
cd the_lab.api.demo
the-lab-agent loop -d 15m --model opus
```

`loop -d 15m` re-invokes Claude every 15 minutes in a continuous loop.
`--model opus` uses Claude Opus, which performs best for the multi-step
reasoning this task requires. Sandbox mode follows whatever is configured
in the dashboard, and port 8000 is the default, so no extra flags are
needed for a standard setup.

If you started the dashboard with HTTP Basic Auth, export the same
credentials before launching the agent:

```bash
export THE_LAB_USER=alice THE_LAB_PASSWORD=secret
the-lab-agent loop -d 15m --model opus
```

---

## Watching experiments run

Each experiment:
1. Creates an isolated git worktree
2. Runs `./run_experiment.sh` (installs packages, launches vLLM, plays the puzzle)
3. Writes `tetris_results.json` and `tetris_replay.gif`
4. Prints `{"metrics": {...}}` as the last line (the lab picks this up)

Follow progress in the **Metrics** pane. When an experiment finishes, click
its row in the **Table** pane and then **Show output** to see the puzzle GIF
replay and the full metrics table.

You can also run a single experiment manually from the repo root:

```bash
./run_experiment.sh --think --max-tokens 4096
```

See `run_tetris_experiment.py --help` for all flags.

---

## When the agent gets stuck

The agent may converge early, stop the recurring loop, or run out of ideas.
This requires manual intervention: type into the Claude Code chat window.

Example message that worked in a previous run (after 23 ideas, when the
agent declared convergence at `total_score = 4`):

> Please never stop the cron-job and continue exploring. There is always more
> to find. For instance, analyze movement stats of each piece type using code
> systematically to see if there is room for improvement. Have you tried very
> long reasoning?

---

## Optional: remote Slurm queue

By default, experiments run locally on the machine where `the-lab .` is
running (one experiment at a time per available GPU). If you have access to
a Slurm cluster you can queue many parallel jobs instead.

Without Slurm, `queue.json` will be auto-created by the lab with a single
local GPU resource, and you can skip this section entirely.

To use Slurm, write `.the_lab/queue.json` **before enabling the sandbox**
(the file is in the read-only list and cannot be edited after that):

```bash
cat > .the_lab/queue.json << 'EOF'
{
  "resources": [
    {
      "name": "slurm-lowprio",
      "kind": "slurm",
      "unit_kind": "gpu",
      "capacity": 8,
      "jobs_per_unit": 1.0,
      "tags": [],
      "executor_config": {
        "ssh_host": "slurm",
        "partition": "lowprio",
        "gpus": 1,
        "base_venv_path": "/home/<user>/.thelab/base-venv",
        "git_repo_path": "~/.thelab/repo.git",
        "remote_base": "$HOME/.thelab/jobs",
        "env_vars": {
          "HF_HOME": "/home/<user>/.thelab/hf_cache"
        }
      }
    }
  ],
  "queue": {
    "paused": false,
    "dispatch_interval_s": 2.0
  }
}
EOF
```

Replace `<user>` with your cluster username. `capacity: 8` means up to 8
jobs run simultaneously; reduce it if the partition is congested. The
`partition` value maps to your cluster's Slurm partition name.

**How it works under the hood.** For each experiment, the lab:

1. Pushes the current idea branch to a bare git repo on the cluster
   (`~/.thelab/repo.git`)
2. SSHes in and creates an isolated worktree for the job
3. Submits an `sbatch` wrapper that `cd`s into the worktree and runs
   `run_experiment.sh`
4. Monitors `squeue` until the job finishes, then rsyncs results back

`base_venv_path` points to a shared venv on the cluster that already
contains vLLM, torch, and triton. Each job inherits from it via
`--system-site-packages` so experiment-specific package changes do not
conflict across parallel jobs. To build the base venv from scratch:

```bash
bash install.sh           # SSHes to the cluster and installs there
bash install.sh --local   # installs on the current machine instead
```

The lab's **Queue** pane shows pending, running, and finished jobs in real
time, along with resource allocation.
