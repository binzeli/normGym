"""
normGym human-experiment web app (multi-page, multi-participant).

This builds on the original single-page web player but turns that single-session
player into a real online experiment you can send to participants:

  * Multi-page flow with "Continue" buttons:
        welcome  ->  setting (world + scoring + task)
                 ->  practice (1 round)  ->  experiment (real round)  ->  done
  * Per-participant game state keyed by the Socket.IO session id, so several
    people can take the study at the same time without clobbering each other.
  * Full data logging: every action, position, reward, violation and timing
    event is written to experiment/human/data/<participant>_<phase>_<ts>.jsonl
    plus a summary .json per completed round.

The live game engine (frame streaming, violation modal timing, replacement pool)
is ported from that player, with state ownership moved from one global
dict to one Session object per connection.

Rounds are seeded and built EXACTLY like experiment/run/run_experiment.py:
per-norm seeds come from experiment/run/experiment_config.json (interactive
round = base seed, same-world test k = base seed + k, new-map test k = base seed
+ num_tests + k) and agents come from the pipeline's own builders in the
pipeline's draw order, so a participant's rounds replay the identical setting the
llm / rl subjects faced for the same norm. Each norm block is one learning round
plus two tests — one in the study world, one in a new-map world — and which of
the pipeline's three runs of each test stage a participant gets is drawn per
participant (see the round-schedule block below).

The real study (run_study.sh) runs in design mode: experiment/human/
study_design.json (built by make_study_design.py) assigns 50 participants two
norms each so that every one of the 50 norms is played by two people, once as
scenario 1 and once as scenario 2, and a participant's two norms are from
different norm categories (queue / region / trajectory / allocation).

Run locally:
    python experiment/human/app.py --norm_type straight_line --world_type cafe

Share a public link (run server, then in another terminal):
    cloudflared tunnel --url http://localhost:5001
See experiment/human/README.md for the full sharing guide.
"""
# Real-time frame streaming needs an async server, or the frame stream blocks
# the server. monkey_patch MUST run before importing socket/threading.
try:
    import eventlet
    eventlet.monkey_patch()
    _ASYNC_MODE = 'eventlet'
except Exception:
    _ASYNC_MODE = 'threading'

import base64
import glob
import io
import json
import os
import queue
import time
import random
import secrets
import shutil
import sys
import tempfile
import threading
import traceback
import argparse
import uuid
import zipfile
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import (Flask, render_template, request, redirect, url_for,
                   session as flask_session, jsonify, make_response, send_file)
from flask_socketio import SocketIO

from src.env.gym_env import NormEnv
from src.agents.oracle_agent import Action
from src.norms.occupation import region_staged
from experiment.agents.human_agent import HumanAgent
# The human rounds are built with the SAME builders as the llm/rl pipeline
# (experiment/run/run_experiment.py). Reproducing the pipeline's setting
# takes more than reusing its seed: the setting IS the sequence of draws the
# builders take from the freshly seeded global RNG (subject entry slot,
# entrances, targets, colors, arrival times, replacement pools). Using the same
# builders in the same order is what makes a human round identical to the
# llm/rl round with the same norm + seed.
from experiment.run.common import (
    _random_subject_positions,
    build_initial_agents,
    build_scripted_pool,
    make_agent,
    pop_scripted_replacement,
    release_region_holders,
    respawn_agent,
    subject_class_schedule,
)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_NORMS_JSON = os.path.join(_HERE, '..', '..', 'src', 'norms', 'norm_descriptions.json')
_DATA_DIR = os.path.join(_HERE, 'data')
# Step images live OUTSIDE data/ so the logs stay small and easy to sync: the
# hundreds of PNGs per round go under data_figures/<pid>/... instead.
_FIGURES_DIR = os.path.join(_HERE, 'data_figures')
os.makedirs(_DATA_DIR, exist_ok=True)
os.makedirs(_FIGURES_DIR, exist_ok=True)

with open(_NORMS_JSON) as _f:
    _NORM_DATA = {n['name']: n for n in json.load(_f)['norms']}

# Shared experiment config: the SAME file run_experiment.py reads, so the
# per-norm seeds (and capacity / step budgets) of the human rounds match the
# llm / rl runs exactly. CLI flags override these values, same as the pipeline.
_PIPELINE_CONFIG_PATH = os.path.join(_HERE, '..', 'run', 'experiment_config.json')


def _load_pipeline_config() -> dict:
    try:
        with open(_PIPELINE_CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            return cfg
        print(f'[exp] WARN: experiment config is not a JSON object: {_PIPELINE_CONFIG_PATH}')
    except Exception as e:
        print(f'[exp] WARN: could not read experiment config {_PIPELINE_CONFIG_PATH}: {e}')
    return {}


_PIPELINE_CONFIG = _load_pipeline_config()

# Fixed assignment design for the real study (--assignment design): one norm
# list per participant slot plus a norm -> category map. Generated and
# verified by make_study_design.py; the app only reads it. Empty unless the
# server was started in design mode.
_DEFAULT_DESIGN_PATH = os.path.join(_HERE, 'study_design.json')
_STUDY_DESIGN: dict = {}


def _load_study_design(path: str) -> dict:
    """Read and sanity-check a study design file (see make_study_design.py).
    Raises ValueError with a readable message on anything malformed."""
    with open(path, encoding='utf-8') as f:
        design = json.load(f)
    schedule = design.get('schedule')
    cats = design.get('categories')
    if not isinstance(schedule, list) or not schedule or not isinstance(cats, dict):
        raise ValueError(f'{path}: expected a "schedule" list and a "categories" map')
    blocks = len(schedule[0])
    for i, row in enumerate(schedule):
        if not isinstance(row, list) or len(row) != blocks:
            raise ValueError(f'{path}: slot {i} does not have {blocks} norms: {row}')
        if len({cats.get(n) for n in row}) != len(row):
            raise ValueError(f'{path}: slot {i} repeats a category: {row}')
    design['blocks'] = blocks
    return design


def _norm_category(norm_type: str):
    """The norm's category (queue / region / trajectory / allocation) under the
    loaded design, or None when the server is not in design mode."""
    return _STUDY_DESIGN.get('categories', {}).get(norm_type)


def _round_base_seed(norm_type: str) -> int:
    """Base seed for one norm: CLI --random_seed > per-norm seed from the shared
    config > config-wide seed > 42. Mirrors run_experiment._resolve_seed."""
    if _server_config.get('random_seed') is not None:
        return _server_config['random_seed']
    norm_seeds = _PIPELINE_CONFIG.get('norm_seeds', {})
    if isinstance(norm_seeds, dict) and norm_type in norm_seeds:
        return norm_seeds[norm_type]
    return _PIPELINE_CONFIG.get('random_seed', 42)


def _pipeline_num_tests() -> int:
    """Same-world test runs the llm/rl pipeline does per norm. The human plays
    ONE of them (see _stage_run_index), but the count still sets where the
    pipeline's new-map seeds start, so a human new-map test lands on one of the
    pipeline's own new-map runs."""
    return _server_config.get('num_tests') or _PIPELINE_CONFIG.get('num_tests', 3)


def _new_map_worlds(norm_type: str) -> list:
    """The norm's new-map worlds, POSITIONALLY identical to the list
    run_experiment passes to run_test_series: new-map run k uses worlds[k-1] and
    seed base + num_tests + k, so the index must not be resorted or filtered.
    Mirrors run_experiment._resolve_new_map_worlds, including its seeded
    fallback for a norm with no config entry."""
    norm_worlds = _PIPELINE_CONFIG.get('norm_new_map_worlds', {})
    entry = norm_worlds.get(norm_type) if isinstance(norm_worlds, dict) else None
    if isinstance(entry, str):
        entry = [entry]
    entry = list(entry or _PIPELINE_CONFIG.get('new_map_worlds') or [])
    if entry and all(w in WORLD_CHOICES for w in entry):
        return entry
    candidates = [w for w in WORLD_CHOICES if w != _server_config['world_type']]
    return random.Random(_round_base_seed(norm_type)).sample(candidates, 3)


def _stage_run_count(norm_type: str, stage: str) -> int:
    """How many runs of `stage` the pipeline does per norm — the pool the
    participant's run is drawn from."""
    if stage == 'test':
        return max(1, _pipeline_num_tests())
    if stage == 'new_map_test':
        return max(1, len(_new_map_worlds(norm_type)))
    return 0


def _stage_run_index(participant_id, norm_type: str, stage: str) -> int:
    """Which of the pipeline's runs of `stage` this participant plays (1-based).

    The pipeline does several runs per test stage (three of each in the current
    config); a human session only has room for one, so each participant is
    handed one AT RANDOM instead of everyone replaying run 1 — over the study
    the pipeline's settings all get covered.

    The draw is a private RNG keyed by participant + norm + stage, so it is
    stable across reloads, restarts and a lost browser session (a returning
    participant id replays the same setting), independent between the two
    stages, and it leaves the global RNG — and therefore every round's setting
    — untouched.
    """
    count = _stage_run_count(norm_type, stage)
    if count <= 0:
        return 0
    key = f'{_safe_pid(participant_id)}|{norm_type}|{stage}'
    return random.Random(key).randint(1, count)


def _round_run_index(participant_id, norm_type: str, episode: int) -> int:
    """The pipeline run index this main-round episode replays (0 = interactive,
    which has only one run)."""
    return _stage_run_index(participant_id, norm_type, _episode_stage(episode))


def _round_seed(norm_type: str, episode: int, run_index: int) -> int:
    """Seed for one main round, derived exactly as run_experiment.py derives its
    stage seeds: the interactive round uses the norm's base seed, same-world
    test k uses base + k (run_test_series with seed_offset=0), and new-map test
    k uses base + num_tests + k (seed_offset=num_tests)."""
    base = _round_base_seed(norm_type)
    stage = _episode_stage(episode)
    if stage == 'interactive':
        return base
    if stage == 'test':
        return base + run_index
    return base + _pipeline_num_tests() + run_index


def _round_world(norm_type: str, episode: int, run_index: int) -> str:
    """World a main-round episode is played in: the study world everywhere
    except the new-map test, which takes the world of the new-map run drawn for
    this participant."""
    if _episode_stage(episode) != 'new_map_test':
        return _server_config['world_type']
    worlds = _new_map_worlds(norm_type)
    return worlds[min(max(run_index, 1), len(worlds)) - 1]


def _pipeline_run_id(stage: str, run_index: int, world_type: str) -> str:
    """Name of the pipeline run this round replays, formatted exactly as
    run_test_series names it, so a human round joins to the llm/rl round of the
    same setting (e.g. 'test_002', 'new_map_test_003_library')."""
    if stage not in ('test', 'new_map_test'):
        return stage
    suffix = f'_{world_type}' if stage == 'new_map_test' else ''
    return f'{stage}_{run_index:03d}{suffix}'


def _round_capacity(norm_type: str) -> int:
    """CLI --capacity > per-norm capacity from the shared config > config-wide
    capacity > 8. Mirrors run_experiment._resolve_capacity."""
    if _server_config.get('capacity') is not None:
        return _server_config['capacity']
    norm_capacity = _PIPELINE_CONFIG.get('norm_capacity', {})
    if isinstance(norm_capacity, dict) and norm_type in norm_capacity:
        return norm_capacity[norm_type]
    return _PIPELINE_CONFIG.get('capacity', 8)

# Norms that are playable but deliberately not experiment norms, so they carry no
# entry in norm_descriptions.json. 'straight_line' is the default tutorial norm:
# it is taught in the Example + Practice intro and excluded from the test pool.
_PRACTICE_ONLY_NORMS = ['straight_line']
NORM_CHOICES = list(_NORM_DATA.keys()) + [
    n for n in _PRACTICE_ONLY_NORMS if n not in _NORM_DATA]

WORLD_CHOICES = ['cafe', 'movie_theater', 'bank', 'pharmacy', 'grocery_store',
                 'clothing_store', 'library', 'post_office', 'hospital', 'barbershop']

# ---------------------------------------------------------------------------
# Round schedule for one norm block
#
# The llm/rl pipeline tests a subject in two stages: `num_tests` runs in the
# world it learned the norm in ('test'), then one run per world in the norm's
# norm_new_map_worlds ('new_map_test'), which probes whether the norm transfers
# to a place it has never seen. A human session has to stay short, so each norm
# block keeps ONE run of each stage: the learning round, then one same-world
# test and one new-map test. WHICH of the pipeline's runs of a stage a
# participant gets is drawn per participant (see _stage_run_index), so every
# human round is a setting an llm/rl subject also faced and the study spreads
# over all of them instead of everyone replaying run 1.
#
#   episode 0 -> interactive     (learning, in the study world)
#   episode 1 -> test            (same world as the learning round)
#   episode 2 -> new_map_test    (that run's new-map world; the /new-map-intro
#                                 page tours that world first, the way the
#                                 tutorial tours the café)
# ---------------------------------------------------------------------------
SAME_WORLD_TESTS = 1
NEW_MAP_TESTS = 1
EPISODES_PER_NORM = 1 + SAME_WORLD_TESTS + NEW_MAP_TESTS
# First episode of the new-map stage (the round the test intro previews).
NEW_MAP_EPISODE = 1 + SAME_WORLD_TESTS


def _episode_stage(episode: int) -> str:
    """Pipeline stage a main-round episode belongs to."""
    if episode <= 0:
        return 'interactive'
    if episode <= SAME_WORLD_TESTS:
        return 'test'
    return 'new_map_test'


# Interactive round: play pauses every this many steps (30, 60, 90, …) to ask for
# an in-progress guess of the norm plus a confidence rating. Set to 0 to disable.
# The client triggers it off the step it is *displaying*, so the prompt lines up
# with what the participant can see rather than with the server's lead.
MID_GUESS_EVERY_STEPS = 30

# Prolific completion code, shown on the final thank-you page. Prolific issues
# this per study, so set COMPLETION_CODE in the environment when deploying
# rather than editing the placeholder below.
COMPLETION_CODE = os.environ.get('COMPLETION_CODE', 'XXXXXXXX')

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
# Pages are already sent no-cache; do the same for static/ so an edited
# experiment.js or style.css never runs against a browser-cached older copy.
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
socketio = SocketIO(app, cors_allowed_origins='*', async_mode=_ASYNC_MODE)

# Populated once from CLI args at startup; shared (read-only) by all participants.
_server_config: dict = {}

# Per-connection game state. Key = Socket.IO sid.
_sessions: dict = {}
_sessions_lock = threading.Lock()


def _safe_pid(participant_id) -> str:
    return ''.join(c for c in str(participant_id)
                   if c.isalnum() or c in '-_')[:40] or 'anon'


def _participant_dirname(participant_id) -> str:
    """Name of this participant's folder under data/ and data_figures/ for the
    CURRENT study. Normally their (sanitised) Prolific id. A participant who
    already took part in an earlier study — their id is in an archived ledger
    data/_assignments_<timestamp>.json — gets '<id>_2' for the second study,
    '<id>_3' for a third, and so on (decided once, at /begin, and recorded as
    'data_dir' in their ledger entry), so every study's data is a separate,
    equally complete folder. Falls back to the plain id when the ledger has no
    entry (e.g. --assignment random)."""
    safe = _safe_pid(participant_id)
    try:
        with _assign_lock:
            entry = _load_assignments()['assigned'].get(str(participant_id))
    except Exception:
        entry = None
    return (entry or {}).get('data_dir') or safe


# ---------------------------------------------------------------------------
# Frame archive — a picture of every step the participant played
#
# The live game renders into a temp dir that is wiped when the round ends. For
# the rounds that are actual data (interactive + the test trials) we also keep a
# permanent copy under the participant's folder in data_figures/ (kept separate
# from data/ so the JSON/JSONL logs stay small):
#
#   data_figures/<pid>/norm1_<norm>/interactive/step_007.png
#   data_figures/<pid>/norm1_<norm>/test2/step_012.png
#   data_figures/<pid>/norm1_<norm>/new_map_test3/step_012.png
#
# (the number on a test folder is the pipeline run this participant drew, not a
# count of rounds: they play one test of each stage)
#
# Only the SETTLED frame of each step is kept. With interpolation on, a step
# writes step_NNN_sub0.png (the position at the end of that step) plus
# step_NNN_subK.png tweens toward the next one; keeping sub0 alone gives exactly
# one image per step, matching the 'step' records in the round's .jsonl.
# ---------------------------------------------------------------------------
def _frames_archive_dir(participant_id, stage, block, norm_type, run_index):
    """Destination folder for this round's frames, or None for rounds we don't
    archive (the autoplay demo and the practice round, which aren't data)."""
    if stage not in ('interactive', 'test', 'new_map_test'):
        return None
    round_name = ('interactive' if stage == 'interactive'
                  else f'{stage}{max(1, run_index)}')
    return os.path.join(_FIGURES_DIR, _participant_dirname(participant_id),
                        f'norm{block + 1}_{norm_type}', round_name)


def _archive_round_frames(sess) -> int:
    """Copy this round's step images out of the temp render dir and into the
    participant's folder under data_figures/. Called once, when the round
    COMPLETES — the
    temp dir holds every frame of the round until the session is cleaned up, so
    there is nothing to accumulate during play, and an aborted round simply never
    reaches here (no half-round folders, and no cleanup race with a reload
    starting the same round again)."""
    out = sess.frames_archive_dir
    if not out:
        return 0
    try:
        # Leftovers here can only come from an earlier COMPLETED run of the same
        # round by the same participant id: move them aside instead of mixing.
        if os.path.isdir(out) and os.listdir(out):
            os.rename(out, f'{out}_prev_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(out, exist_ok=True)
    except Exception as e:
        print(f'[frames] could not prepare {out}: {e}')
        return 0
    n = 0
    try:
        names = sorted(os.listdir(sess.frames_dir))
    except Exception as e:
        print(f'[frames] could not read render dir: {e}')
        return 0
    for fname in names:
        if not fname.startswith('step_') or not fname.endswith('.png'):
            continue
        if '_sub' in fname and not fname.endswith('_sub0.png'):
            continue  # in-between interpolation frame, not a step position
        try:
            shutil.copy2(os.path.join(sess.frames_dir, fname),
                         os.path.join(out, fname.replace('_sub0.png', '.png')))
            n += 1
        except Exception as e:
            print(f'[frames] archive failed for {fname}: {e}')
    return n


# ---------------------------------------------------------------------------
# Per-participant logging
# ---------------------------------------------------------------------------
class ParticipantLogger:
    """Append-only JSONL event log + final summary for one round (phase)."""

    def __init__(self, participant_id, phase, stage, norm_type, world_type,
                 block, episode, run_index, config):
        # Folder + file prefix: the id, or '<id>_2' etc. for a repeat study
        # (see _participant_dirname). The raw id is kept as prolific_id.
        safe_pid = _participant_dirname(participant_id)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.participant_id = safe_pid
        self.prolific_id = _safe_pid(participant_id)
        self.phase = phase
        self.stage = stage
        self.norm_type = norm_type
        self.world_type = world_type
        self.block = block
        self.episode = episode
        # Which pipeline run of the stage this participant drew, and its
        # pipeline run id — the join key to the llm/rl run of the same setting.
        self.stage_run = run_index
        self.pipeline_run_id = _pipeline_run_id(stage, run_index, world_type)
        self.start_time = time.time()
        # Intro phases (demo/practice) aren't tied to a test-norm block; tag them
        # by phase only. Main rounds are named by their pipeline stage
        # (interactive / test / new_map_test) plus block + episode.
        if phase in ('demo', 'practice'):
            base = f'{safe_pid}_{phase}_{norm_type}_{ts}'
        else:
            base = f'{safe_pid}_b{block}_{norm_type}_{stage}_ep{episode}_{ts}'
        # Each participant gets their own subfolder under data/.
        pdir = os.path.join(_DATA_DIR, safe_pid)
        os.makedirs(pdir, exist_ok=True)
        self.events_path = os.path.join(pdir, base + '.jsonl')
        self.summary_path = os.path.join(pdir, base + '_summary.json')
        self._lock = threading.Lock()
        self.log('session_start', prolific_id=self.prolific_id, participant_id=safe_pid, phase=phase,
                 stage=stage, block=block, episode=episode, norm_type=norm_type,
                 world_type=world_type, stage_run=self.stage_run,
                 pipeline_run_id=self.pipeline_run_id, config=config)

    def log(self, event, **data):
        rec = {'t': round(time.time() - self.start_time, 3), 'event': event, **data}
        try:
            with self._lock, open(self.events_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except Exception as e:
            print(f'[log] write failed: {e}')

    def discard(self):
        """Remove this round's log files. Used when a round is aborted (the
        participant restarted, reloaded, or disconnected before finishing) so we
        don't leave orphan .jsonl files with no summary."""
        for p in (self.events_path, self.summary_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                print(f'[log] discard failed: {e}')

    def write_summary(self, summary: dict):
        try:
            with open(self.summary_path, 'w') as f:
                json.dump({
                    'participant_id': self.participant_id,
                    'phase': self.phase,
                    'stage': self.stage,
                    'block': self.block,
                    'episode': self.episode,
                    'stage_run': self.stage_run,
                    'pipeline_run_id': self.pipeline_run_id,
                    'norm_type': self.norm_type,
                    'world_type': self.world_type,
                    'duration_sec': round(time.time() - self.start_time, 1),
                    **summary,
                }, f, indent=2)
        except Exception as e:
            print(f'[log] summary write failed: {e}')


# ---------------------------------------------------------------------------
# Session object — one per connected browser
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, sid, participant_id, phase, stage, norm_type, world_type,
                 block, episode, run_index, config):
        self.sid = sid
        self.participant_id = participant_id
        self.phase = phase            # 'demo' | 'practice' | 'interactive' | 'test'
        # Pipeline stage this round is data for: 'interactive' | 'test' |
        # 'new_map_test' for main rounds, the phase itself for demo/practice.
        # The phase drives the engine and the UI; the stage names the data.
        self.stage = stage
        self.norm_type = norm_type
        self.world_type = world_type
        self.block = block            # which norm block (0-indexed) this round belongs to
        self.episode = episode        # 0=interactive, 1=same-world test, 2=new-map test
        self.run_index = run_index    # which pipeline run of the stage this replays
        self.config = config
        self.action_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.frames_dir = tempfile.mkdtemp(prefix='normgym_exp_')
        # Where this round's step images are kept once it completes (interactive
        # and test rounds only; None for the demo/practice rounds).
        self.frames_archive_dir = _frames_archive_dir(
            participant_id, stage, block, norm_type, run_index)
        self.thread = None
        self.env = None
        self.logger = ParticipantLogger(participant_id, phase, stage, norm_type,
                                        world_type, block, episode, run_index,
                                        config)

    def cleanup(self):
        self.stop_event.set()
        self.action_queue.put(Action.WAIT)  # unblock a thread waiting on get()
        try:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Round setup — agents come from experiment/run/common.py builders.
#
# A round's setup (subject slot, entrances, targets, colors, arrival times,
# replacement pools) is drawn from the global RNG immediately after re-seeding
# it with the round's pipeline seed. The lock keeps two sessions that start at
# the same moment from interleaving their draw sequences (eventlet can switch
# greenlets on the file I/O inside env construction). In-run draws (replacement
# entrances/targets) depend on the participant's own timing, exactly as the
# pipeline's depend on the subject's behavior.
# ---------------------------------------------------------------------------
_setup_lock = threading.Lock()


def _make_env(norm_type, figures_dir, num_agents, world_type, max_steps,
              use_isometric=False, map_type='default', interpolation_steps=0):
    env = NormEnv(
        num_agents=num_agents, max_steps=max_steps, norm_type=norm_type,
        output_dir=figures_dir, render_mode='rgb_array', world_type=world_type,
        use_isometric=use_isometric, map_type=map_type,
        interpolation_steps=interpolation_steps,
    )
    if hasattr(env, 'flat_renderer') and env.flat_renderer is not None:
        env.flat_renderer.save_dpi = 72
        env.flat_renderer.tight_bbox = False
    return env


# ---------------------------------------------------------------------------
# Scene picture for the new-map tour
#
# Before the new-map test the participant is shown around that world, the way
# the tutorial shows them around the café. The picture is the world's own board
# — floor, furniture, doors and waiting zone drawn with the game's sprites,
# through the game's own renderer, with no characters on it.
#
# It is drawn into an axes that FILLS the figure (no title, no tick margins),
# because the tour's arrows have to land on specific cells: with the axes
# filling the frame, cell (r, c) sits at exactly ((c + 0.5) / width,
# (r + 0.5) / height) of the image, in every world, whatever matplotlib's
# layout would otherwise do with a title and axis labels.
#
# One render per world, cached in memory (the boards are fixed).
# ---------------------------------------------------------------------------
_SCENE_CELL_PX = 80          # matches the env's sprite resolution
_scene_cache: dict = {}
_scene_lock = threading.Lock()


def _scene_world(world_type: str):
    """A bare world object for `world_type` — no env, no agents."""
    from src.env.gym_env import WORLD_CLASSES
    world_cls = WORLD_CLASSES.get(world_type) or WORLD_CLASSES['cafe']
    return world_cls(map_type=_server_config.get('map_type', 'default'))


def _render_scene_png(world_type: str) -> bytes:
    """PNG of the world's empty board, edge to edge (see the note above)."""
    import matplotlib.pyplot as plt
    from src.env.sprite_loader import SpriteCache
    from src.env.flat_renderer import FlatRenderer

    world = _scene_world(world_type)
    assets_dir = os.path.join(_HERE, '..', '..', 'assets', 'sprites')
    sprites = SpriteCache(assets_dir, px_per_cell=_SCENE_CELL_PX)
    sprites.load_world_sprites(world_type)
    renderer = FlatRenderer(sprites, world)

    fig = plt.figure(figsize=(world.width, world.height), dpi=_SCENE_CELL_PX)
    try:
        ax = fig.add_axes([0, 0, 1, 1])  # axes fills the figure exactly
        ax.imshow(
            sprites.build_static_background(world.grid,
                                            getattr(world, 'x_overlay', set())),
            origin='upper',
            extent=(-0.5, world.width - 0.5, world.height - 0.5, -0.5),
            interpolation='nearest', zorder=0,
        )
        # The same overlays the game draws on top of the static background.
        renderer._draw_display_entrances(ax)
        renderer._draw_waiting_region(ax)
        ax.set_xlim(-0.5, world.width - 0.5)
        ax.set_ylim(world.height - 0.5, -0.5)
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=_SCENE_CELL_PX)
    finally:
        plt.close(fig)
    return buf.getvalue()


def _scene_png(world_type: str) -> bytes:
    """The world's scene picture, rendered once and kept in memory. The lock
    also keeps the render off matplotlib's global state while a round's frames
    are being drawn."""
    with _scene_lock:
        png = _scene_cache.get(world_type)
        if png is None:
            png = _render_scene_png(world_type)
            _scene_cache[world_type] = png
        return png


# Where the scene sits inside the tour page's SVG canvas, in viewBox units: a
# gutter each side holds the callout labels, the board goes between them. Same
# proportions as the tutorial's café page, so the two pages read alike.
_SCENE_CANVAS = dict(gutter=308, gap=26, img_x=340, img_y=20, img_w=1022,
                     label_w=308, label_h=132, canvas_w=1702)


def _scene_features(world, world_type: str):
    """The four things the tour points at, as (key, title, text, cell).

    Every world has all four (each is built with one counter, a carpet strip,
    chairs and three doors); anything missing is simply dropped.
    """
    zones = world.zones
    counter_cells = list(zones.get('counter') or [])
    chairs = list(zones.get('chairs') or [])
    carpet = list(zones.get('affordance_x') or [])
    doors = list(getattr(world, 'entrances', ()) or zones.get('entrance') or [])

    features = []
    if doors:
        # Point at the door nearest a side wall: its arrow has the shortest,
        # least crossing path from the gutter.
        door = min(doors, key=lambda p: min(p[1], world.width - 1 - p[1]))
        features.append((
            'door', 'Door',
            f'There are three doors at the edges. You come in and leave '
            f'through them.', door))
    if chairs:
        # Take the chair furthest from the counter, so the two arrows come in
        # from opposite sides instead of crossing.
        anchor_col = counter_cells[0][1] if counter_cells else world.width // 2
        chair = max(chairs, key=lambda p: abs(p[1] - anchor_col))
        features.append((
            'seat', 'Chair', 'You can <strong>sit on a chair</strong>.', chair))
    if counter_cells:
        features.append((
            'counter', 'Counter', 'The service counter where you are served.',
            counter_cells[0]))
    if carpet:
        # The middle of the carpet strip reads best as "this grey area".
        carpet_sorted = sorted(carpet)
        features.append((
            'carpet', 'Grey carpet', 'The grey floor area.',
            carpet_sorted[len(carpet_sorted) // 2]))
    return features


def scene_annotation(world_type: str) -> dict:
    """Everything the tour page needs to draw the annotated scene: the picture,
    and for each feature a callout box plus an arrow to the cell it describes.

    Sides are chosen from where the cell actually is (left half of the board ->
    left gutter), then balanced so neither gutter takes more than half the
    labels, and the labels on a side are stacked in the order of the cells they
    point at, so the arrows never cross.
    """
    world = _scene_world(world_type)
    cv = _SCENE_CANVAS
    img_h = round(cv['img_w'] * world.height / world.width)
    canvas_h = cv['img_y'] * 2 + img_h

    features = _scene_features(world, world_type)

    def anchor(cell):
        r, c = cell
        return (cv['img_x'] + (c + 0.5) / world.width * cv['img_w'],
                cv['img_y'] + (r + 0.5) / world.height * img_h)

    # Preferred side from the cell's column, then balance the two gutters by
    # moving whichever feature sits closest to the board's midline.
    mid = (world.width - 1) / 2
    placed = [{'key': k, 'title': t, 'text': x, 'cell': cell,
               'anchor': anchor(cell), 'col': cell[1],
               'side': 'left' if cell[1] <= mid else 'right'}
              for k, t, x, cell in features]
    cap = max(1, (len(placed) + 1) // 2)
    for side, other in (('left', 'right'), ('right', 'left')):
        while len([f for f in placed if f['side'] == side]) > cap:
            movable = [f for f in placed if f['side'] == side]
            movable.sort(key=lambda f: abs(f['col'] - mid))
            movable[0]['side'] = other

    items = []
    for side in ('left', 'right'):
        column = sorted([f for f in placed if f['side'] == side],
                        key=lambda f: f['anchor'][1])
        for i, f in enumerate(column):
            # Spread the labels down the height of the picture.
            centre_y = cv['img_y'] + (i + 0.5) / len(column) * img_h
            label_x = (0 if side == 'left'
                       else cv['canvas_w'] - cv['label_w'])
            start_x = (cv['label_w'] + cv['gap'] if side == 'left'
                       else cv['canvas_w'] - cv['label_w'] - cv['gap'])
            items.append({
                'key': f['key'], 'title': f['title'], 'text': f['text'],
                'side': side, 'cell': list(f['cell']),
                'label_x': label_x,
                'label_y': round(centre_y - cv['label_h'] / 2, 1),
                'label_w': cv['label_w'], 'label_h': cv['label_h'],
                'x1': start_x, 'y1': round(centre_y, 1),
                'x2': round(f['anchor'][0], 1), 'y2': round(f['anchor'][1], 1),
            })

    return {
        'world_type': world_type,
        'canvas_w': cv['canvas_w'], 'canvas_h': canvas_h,
        'img_x': cv['img_x'], 'img_y': cv['img_y'],
        'img_w': cv['img_w'], 'img_h': img_h,
        'items': items,
    }


# ---------------------------------------------------------------------------
# Frame helpers (unchanged)
# ---------------------------------------------------------------------------
def _collect_new_frames(frames_dir: str, already_sent: set):
    try:
        all_pngs = sorted(
            f for f in os.listdir(frames_dir)
            if f.startswith('step_') and f.endswith('.png')
        )
    except Exception:
        return [], already_sent
    new_pngs = [f for f in all_pngs if f not in already_sent]
    frames_b64 = []
    for fname in new_pngs:
        try:
            with open(os.path.join(frames_dir, fname), 'rb') as fh:
                frames_b64.append(base64.b64encode(fh.read()).decode('utf-8'))
        except Exception:
            pass
    return frames_b64, already_sent | set(new_pngs)


def _find_human_agent(env):
    active = [a for a in env.agents if isinstance(a, HumanAgent) and a.active and not a.done]
    if active:
        return active[0]
    all_humans = [a for a in env.agents if isinstance(a, HumanAgent)]
    return all_humans[0] if all_humans else None


def _human_agent_ids(env) -> set:
    return {a.agent_id for a in env.agents if isinstance(a, HumanAgent)}


def _describe_background_violations(env, raw_violations, human_ids):
    """The scripted characters' norm violations this step, as loggable dicts.

    The env reports violations as (agent_id, from_pos, to_pos). Everything the
    participant didn't do is a background character; `no_norm_agent` marks the
    deliberate rule-breakers the interactive round mixes in (built with norm=None),
    which is what separates a scripted violation the participant was *meant* to
    see from an incidental one by a norm-following customer.
    """
    others = [v for v in raw_violations if v[0] not in human_ids]
    if not others:
        return []
    by_id = {a.agent_id: a for a in env.agents}
    out = []
    for agent_id, from_pos, to_pos in others:
        agent = by_id.get(agent_id)
        out.append({
            'agent_id': agent_id,
            'from': list(from_pos) if from_pos is not None else None,
            'to': list(to_pos) if to_pos is not None else None,
            'agent_class': getattr(agent, 'agent_class', None),
            'no_norm_agent': (agent is not None and getattr(agent, 'norm', None) is None),
        })
    return out


# ---------------------------------------------------------------------------
# Game loop — ported from the single-page player's game loop, now session-scoped.
# ---------------------------------------------------------------------------
def _game_loop(sess: Session, initial_agents, waiting_human, waiting_norm_customer,
               waiting_no_norm_customer, norm_type, mode):
    env = sess.env
    sid = sess.sid
    frames_dir = sess.frames_dir
    action_queue = sess.action_queue
    stop_event = sess.stop_event
    logger = sess.logger

    sent_frames: set = set()
    # Both counters are HUMAN-ONLY: they tally collisions and norm violations that
    # the participant's own character is involved in, not the whole environment.
    total_norm_violations: int = 0
    total_human_collisions: int = 0
    # Norm violations by the scripted background characters (NOT the participant's).
    total_background_violations: int = 0
    _last_emit_state = {'norm_violations': 0, 'total_collisions': 0,
                        'human_violated': False, 'step_reward': 0.0}

    def _emit_frames(output_dir, waiting_for_input, step, step_reward=0.0,
                     norm_violations=0, human_violated=False,
                     terminated=False, truncated=False, include_frames=False):
        nonlocal sent_frames
        _human = _find_human_agent(env)
        if include_frames:
            frames_b64, sent_frames = _collect_new_frames(output_dir, sent_frames)
        else:
            frames_b64 = []
        socketio.emit('game_state', {
            'step': step,
            'step_reward': step_reward,
            'cumulative_reward': _human.total_reward if _human else 0.0,
            'total_collisions': _last_emit_state['total_collisions'],
            'norm_violations': norm_violations,
            'human_violated': human_violated,
            'human_done': _human.done if _human else True,
            'human_class': _human.agent_class if _human else None,
            'terminated': terminated,
            'truncated': truncated,
            'frames_b64': frames_b64,
            'human_position': list(_human.current_position) if _human and _human.current_position is not None else [],
            'human_target': list(_human.display_target()) if _human and _human.display_target() is not None else [],
            'human_goal': _human.goal_text() if _human else None,
            'waiting_for_input': waiting_for_input,
        }, room=sid)

    def _stream_frame(path, frame_violations=None):
        nonlocal sent_frames
        fname = os.path.basename(path)
        if fname in sent_frames:
            return
        try:
            with open(path, 'rb') as fh:
                frame_b64 = base64.b64encode(fh.read()).decode('utf-8')
        except Exception:
            return
        sent_frames = sent_frames | {fname}
        _human = _find_human_agent(env)
        human_ids = _human_agent_ids(env)
        human_violated = any(v[0] in human_ids for v in (frame_violations or []))
        socketio.emit('game_state', {
            'step': env.current_step,
            'step_reward': 0.0,
            'cumulative_reward': _human.total_reward if _human else 0.0,
            'total_collisions': _last_emit_state['total_collisions'],
            'norm_violations': _last_emit_state['norm_violations'],
            'human_violated': human_violated,
            'human_done': _human.done if _human else True,
            'human_class': _human.agent_class if _human else None,
            'terminated': False, 'truncated': False,
            'frames_b64': [frame_b64],
            'human_position': list(_human.current_position) if _human and _human.current_position is not None else [],
            'human_target': list(_human.display_target()) if _human and _human.display_target() is not None else [],
            'human_goal': _human.goal_text() if _human else None,
            'waiting_for_input': False,
        }, room=sid)
        socketio.sleep(0.01)

    def _pre_action_emit(output_dir):
        _emit_frames(output_dir, waiting_for_input=True, step=env.current_step,
                     norm_violations=_last_emit_state['norm_violations'],
                     human_violated=_last_emit_state['human_violated'],
                     step_reward=_last_emit_state['step_reward'])
        logger.log('await_input', step=env.current_step,
                   norm_violations=_last_emit_state['norm_violations'])
        _last_emit_state['human_violated'] = False

    for _a in initial_agents:
        if isinstance(_a, HumanAgent):
            _a.emit_callback = _pre_action_emit
    for _a in waiting_human:
        _a.emit_callback = _pre_action_emit

    obs, info = env.reset(options={'agents': initial_agents})
    env.output_dir = frames_dir
    env.render()
    _emit_frames(frames_dir, waiting_for_input=False, step=0, include_frames=True)
    env.frame_callback = _stream_frame

    initial_agent_ids = {a.agent_id for a in initial_agents}
    # The participant's own character. In PRACTICE and TEST rounds, the round ends
    # as soon as this agent reaches its goal. In the INTERACTIVE round it does NOT:
    # the human is replaced and play continues until max_steps.
    is_interactive = (mode == 'interactive')
    human_player_id = next((a.agent_id for a in initial_agents
                            if isinstance(a, HumanAgent)), None)
    done_agent_ids: set = set()
    terminated = truncated = False
    human_finished = False
    step_count = 0
    next_replacement_step = 1
    last_spawn_entrance = None

    def _should_stop():
        return stop_event.is_set()

    # A disconnect wipes the temp render dir out from under the renderer, so a
    # round torn down mid-step raises here. Any crash is treated as an aborted
    # round: fall through to the stop_event branch below, which drops the partial
    # log. Without this the thread would die in place and orphan a half-written
    # .jsonl with no summary. An unexpected crash (one we didn't ask for) still
    # prints its traceback.
    try:
        while not (terminated or truncated or human_finished) and not _should_stop():
            if mode == 'test' and initial_agent_ids.issubset(done_agent_ids):
                break
            try:
                obs, reward, terminated, truncated, info = env.step()
            except SystemExit:
                break
            if _should_stop():
                break

            step_count = info['step']
            env.render()

            # Test rounds under a staged region norm: free the waiting-zone holders
            # one step after the participant reaches the zone edge — the same
            # protocol the pipeline's test runs apply to llm/rl subjects.
            if mode == 'test':
                release_region_holders(env, _human_agent_ids(env), norm_type)

            step_rewards = info.get('step_rewards', {})
            human_ids = _human_agent_ids(env)
            # The participant's own counters stay HUMAN-ONLY (they drive the UI and
            # the score); the background characters' violations are tracked
            # separately so the log shows what the participant could observe others
            # doing on the same step.
            raw_violations = info.get('norm_violations', [])
            human_violations = [v for v in raw_violations if v[0] in human_ids]
            total_norm_violations += len(human_violations)
            human_violated = bool(human_violations)
            background_violations = _describe_background_violations(
                env, raw_violations, human_ids)
            total_background_violations += len(background_violations)
            raw_collisions = info.get('collisions', {})
            total_human_collisions += sum(
                1 for agents_at_pos in raw_collisions.values()
                if any(a.agent_id in human_ids for a in agents_at_pos))

            # --- Replacement logic (same protocol as the pipeline) ---
            # Interactive: everyone is replaced when done — INCLUDING the human, so
            # the participant immediately controls a fresh character and play
            # continues. Practice/test: scripted customers are replaced but the
            # participant is not (the round ends when their character finishes),
            # matching run_test_simulation. Demo keeps its initial cast so the
            # autoplay example ends cleanly.
            newly_done = [a for a in env.agents if a.done and a.agent_id not in done_agent_ids]
            for agent in newly_done:
                done_agent_ids.add(agent.agent_id)
                if isinstance(agent, HumanAgent):
                    if mode != 'interactive':
                        continue
                    new_agent = waiting_human.popleft() if waiting_human else None
                else:
                    if sess.phase == 'demo':
                        continue
                    new_agent = pop_scripted_replacement(
                        agent, waiting_norm_customer, waiting_no_norm_customer)
                if new_agent is None:
                    continue  # pool exhausted for this type
                next_replacement_step, last_spawn_entrance = respawn_agent(
                    env, new_agent, next_replacement_step, last_spawn_entrance,
                    keep_out_of_region=(mode == 'test' and region_staged(norm_type)),
                )

            # The participant's first character finishing (reaching the exit and leaving).
            human_left = human_player_id is not None and human_player_id in done_agent_ids
            # End practice/test the moment the participant's own character is done and
            # has left the screen. (The interactive round ignores this and runs on with
            # replacements.)
            if not is_interactive and human_left:
                human_finished = True

            human = _find_human_agent(env)
            human_reward_this_step = step_rewards.get(human.agent_id, 0.0) if human else 0.0
            _last_emit_state['norm_violations'] = total_norm_violations
            _last_emit_state['total_collisions'] = total_human_collisions
            _last_emit_state['step_reward'] = human_reward_this_step

            # --- Per-step data log ---
            logger.log('step', step=step_count,
                       human_position=list(human.current_position) if human and human.current_position is not None else None,
                       human_target=list(human.display_target()) if human and human.display_target() is not None else None,
                       human_goal=human.goal_text() if human else None,
                       step_reward=round(human_reward_this_step, 3),
                       cumulative_reward=round(human.total_reward, 3) if human else 0.0,
                       total_collisions=total_human_collisions,
                       norm_violations=total_norm_violations,
                       human_violated=bool(human_violated),
                       background_violations=background_violations,
                       background_violations_total=total_background_violations)

            _emit_frames(frames_dir, waiting_for_input=False, step=step_count,
                         step_reward=human_reward_this_step,
                         norm_violations=total_norm_violations, human_violated=False,
                         terminated=bool(terminated or human_finished),
                         truncated=bool(truncated))

    except Exception:
        if not stop_event.is_set():
            print('[exp] round crashed; discarding it as an aborted round:')
            traceback.print_exc()
        stop_event.set()

    # Aborted round (restart / page reload / disconnect set stop_event): drop the
    # partial log so only completed rounds leave a file. Guard against a
    # same-second restart reusing this filename: never delete a path that a live
    # session currently owns.
    if stop_event.is_set():
        with _sessions_lock:
            in_use = any(s.logger.events_path == logger.events_path
                         for s in _sessions.values() if s.logger is not logger)
        if not in_use:
            logger.discard()
        return

    # Keep the round's step images before anything can clean up the temp dir (a
    # participant clicking through to the next page disconnects, which cleans it).
    n_frames = _archive_round_frames(sess)
    if n_frames:
        logger.log('frames_archived', count=n_frames,
                   path=os.path.relpath(sess.frames_archive_dir, _FIGURES_DIR))

    human = _find_human_agent(env)
    final_reward = human.total_reward if human else 0.0
    summary = {
        'norm_category': _norm_category(norm_type),
        'total_steps': env.current_step,
        'total_collisions': total_human_collisions,
        'norm_violations': total_norm_violations,
        'background_norm_violations': total_background_violations,
        'final_reward': round(final_reward, 3),
        'agents_processed': len(done_agent_ids),
        'frames_saved': n_frames,
    }
    logger.log('game_over', **summary)
    logger.write_summary(summary)
    socketio.emit('game_over', summary, room=sid)


# ---------------------------------------------------------------------------
# Page routes (multi-page flow)
# ---------------------------------------------------------------------------
def _no_cache(resp):
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


def _participant():
    """Return participant id for this browser, or None if not registered."""
    return flask_session.get('participant_id')


def _assigned_norms():
    """The ordered list of norms this participant is doing (set at /begin)."""
    return flask_session.get('norms') or []


# ---------------------------------------------------------------------------
# Norm assignment
#
# 'balanced' (default) hands out a fixed counterbalanced schedule instead of an
# independent random draw per participant, so the design is exactly filled:
# with K test norms and B norms per participant, participant i (in arrival
# order) gets pool[i], pool[i+1], ..., pool[i+B-1] (indices mod K). Over the
# first K participants every norm is played by exactly B participants and
# appears exactly once in each order position. E.g. K=6, B=2 -> 6 participants,
# each doing 2 norms, each norm played by 2 people, each norm first for one of
# them and second for the other.
#
# Slots are handed out in arrival order and persisted to data/, so restarting
# the server mid-study continues the schedule rather than restarting it, and a
# participant id that comes back gets the norms it was already assigned.
# ---------------------------------------------------------------------------
_ASSIGNMENTS_PATH = os.path.join(_DATA_DIR, '_assignments.json')
_assign_lock = threading.Lock()


def _testable_pool():
    """Test-norm pool: the configured pool minus the tutorial norm (used in the
    shared intro, so reusing it as a test norm would leak its rule)."""
    return [n for n in _server_config['norm_pool']
            if n != _server_config['tutorial_norm']]


def _assignment_schedule():
    """One norm list per participant slot: the fixed study design when the
    server runs in design mode, else the counterbalanced rotation."""
    if _server_config.get('assignment') == 'design':
        return [list(row) for row in _STUDY_DESIGN['schedule']]
    pool = _testable_pool()
    k = len(pool)
    b = min(_server_config['blocks'], k)
    return [[pool[(i + j) % k] for j in range(b)] for i in range(k)]


def _load_assignments() -> dict:
    try:
        with open(_ASSIGNMENTS_PATH, encoding='utf-8') as f:
            rec = json.load(f)
        if isinstance(rec, dict) and isinstance(rec.get('assigned'), dict):
            return rec
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f'[exp] WARN: could not read {_ASSIGNMENTS_PATH}: {e}')
    return {'schedule': [], 'next_slot': 0, 'assigned': {}}


def _prior_study_assignments(pid: str) -> list:
    """This participant id's assignments in EARLIER studies (the archived
    ledgers data/_assignments_<timestamp>.json), oldest first. Its length
    numbers the data folder of a participant who takes part again
    (see _participant_dirname)."""
    prior = []
    for path in sorted(glob.glob(os.path.join(_DATA_DIR, '_assignments_*.json'))):
        try:
            with open(path, encoding='utf-8') as f:
                rec = json.load(f)
            entry = rec.get('assigned', {}).get(pid)
        except Exception:
            continue
        if entry:
            prior.append({'ledger': os.path.basename(path),
                          'norms': list(entry.get('norms', []))})
    return prior


def _save_assignments(rec: dict) -> None:
    try:
        with open(_ASSIGNMENTS_PATH, 'w') as f:
            json.dump(rec, f, indent=2)
    except Exception as e:
        print(f'[exp] WARN: assignment ledger write failed: {e}')


def _assign_norms(pid: str):
    """Norms for this participant, in the order they will play them.

    balanced: take the next slot of the counterbalanced schedule (see above).
    design:   take the next slot of the fixed study design (study_design.json).
    random:   the old behaviour — an independent random sample per participant.

    balanced/design hand out the LOWEST slot nobody currently holds, so a slot
    freed by /admin/release (a dropout) is refilled by the next arrival before
    the design wraps around; once every slot is held the design starts over.
    """
    if _server_config.get('assignment') not in ('balanced', 'design'):
        pool = _testable_pool()
        return random.sample(pool, min(_server_config['blocks'], len(pool)))

    schedule = _assignment_schedule()
    with _assign_lock:
        rec = _load_assignments()
        # A changed norm pool / blocks setting invalidates the running count:
        # archive the old ledger and start the new design from slot 0.
        if rec.get('schedule') and rec['schedule'] != schedule:
            try:
                os.replace(_ASSIGNMENTS_PATH,
                           _ASSIGNMENTS_PATH.replace(
                               '.json', f'_{time.strftime("%Y%m%d_%H%M%S")}.json'))
            except Exception as e:
                print(f'[exp] WARN: could not archive old assignment ledger: {e}')
            print('[exp] note: norm pool/blocks changed — starting a fresh '
                  'assignment schedule.')
            rec = {'schedule': [], 'next_slot': 0, 'assigned': {}}
        prior = rec['assigned'].get(pid)
        if prior:  # same participant id coming back: keep their original norms
            return list(prior['norms'])
        taken = {int(e.get('slot', -1)) for e in rec['assigned'].values()}
        free = [s for s in range(len(schedule)) if s not in taken]
        slot = free[0] if free else int(rec.get('next_slot', 0))
        norms = list(schedule[slot % len(schedule)])
        rec['schedule'] = schedule
        rec['next_slot'] = max(int(rec.get('next_slot', 0)), slot + 1)
        entry = {'slot': slot, 'norms': norms,
                 'timestamp': time.strftime('%Y%m%d_%H%M%S')}
        if _STUDY_DESIGN:
            entry['categories'] = [_norm_category(n) for n in norms]
        # Data folder for this study: '<id>' the first time this id takes part
        # in a study, '<id>_2' / '<id>_3' / ... for later studies (see
        # _participant_dirname), so each study's data stays separate.
        n_prior = len(_prior_study_assignments(pid))
        entry['data_dir'] = _safe_pid(pid) if n_prior == 0 else f'{_safe_pid(pid)}_{n_prior + 1}'
        rec['assigned'][pid] = entry
        _save_assignments(rec)
        print(f'[exp] assigned {pid} -> slot {slot} ({slot % len(schedule) + 1}'
              f'/{len(schedule)} of the design): {norms}'
              + (f'  [data folder: {entry["data_dir"]}]' if n_prior else ''))
        return norms


def _valid_block(block):
    return 0 <= block < len(_assigned_norms())


@app.route('/')
def welcome():
    return _no_cache(make_response(render_template('welcome.html', config=_server_config)))


@app.route('/begin', methods=['POST'])
def begin():
    pid = (request.form.get('participant_id') or '').strip()
    if not pid:
        pid = 'p_' + uuid.uuid4().hex[:8]
    flask_session['participant_id'] = pid
    flask_session['consent'] = bool(request.form.get('consent'))
    norms = _assign_norms(pid)
    flask_session['norms'] = norms

    # Record the per-participant norm assignment/order for analysis.
    try:
        safe_pid = _participant_dirname(pid)
        ts = time.strftime('%Y%m%d_%H%M%S')
        pdir = os.path.join(_DATA_DIR, safe_pid)
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(pdir, f'{safe_pid}_assignment_{ts}.json'), 'w') as f:
            json.dump({'participant_id': safe_pid, 'prolific_id': _safe_pid(pid),
                       'norm_order': norms,
                       'norm_categories': [_norm_category(n) for n in norms],
                       'tutorial_norm': _server_config['tutorial_norm'],
                       'world_type': _server_config['world_type']}, f, indent=2)
    except Exception as e:
        print(f'[log] assignment write failed: {e}')

    return redirect(url_for('cover_story'))


# ---- Shared intro stages ----
@app.route('/story')
def cover_story():
    """Setting (narrative): introduce the virtual community, its three shirt
    colours, and the idea of hidden (possibly non-real-world) norms. The café
    glimpse and the task/scoring details now live on the Tutorial page."""
    if not _participant():
        return redirect(url_for('welcome'))
    return _no_cache(make_response(render_template(
        'cover_story.html', config=_server_config,
        total_blocks=len(_assigned_norms()),
        next_url=url_for('setting'))))


@app.route('/setting')
def setting():
    """Tutorial: the world, how to read the characters, norm-violation signals,
    and the scoring/task rules — all in one multi-part page."""
    if not _participant():
        return redirect(url_for('welcome'))
    return _no_cache(make_response(render_template(
        'setting.html', config=_server_config,
        total_blocks=len(_assigned_norms()),
        next_url=url_for('quiz'))))


@app.route('/practice-intro')
def practice_intro():
    """Brief interstitial shown after the quiz, before the practice round:
    tells the participant they'll practice first, then start the actual study."""
    if not _participant():
        return redirect(url_for('welcome'))
    return _no_cache(make_response(render_template(
        'practice_intro.html', config=_server_config,
        total_blocks=len(_assigned_norms()),
        next_url=url_for('practice'))))


@app.route('/practice')
def practice():
    if not _participant():
        return redirect(url_for('welcome'))
    return _no_cache(make_response(render_template(
        'experiment.html', config=_server_config,
        stage='practice', mode='play', phase='practice',
        norm_type=_server_config['tutorial_norm'],
        world_type=_server_config['world_type'],
        block=0, total_blocks=len(_assigned_norms()), episode=0,
        page_title='Practice — try the controls',
        next_url=url_for('study_start'),
        participant_id=flask_session['participant_id'])))


@app.route('/study-start')
def study_start():
    """Milestone page between the practice round and the first scenario: marks
    practice as over and the real study as beginning. Shown once, only here."""
    if not _participant():
        return redirect(url_for('welcome'))
    return _no_cache(make_response(render_template(
        'study_start.html', config=_server_config,
        total_blocks=len(_assigned_norms()),
        next_url=url_for('norm_intro', block=0))))


# Comprehension quiz. Each item: (prompt, [options], index-of-correct-option[,
# option_images]). When option_images is present it is a list parallel to
# [options]; each entry is a list of sprite filenames (stacked, so ['blue.png',
# 'llm_star.png'] draws the base character with the gold star on top) and the
# quiz page renders those images instead of the option text.
_QUIZ_QUESTIONS = [
    ("We will assign a character to you. Which one is you?",
     ["The orange character", "The green character", "The blue character",
      "The character marked with a gold star"],
     3,
     [["orange.png"], ["green.png"], ["blue.png"], ["blue.png", "llm_star.png"]]),
    ("In each scenario, what are you trying to do?",
     ["Just reach my goal locations as fast as possible.",
      "Reach my goal locations AND figure out the social norm.",
      "Collide with as many characters as possible.",
      "Memorize where every other character goes."],
     1),
    ("Will we tell you what the hidden social norm is?",
     ["Yes, it is shown at the top of the screen.",
      "Yes, another character explains it before each round.",
      "No — I have to work it out myself from the feedback.",
      "No, but there is never any way to tell if I broke it."],
     2),
    ("How do you know you have broken the hidden rule?",
     ["Nothing happens; I just lose silently.",
      "A “Rule Violation” popup appears and a red ✗ shows on my character.",
      "My character turns invisible.",
      "The round restarts from the beginning."],
     1),
    ("You bump into another character. Is that a norm violation?",
     ["Yes — bumping into someone is always a norm violation.",
      "No — a collision is just a collision. It costs points, but the social "
      "norm is a separate rule and I would not get a violation for it.",
      "Yes, but only if the other character wears a different colour.",
      "Only if I bump into the same character more than once."],
     1),
]


@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    """Comprehension check after the tutorial, before the practice round. The
    participant must answer every question correctly (enforced client-side) to
    continue; their final answers + number of attempts are logged."""
    if not _participant():
        return redirect(url_for('welcome'))
    if request.method == 'POST':
        answers = [request.form.get(f'q{i}', '') for i in range(len(_QUIZ_QUESTIONS))]
        try:
            attempts = int(request.form.get('attempts', '1'))
        except (TypeError, ValueError):
            attempts = None
        _save_quiz(answers, attempts)
        return redirect(url_for('practice_intro'))
    sprite_base = url_for('static', filename='sprites/')
    questions = []
    for item in _QUIZ_QUESTIONS:
        prompt, opts, correct = item[0], item[1], item[2]
        option_images = item[3] if len(item) > 3 else None
        if option_images:
            option_images = [[sprite_base + fn for fn in imgs]
                             for imgs in option_images]
        questions.append({'prompt': prompt, 'options': opts,
                          'correct': correct, 'images': option_images})
    return _no_cache(make_response(render_template(
        'quiz.html', config=_server_config,
        total_blocks=len(_assigned_norms()),
        questions=questions)))


def _save_quiz(answers, attempts):
    """Persist the comprehension-quiz answers to the participant's data folder."""
    pid = flask_session.get('participant_id', 'anon')
    safe_pid = _participant_dirname(pid)
    ts = time.strftime('%Y%m%d_%H%M%S')
    pdir = os.path.join(_DATA_DIR, safe_pid)
    os.makedirs(pdir, exist_ok=True)
    path = os.path.join(pdir, f'{safe_pid}_quiz_{ts}.json')
    try:
        with open(path, 'w') as f:
            json.dump({'participant_id': safe_pid, 'answers': answers,
                       'attempts': attempts, 'timestamp': ts}, f, indent=2)
    except Exception as e:
        print(f'[log] quiz write failed: {e}')


@app.route('/intro/<int:block>')
def norm_intro(block):
    """Instruction page shown before each norm's interactive round: infer the
    hidden rule while you play; the round runs its full step budget."""
    if not _participant():
        return redirect(url_for('welcome'))
    if not _valid_block(block):
        return redirect(url_for('done'))
    return _no_cache(make_response(render_template(
        'norm_intro.html', config=_server_config,
        block=block, total_blocks=len(_assigned_norms()),
        next_url=url_for('main', block=block, episode=0))))


@app.route('/guess/<int:block>', methods=['GET', 'POST'])
def guess(block):
    """Shown after the interactive round, before the test rounds: collect the
    participant's guess of the rule + confidence, and explain the test stage."""
    if not _participant():
        return redirect(url_for('welcome'))
    if not _valid_block(block):
        return redirect(url_for('done'))
    norm_name = _assigned_norms()[block]
    if request.method == 'POST':
        _save_guess(block, norm_name,
                    request.form.get('norm_guess', ''),
                    request.form.get('confidence', ''))
        return redirect(url_for('test_intro', block=block))
    return _no_cache(make_response(render_template(
        'guess.html', config=_server_config,
        block=block, total_blocks=len(_assigned_norms()),
        norm_type=norm_name)))


@app.route('/test-intro/<int:block>')
def test_intro(block):
    """Milestone page between a block's learning round and its test rounds:
    marks learning as over and frames the tests as applying what was learned."""
    if not _participant():
        return redirect(url_for('welcome'))
    if not _valid_block(block):
        return redirect(url_for('done'))
    norm_name = _assigned_norms()[block]
    # The new-map world of the run THIS participant drew.
    new_map_world = _round_world(
        norm_name, NEW_MAP_EPISODE,
        _round_run_index(_participant(), norm_name, NEW_MAP_EPISODE))
    return _no_cache(make_response(render_template(
        'test_intro.html', config=_server_config,
        block=block, total_blocks=len(_assigned_norms()),
        num_tests=EPISODES_PER_NORM - 1,
        same_world=_server_config['world_type'],
        new_map_world=new_map_world,
        next_url=url_for('main', block=block, episode=1))))


@app.route('/scene/<world_type>.png')
def scene_png(world_type):
    """The empty board of one world, for the new-map tour (see _scene_png)."""
    if world_type not in WORLD_CHOICES:
        return ('unknown world', 404)
    resp = make_response(_scene_png(world_type))
    resp.headers['Content-Type'] = 'image/png'
    # Fixed per world, so let the browser keep it for the session.
    resp.headers['Cache-Control'] = 'public, max-age=3600'
    return resp


@app.route('/new-map-intro/<int:block>')
def new_map_intro(block):
    """Tour of the new-map world, shown right before the new-map test.

    The participant has only ever played in the study world, so this is the
    counterpart of the tutorial's café layout page: the same annotated-scene
    treatment (door / seat / counter / carpet), drawn for the world THIS
    participant's new-map run uses. The heading names that world; the rest of
    the copy just says the round happens somewhere they have not played before.
    """
    if not _participant():
        return redirect(url_for('welcome'))
    if not _valid_block(block):
        return redirect(url_for('done'))
    norm_name = _assigned_norms()[block]
    world_type = _round_world(
        norm_name, NEW_MAP_EPISODE,
        _round_run_index(_participant(), norm_name, NEW_MAP_EPISODE))
    scene = scene_annotation(world_type)
    return _no_cache(make_response(render_template(
        'new_map_intro.html', config=_server_config,
        block=block, total_blocks=len(_assigned_norms()),
        episode=NEW_MAP_EPISODE,
        world_type=world_type,
        world_name=world_type.replace('_', ' '),
        scene=scene,
        next_url=url_for('main', block=block, episode=NEW_MAP_EPISODE))))


def _write_guess(pid, block, norm_name, norm_guess, confidence, kind='final', step=None):
    """Persist an inferred-rule guess to the participant's data folder.

    kind='final' is the post-interactive guess page; kind='mid' is the in-round
    recurring in-round check-in. Both share a shape so they read together.
    """
    safe_pid = _participant_dirname(pid)
    try:
        conf = int(confidence)
    except (TypeError, ValueError):
        conf = None
    ts = time.strftime('%Y%m%d_%H%M%S')
    pdir = os.path.join(_DATA_DIR, safe_pid)
    os.makedirs(pdir, exist_ok=True)
    suffix = 'guess' if kind == 'final' else f'midguess_s{step}'
    path = os.path.join(pdir, f'{safe_pid}_b{block}_{norm_name}_{suffix}_{ts}.json')
    try:
        with open(path, 'w') as f:
            json.dump({'participant_id': safe_pid, 'block': block,
                       'norm_type': norm_name, 'norm_guess': norm_guess,
                       'confidence': conf, 'kind': kind, 'step': step,
                       'timestamp': ts}, f, indent=2)
    except Exception as e:
        print(f'[log] {kind} guess write failed: {e}')


def _save_guess(block, norm_name, norm_guess, confidence):
    """Persist the post-interactive guess (from the guess page form)."""
    _write_guess(flask_session.get('participant_id', 'anon'),
                 block, norm_name, norm_guess, confidence, kind='final')


# ---- Main rounds: per norm, episode 0 = interactive, 1 = same-world test,
# ---- 2 = new-map test (see the round-schedule block at the top of the file).
@app.route('/main/<int:block>/<int:episode>')
def main(block, episode):
    if not _participant():
        return redirect(url_for('welcome'))
    if not _valid_block(block) or not (0 <= episode < EPISODES_PER_NORM):
        return redirect(url_for('done'))

    norms = _assigned_norms()
    total = len(norms)
    norm_name = norms[block]

    round_stage = _episode_stage(episode)
    is_interactive = (round_stage == 'interactive')
    phase = 'interactive' if is_interactive else 'test'
    label = ('Learning round' if is_interactive
             else f'Test {episode} of {EPISODES_PER_NORM - 1}')
    # The new-map test moves to a place the participant has never played in, so
    # say so up front: the point is transferring the norm, not spotting the swap.
    run_index = _round_run_index(flask_session['participant_id'], norm_name, episode)
    round_world = _round_world(norm_name, episode, run_index)
    world_name = round_world.replace('_', ' ')
    subtitle = (f'A different place: the {world_name}. The same norm still applies.'
                if round_stage == 'new_map_test' else None)

    # Next: interactive -> the guess page; a test walks to the next test, with
    # the new-map tour in between (the next round is somewhere they have never
    # played, so they meet the place before they are scored in it); the last
    # test -> the next norm's intro page; finally -> done.
    if is_interactive:
        nxt = url_for('guess', block=block)
    elif episode + 1 == NEW_MAP_EPISODE:
        nxt = url_for('new_map_intro', block=block)
    elif episode + 1 < EPISODES_PER_NORM:
        nxt = url_for('main', block=block, episode=episode + 1)
    elif block + 1 < total:
        nxt = url_for('norm_intro', block=block + 1)
    else:
        nxt = url_for('done')

    ctx = dict(
        config=_server_config, stage='main', mode='play', phase=phase,
        norm_type=norm_name, world_type=round_world,
        block=block, total_blocks=total, episode=episode,
        page_title=f'Scenario {block + 1} of {total} · {label}',
        subtitle=subtitle, next_url=nxt,
        participant_id=flask_session['participant_id'],
        mid_guess_every=MID_GUESS_EVERY_STEPS,
    )
    return _no_cache(make_response(render_template('experiment.html', **ctx)))


@app.route('/done', methods=['GET', 'POST'])
def done():
    if not _participant():
        return redirect(url_for('welcome'))
    # POST = the participant submitted the background survey (age + gender).
    submitted = False
    if request.method == 'POST':
        _save_survey(request.form)
        submitted = True
    return _no_cache(make_response(render_template(
        'done.html', config=_server_config,
        total_blocks=len(_assigned_norms()),
        submitted=submitted,
        completion_code=COMPLETION_CODE,
        participant_id=flask_session['participant_id'])))


# Free-text survey answers, as (form field, JSON key) pairs. Add to this list and
# to the matching field in done.html to extend the survey — nothing else needs to
# change.
_SURVEY_TEXT_FIELDS = [
    ('strategy', 'strategy'),               # how they went about inferring the norm
    ('info_source_other', 'info_source_other'),  # free text beside the choice below
    ('test_mistake', 'test_mistake'),       # why they slipped in a test round
    ('culture', 'culture'),                 # country of residence / culture raised in
]
_SURVEY_MAX_TEXT = 4000


def _save_survey(form):
    """Persist the end-of-study background survey to the participant's data
    folder. `form` is the raw request form; unanswered fields save as ''."""
    pid = flask_session.get('participant_id', 'anon')
    safe_pid = _participant_dirname(pid)
    try:
        age_val = int(form.get('age', ''))
    except (TypeError, ValueError):
        age_val = None
    record = {'participant_id': safe_pid, 'age': age_val,
              'gender': (form.get('gender', '') or '').strip(),
              'info_source': (form.get('info_source', '') or '').strip()}
    for field, key in _SURVEY_TEXT_FIELDS:
        record[key] = (form.get(field, '') or '').strip()[:_SURVEY_MAX_TEXT]
    ts = time.strftime('%Y%m%d_%H%M%S')
    record['timestamp'] = ts
    pdir = os.path.join(_DATA_DIR, safe_pid)
    os.makedirs(pdir, exist_ok=True)
    path = os.path.join(pdir, f'{safe_pid}_survey_{ts}.json')
    try:
        with open(path, 'w') as f:
            json.dump(record, f, indent=2)
    except Exception as e:
        print(f'[log] survey write failed: {e}')


@app.route('/health')
def health():
    return jsonify({'ok': True, 'active_sessions': len(_sessions)})


def _check_admin():
    """Token gate for the data-download endpoints. Returns an error response
    (tuple) if access is denied, else None. Disabled unless ADMIN_TOKEN is set."""
    token = os.environ.get('ADMIN_TOKEN')
    if not token:
        return ('Admin download is disabled. Set the ADMIN_TOKEN env var to enable it.', 404)
    if request.args.get('token') != token:
        return ('Forbidden — wrong or missing ?token=', 403)
    return None


def _result_files(include_images=False):
    """All result files as (name, absolute path) pairs, where `name` is the path
    relative to experiment/human/ — so data/ files keep their `data/...` prefix
    and the step images keep their `data_figures/...` one. Recurses into the
    per-participant subfolders. The step images are excluded by default — there
    are hundreds per round, so bundling them would turn the download into a
    multi-hundred-MB zip."""
    roots = [_DATA_DIR] + ([_FIGURES_DIR] if include_images else [])
    out = []
    for base in roots:
        for root, _dirs, files in os.walk(base):
            for n in files:
                if n == '.gitignore':
                    continue
                if not include_images and n.lower().endswith('.png'):
                    continue
                path = os.path.join(root, n)
                out.append((os.path.relpath(path, _HERE), path))
    return sorted(out)


@app.route('/admin/list')
def admin_list():
    """JSON index of saved result files. Usage: /admin/list?token=YOUR_TOKEN
    Add &images=1 to include the saved step images under data_figures/."""
    denied = _check_admin()
    if denied:
        return denied
    files = [{'name': name, 'bytes': os.path.getsize(path)}
             for name, path in _result_files(
                 include_images=request.args.get('images') == '1')]
    return jsonify({'count': len(files), 'files': files})


@app.route('/admin/assignments')
def admin_assignments():
    """Who got which norms, and how many participants each norm still needs.
    Usage: /admin/assignments?token=YOUR_TOKEN"""
    denied = _check_admin()
    if denied:
        return denied
    schedule = _assignment_schedule()
    with _assign_lock:
        rec = _load_assignments()
    counts = {n: 0 for n in _testable_pool()}
    for entry in rec['assigned'].values():
        for n in entry.get('norms', []):
            counts[n] = counts.get(n, 0) + 1
    target = _server_config['blocks']  # per norm, over one full pass of the design
    taken = {int(e.get('slot', -1)) for e in rec['assigned'].values()}
    out = {
        'assignment_mode': _server_config.get('assignment'),
        'participants_per_design': len(schedule),
        'participants_assigned': len(rec['assigned']),
        'free_slots': [s for s in range(len(schedule)) if s not in taken],
        'norms_per_participant': _server_config['blocks'],
        'target_participants_per_norm': target,
        'participants_per_norm': counts,
        'remaining_per_norm': {n: max(0, target - c) for n, c in counts.items()},
        'schedule': schedule,
        'assigned': rec['assigned'],
        'released': rec.get('released', []),
    }
    if _STUDY_DESIGN:
        per_cat: dict = {}
        for n, c in counts.items():
            cat = _norm_category(n) or 'unknown'
            d = per_cat.setdefault(cat, {'norms': 0, 'plays_assigned': 0, 'plays_target': 0})
            d['norms'] += 1
            d['plays_assigned'] += c
            d['plays_target'] += target
        out['per_category'] = per_cat
    return jsonify(out)


@app.route('/admin/release')
def admin_release():
    """Free a participant's slot (a dropout who will not finish): the next
    arrival takes that slot, so the design still fills exactly. The entry is
    kept under 'released' in the ledger for the record; that participant's
    already-written round data is untouched. A released participant id that
    comes back is treated as new and gets the lowest free slot.
    Usage: /admin/release?token=YOUR_TOKEN&pid=PARTICIPANT_ID"""
    denied = _check_admin()
    if denied:
        return denied
    pid = (request.args.get('pid') or '').strip()
    with _assign_lock:
        rec = _load_assignments()
        entry = rec['assigned'].pop(pid, None)
        if entry is None:
            return jsonify({'released': False,
                            'error': f'no assignment for participant {pid!r}'}), 404
        entry = dict(entry, pid=pid, released_at=time.strftime('%Y%m%d_%H%M%S'))
        rec.setdefault('released', []).append(entry)
        _save_assignments(rec)
    print(f'[exp] released slot {entry["slot"]} of {pid}: {entry["norms"]}')
    return jsonify({'released': True, 'pid': pid, 'slot': entry['slot'],
                    'norms': entry['norms']})


@app.route('/admin/results')
def admin_results():
    """Download ALL participant data as a single zip (logs, guesses, surveys).
    Usage: open  https://<your-app>/admin/results?token=YOUR_TOKEN  in a browser.
    Add &images=1 to also bundle the saved step images (much larger)."""
    denied = _check_admin()
    if denied:
        return denied
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, path in _result_files(
                include_images=request.args.get('images') == '1'):
            zf.write(path, arcname=name)
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', as_attachment=True,
                     download_name=f'normgym_results_{time.strftime("%Y%m%d_%H%M%S")}.zip')


# ---------------------------------------------------------------------------
# Socket.IO events — per-connection sessions
# ---------------------------------------------------------------------------
@socketio.on('connect')
def on_connect():
    print(f'[exp] client connected: {request.sid}')


@socketio.on('disconnect')
def on_disconnect():
    _stop_session(request.sid)
    print('[exp] client disconnected')


def _stop_session(sid):
    with _sessions_lock:
        sess = _sessions.pop(sid, None)
    if sess:
        sess.cleanup()


@socketio.on('start_session')
def on_start_session(data):
    """Start a round for THIS connection.

    data: { participant_id, phase, norm_type, world_type, block, episode }
    phase: 'demo' (example) | 'practice' | 'interactive' | 'test'
    The norm/world come from the page the participant is on.
    """
    sid = request.sid
    _stop_session(sid)  # clear any prior round on this connection

    data = data or {}
    participant_id = data.get('participant_id') or 'anon'
    phase = data.get('phase', 'test')
    block = int(data.get('block', 0))
    episode = int(data.get('episode', 0))

    cfg = _server_config
    # Use the norm the participant's current page is on; fall back to the
    # configured default if a client sends something unknown.
    norm_type = data.get('norm_type')
    if norm_type not in NORM_CHOICES:
        norm_type = cfg['norm_pool'][0]
    # Which pipeline stage this round is data for. Main rounds derive it from
    # the episode (the client only ever reports the phase, and the world a test
    # runs in follows from the stage), so a stale page can't mix the two up.
    stage = _episode_stage(episode) if phase in ('interactive', 'test') else phase

    # Which of the pipeline's runs of this stage the participant drew, and the
    # world that run uses. Both are recomputed here from (participant, norm,
    # episode) rather than taken from the client, so they always match what the
    # page showed. Demo/practice take the page's world.
    if phase in ('interactive', 'test'):
        run_index = _round_run_index(participant_id, norm_type, episode)
        world_type = _round_world(norm_type, episode, run_index)
    else:
        run_index = 0
        world_type = data.get('world_type')
        if world_type not in WORLD_CHOICES:
            world_type = cfg['world_type']
    capacity = _round_capacity(norm_type)
    base_seed = _round_base_seed(norm_type)

    # Round seed, derived exactly as run_experiment.py derives its stage seeds
    # (see _round_seed). Practice/demo have no pipeline counterpart; they run on
    # the tutorial norm's base seed so they are reproducible too.
    if phase in ('interactive', 'test'):
        round_seed = _round_seed(norm_type, episode, run_index)
    else:
        round_seed = base_seed

    # Phase drives the run shape:
    #   demo        -> no human, autoplay, capped at demo_max_steps
    #   practice    -> human plays, 'test' engine (bounded, ends when the human finishes)
    #   interactive -> human plays, 'interactive' engine (continuous, replacements)
    #   test        -> human plays, 'test' engine (scripted turnover, ends when the human finishes)
    run_mode = 'interactive' if phase == 'interactive' else 'test'

    # Step budgets come from the shared pipeline config so the human faces the
    # same time limits as the llm/rl subjects: max_steps for the interactive
    # stage, test_max_steps for test rounds. Practice uses the test budget; the
    # demo keeps its own cap.
    if phase == 'demo':
        env_max_steps = cfg['demo_max_steps']
    elif phase == 'interactive':
        env_max_steps = cfg['max_steps']
    else:
        env_max_steps = cfg['test_max_steps']

    # Scripted no-norm agents (norm=None, so the map shows a non-compliant
    # character alongside the norm-followers): the interactive round mixes in
    # config num_no_norm_agents exactly like the pipeline's interactive stage;
    # practice keeps one so the intro shows a rule-breaker; demo stays
    # all-compliant so it cleanly teaches the norm, and test stays all-compliant
    # for clean measurement (as in the pipeline's test runs).
    if phase == 'interactive':
        num_no_norm_agents = cfg['num_no_norm_agents']
    elif phase == 'practice':
        num_no_norm_agents = min(1, capacity - 1)
    else:
        num_no_norm_agents = 0

    sess = Session(sid, participant_id, phase, stage, norm_type, world_type,
                   block, episode, run_index, cfg)
    with _sessions_lock:
        _sessions[sid] = sess

    with _setup_lock:
        # Everything from the seed to the pools replays the pipeline's draw
        # sequence in the pipeline's order; see the note above _setup_lock.
        random.seed(round_seed)
        subject_positions = (set() if phase == 'demo'
                             else _random_subject_positions(capacity, norm_type))

        # Always bound the episode so a round can't run forever (some norms'
        # oracle agents never all reach 'done'). 'test' mode still early-breaks
        # when the human finishes; the cap is the truncation point.
        env = _make_env(norm_type, sess.frames_dir, capacity, world_type, env_max_steps,
                        cfg['use_isometric'], cfg['map_type'], cfg['interpolation_steps'])
        sess.env = env

        # Subject colors follow the same fixed per-norm schedule as the
        # pipeline: interactive incarnations walk down the schedule from index
        # 0; run k of a test stage takes schedule(stage runs)[k-1], so the
        # participant wears the color the llm/rl subject wore in the very run
        # they drew. The schedule uses a private RNG, so the global draw stream
        # is untouched.
        pool_size = capacity * 20
        if phase == 'interactive':
            subject_pool_size = pool_size // capacity
            customer_pool_size = pool_size - subject_pool_size
            subject_classes = subject_class_schedule(base_seed, 1 + subject_pool_size)
            subject_class = subject_classes[0]
        else:
            subject_pool_size = 0
            customer_pool_size = pool_size
            subject_classes = []
            subject_class = None
            if phase == 'test':
                stage_runs = _stage_run_count(norm_type, stage)
                subject_class = subject_class_schedule(
                    base_seed, stage_runs)[run_index - 1]

        initial_agents = build_initial_agents(
            capacity, subject_positions, env.world, norm_type, None,
            ground_truth='without',
            subject_kind='human',
            prompt_stage='test' if phase == 'test' else 'interactive',
            num_no_norm_agents=num_no_norm_agents,
            subject_class=subject_class,
        )

        # Interactive replacement pool for the participant's next characters —
        # built before the scripted pool, in pipeline order.
        agent_id_counter = capacity + 1
        waiting_human: deque = deque()
        for k in range(subject_pool_size):
            waiting_human.append(make_agent(
                f"C{agent_id_counter}", env.world, norm_type, None,
                subject_kind='human',
                ground_truth='without',
                prompt_stage='interactive',
                subject_class=subject_classes[k + 1],
            ))
            agent_id_counter += 1

        background_slots = capacity - len(subject_positions)
        waiting_norm_customer, waiting_no_norm_customer, _ = build_scripted_pool(
            customer_pool_size, background_slots, num_no_norm_agents,
            env.world, norm_type, None, ground_truth='without',
            first_agent_number=agent_id_counter,
        )

    # Wire the participant's browser into every human-controlled character.
    for agent in list(initial_agents) + list(waiting_human):
        if isinstance(agent, HumanAgent):
            agent.action_queue = sess.action_queue

    no_norm_ids = sorted(a.agent_id for a in initial_agents
                         if not isinstance(a, HumanAgent) and a.norm is None)
    sess.logger.log('round_start', phase=phase, stage=stage, mode=run_mode,
                    stage_run=run_index,
                    pipeline_run_id=sess.logger.pipeline_run_id,
                    block=block, episode=episode, random_seed=round_seed,
                    human_positions=sorted(subject_positions),
                    no_norm_agents=no_norm_ids,
                    capacity=capacity, max_steps=env_max_steps,
                    norm_type=norm_type, norm_category=_norm_category(norm_type),
                    world_type=world_type)

    thread = threading.Thread(
        target=_game_loop,
        args=(sess, initial_agents, waiting_human, waiting_norm_customer,
              waiting_no_norm_customer, norm_type, run_mode),
        daemon=True,
    )
    sess.thread = thread
    thread.start()
    print(f'[exp] round started: pid={participant_id} ep{episode} '
          f'run={sess.logger.pipeline_run_id} '
          f'{world_type}/{norm_type} cap={capacity} seed={round_seed}')


@socketio.on('player_action')
def on_player_action(data):
    _str_to_action = {
        'up': Action.MOVE_UP, 'down': Action.MOVE_DOWN,
        'left': Action.MOVE_LEFT, 'right': Action.MOVE_RIGHT,
        'wait': Action.WAIT,
    }
    action_str = (data or {}).get('action', 'wait')
    action = _str_to_action.get(action_str, Action.WAIT)
    with _sessions_lock:
        sess = _sessions.get(request.sid)
    if sess:
        sess.logger.log('action', action=action_str,
                        step=sess.env.current_step if sess.env else None)
        sess.action_queue.put(action)


@socketio.on('mid_guess')
def on_mid_guess(data):
    """In-round check-in (every MID_GUESS_EVERY_STEPS steps): the participant's
    guess of the norm so far, plus confidence. Logged to the round timeline and
    written as its own data file alongside the final guess."""
    with _sessions_lock:
        sess = _sessions.get(request.sid)
    if not sess:
        return
    payload = data or {}
    guess = str(payload.get('guess', ''))[:2000]
    confidence = payload.get('confidence')
    step = sess.env.current_step if sess.env else payload.get('step')
    sess.logger.log('mid_guess', step=step, norm_guess=guess, confidence=confidence)
    _write_guess(sess.participant_id, sess.block, sess.norm_type,
                 guess, confidence, kind='mid', step=step)


@socketio.on('reset_game')
def on_reset_game():
    _stop_session(request.sid)
    socketio.emit('reset_ack', {}, room=request.sid)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global _server_config
    parser = argparse.ArgumentParser(description='normGym human-experiment web app')
    # host/port default from env so cloud platforms (Render sets $PORT) work
    # without passing flags.
    parser.add_argument('--host', type=str, default=os.environ.get('HOST', '0.0.0.0'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5001)))
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--capacity', type=int,
                        default=int(os.environ['CAPACITY']) if os.environ.get('CAPACITY') else None,
                        help='Max agents on the map at once (default: per-norm '
                             'capacity from experiment/run/experiment_config.json, '
                             'same as run_experiment.py).')
    parser.add_argument('--max_steps', type=int,
                        default=int(os.environ['MAX_STEPS']) if os.environ.get('MAX_STEPS') else None,
                        help='Step cap for the interactive round (default: '
                             'max_steps from the shared experiment config).')
    parser.add_argument('--test_max_steps', type=int,
                        default=int(os.environ['TEST_MAX_STEPS']) if os.environ.get('TEST_MAX_STEPS') else None,
                        help='Step cap per practice/test round (default: '
                             'test_max_steps from the shared experiment config).')
    parser.add_argument('--demo_max_steps', type=int,
                        default=int(os.environ.get('DEMO_MAX_STEPS', 80)),
                        help='Hard cap on the autoplay Example round (default: 80).')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Override the per-norm seeds from the shared experiment '
                             'config for every round (default: use the per-norm '
                             'seeds, identical to the llm/rl runs).')
    parser.add_argument('--norm_types', type=str, nargs='+', default=None,
                        choices=NORM_CHOICES, metavar='NORM',
                        help='Pool of norms. Each participant is randomly assigned '
                             '--blocks of them, in randomized order.')
    parser.add_argument('--norm_type', type=str, default=None, choices=NORM_CHOICES,
                        help='Single-norm shorthand (equivalent to --norm_types X).')
    parser.add_argument('--blocks', type=int, default=int(os.environ.get('BLOCKS', 2)),
                        help='How many norms each participant does (default: 2). '
                             'Clamped to the size of the norm pool.')
    parser.add_argument('--assignment', type=str,
                        default=os.environ.get('ASSIGNMENT', 'balanced'),
                        choices=['balanced', 'design', 'random'],
                        help="How norms are assigned to participants. 'balanced' "
                             '(default) walks a counterbalanced rotation in arrival '
                             'order, so with K test norms and --blocks B, K '
                             'participants cover every norm exactly B times, each '
                             'norm appearing once in each order position. '
                             "'design' walks the fixed study design in --design_file "
                             '(every norm played twice, once per order position, '
                             'the two norms of a participant from different '
                             'categories); the norm pool and --blocks come from '
                             "the file. 'random' samples independently per "
                             'participant (unbalanced).')
    parser.add_argument('--design_file', type=str,
                        default=os.environ.get('DESIGN_FILE', _DEFAULT_DESIGN_PATH),
                        help='Study design for --assignment design (default: '
                             'experiment/human/study_design.json, built by '
                             'make_study_design.py).')
    parser.add_argument('--tutorial_norm', type=str,
                        default=os.environ.get('TUTORIAL_NORM', 'straight_line'),
                        choices=NORM_CHOICES,
                        help='Norm used for the shared Example + Practice intro. '
                             'Excluded from the test pool so it cannot leak a test '
                             'norm (default: straight_line).')
    parser.add_argument('--world_type', type=str,
                        default=os.environ.get('WORLD_TYPE')
                        or _PIPELINE_CONFIG.get('world_type', 'cafe'),
                        choices=WORLD_CHOICES)
    parser.add_argument('--isometric', action='store_true')
    parser.add_argument('--map_type', type=str, default='default',
                        choices=['default', 'navigation', 'affordance'])
    parser.add_argument('--interpolation_steps', type=int,
                        default=int(os.environ.get('INTERPOLATION_STEPS', 4)),
                        help='Intermediate frames between steps for smooth motion '
                             '(default: 4, same as runner/human_play.py).')
    args = parser.parse_args()

    # Build the norm pool from --norm_types / --norm_type, falling back to the
    # NORM_TYPES env var (space-separated) for cloud deploys. De-duplicated.
    norm_types = args.norm_types
    if not norm_types and os.environ.get('NORM_TYPES'):
        norm_types = os.environ['NORM_TYPES'].split()
    pool: list = []
    for n in (norm_types or []):
        if n not in NORM_CHOICES:
            parser.error(f'unknown norm: {n}')
        if n not in pool:
            pool.append(n)
    if args.norm_type and args.norm_type not in pool:
        pool.append(args.norm_type)

    # Fixed study design: the norm pool, the tutorial norm and --blocks are
    # whatever the design file says, so the served study is exactly the
    # committed design. --norm_types may be omitted; if given it must match.
    global _STUDY_DESIGN
    if args.assignment == 'design':
        try:
            _STUDY_DESIGN = _load_study_design(args.design_file)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            parser.error(f'cannot load study design {args.design_file}: {e}')
        design_norms = sorted({n for row in _STUDY_DESIGN['schedule'] for n in row})
        unknown = [n for n in design_norms if n not in NORM_CHOICES]
        if unknown:
            parser.error(f'study design names unknown norms: {unknown}')
        design_tutorial = _STUDY_DESIGN.get('tutorial_norm')
        if design_tutorial and design_tutorial != args.tutorial_norm:
            parser.error(f"--tutorial_norm {args.tutorial_norm} but the design's "
                         f'tutorial norm is {design_tutorial}')
        if args.tutorial_norm in design_norms:
            parser.error(f'--tutorial_norm {args.tutorial_norm} is a test norm of '
                         'the design; the tutorial must not leak a test norm')
        given = [n for n in pool if n != args.tutorial_norm]
        if given and set(given) != set(design_norms):
            parser.error('--norm_types must match the norms of the design file '
                         '(or be omitted in design mode)')
        pool = design_norms
        if args.blocks != _STUDY_DESIGN['blocks']:
            print(f'[exp] note: design mode — each participant does '
                  f'{_STUDY_DESIGN["blocks"]} norms (design file), ignoring '
                  f'--blocks {args.blocks}.')
        args.blocks = _STUDY_DESIGN['blocks']
    if not pool:
        parser.error('provide at least one norm via --norm_types, --norm_type, '
                     'or the NORM_TYPES env var')

    # Test norms are drawn from the pool excluding the tutorial norm (the intro
    # uses the tutorial norm, so reusing it as a test norm would leak its rule).
    testable = [n for n in pool if n != args.tutorial_norm]
    if not testable:
        parser.error('the norm pool only contains the tutorial norm '
                     f'({args.tutorial_norm}); add test norms via --norm_types, '
                     'or pick a different --tutorial_norm.')

    blocks = max(1, min(args.blocks, len(testable)))
    if blocks < args.blocks:
        print(f'[exp] note: {len(testable)} testable norm(s) available; each '
              f'participant does {blocks} (requested {args.blocks}).')

    # Settings shared with the llm/rl pipeline: CLI flag > shared experiment
    # config > built-in fallback, exactly like run_experiment.py. Seeds and
    # capacity stay None here when not given on the CLI — they resolve per norm
    # at round start (_round_base_seed / _round_capacity).
    max_steps = (args.max_steps if args.max_steps is not None
                 else _PIPELINE_CONFIG.get('max_steps', 200))
    test_max_steps = (args.test_max_steps if args.test_max_steps is not None
                      else _PIPELINE_CONFIG.get('test_max_steps', 100))
    # The pipeline's same-world test count. A participant plays ONE of those
    # runs, drawn per participant (see the round-schedule block at the top of
    # the file), and the count also fixes where the pipeline's new-map seeds
    # start, so a human new-map test lands on a real pipeline new-map run.
    num_tests = _PIPELINE_CONFIG.get('num_tests', 3)
    num_no_norm_agents = _PIPELINE_CONFIG.get('num_no_norm_agents', 1)

    _server_config = {
        'norm_pool': pool,
        'tutorial_norm': args.tutorial_norm,
        'blocks': blocks,
        'assignment': args.assignment,
        'world_type': args.world_type,
        'capacity': args.capacity,
        'max_steps': max_steps,
        'test_max_steps': test_max_steps,
        'num_tests': num_tests,
        'num_no_norm_agents': num_no_norm_agents,
        'random_seed': args.random_seed,
        'demo_max_steps': args.demo_max_steps,
        'map_type': args.map_type,
        'use_isometric': args.isometric,
        'interpolation_steps': args.interpolation_steps,
    }
    print(f'[exp] normGym human-experiment server on http://{args.host}:{args.port}')
    print(f'[exp] shared experiment config: {os.path.abspath(_PIPELINE_CONFIG_PATH)} '
          f'({"loaded" if _PIPELINE_CONFIG else "NOT FOUND — using fallbacks"})')
    print(f'[exp] tutorial norm: {args.tutorial_norm}  |  test pool: {testable}  '
          f'|  blocks/participant: {blocks}')
    if args.assignment in ('balanced', 'design'):
        schedule = _assignment_schedule()
        done = len(_load_assignments()['assigned'])
        print(f'[exp] {args.assignment} assignment: {len(schedule)} participants x {blocks} norms '
              f'= each norm played by {blocks} participant(s). '
              f'{done} participant(s) already assigned.')
        if _STUDY_DESIGN:
            from collections import Counter as _Counter
            sizes = _Counter(_norm_category(n) for n in pool)
            print(f'[exp] design file: {os.path.abspath(args.design_file)}  |  '
                  f'categories: {dict(sizes)}')
        for i, norms in enumerate(schedule):
            print(f'[exp]   participant {i + 1}: {" -> ".join(norms)}')
    else:
        print('[exp] assignment: random (independent sample per participant)')
    print(f'[exp] config: {_server_config}')
    print(f'[exp] data -> {os.path.abspath(_DATA_DIR)}')
    print(f'[exp] figures -> {os.path.abspath(_FIGURES_DIR)}')
    socketio.run(app, host=args.host, port=args.port, debug=args.debug,
                 allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
