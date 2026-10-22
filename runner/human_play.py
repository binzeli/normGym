"""Free play: drive one character around a world under a norm, in the browser.

The human study's learning round with no step limit and none of the study
around it (no intro pages, check-ins, tips or data logging): when your
character reaches its goal and leaves, you take over a new one and play goes
on. The game engine, page script and styles are the human study's own
(experiment/human/), reused unchanged.

Usage (from the repo root, or via scripts/run_human_play.sh):
    python runner/human_play.py --norm_type edge_left_line --world_type cafe
then open http://localhost:5001.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# First: the study app monkey-patches eventlet before socket/threading load.
import experiment.human.app as study  # noqa: E402

import argparse  # noqa: E402
import queue  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
from collections import deque  # noqa: E402

from flask import Flask, render_template, request  # noqa: E402
from flask_socketio import SocketIO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.join(HERE, '..', 'experiment', 'human')
# The study's frames are deleted once this many newer ones exist; play has no
# step limit, so the render dir would otherwise grow without bound.
KEEP_FRAMES = 100

app = Flask(__name__, template_folder=os.path.join(HERE, 'templates'),
            static_folder=os.path.join(STUDY_DIR, 'static'))
socketio = SocketIO(app, cors_allowed_origins='*', async_mode=study._ASYNC_MODE)
# The study's game loop emits through its module-level socketio.
study.socketio = socketio

_config: dict = {}
_sessions: dict = {}
_sessions_lock = threading.Lock()


class _NullLogger:
    """Same interface as the study's ParticipantLogger; writes nothing."""
    events_path = None
    pipeline_run_id = None

    def log(self, event, **data):
        pass

    def discard(self):
        pass

    def write_summary(self, summary):
        pass


class _EndlessPool(deque):
    """A replacement pool that is never empty: it refills itself when drained."""

    def __init__(self, refill):
        super().__init__()
        self._refill = refill

    def __bool__(self):
        return True

    def popleft(self):
        if not len(self):
            self._refill(self)
        return super().popleft()


class PlaySession:
    """The attributes the study's game loop reads from its Session."""

    def __init__(self, sid):
        self.sid = sid
        self.phase = 'play'
        self.frames_archive_dir = None
        self.logger = _NullLogger()
        self.action_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.frames_dir = tempfile.mkdtemp(prefix='normgym_play_')
        self.env = None

    def cleanup(self):
        self.stop_event.set()
        self.action_queue.put(study.Action.WAIT)  # unblock a thread waiting on get()
        shutil.rmtree(self.frames_dir, ignore_errors=True)


def _prune_frames(sess):
    """Delete all but the newest KEEP_FRAMES frames while the session runs."""
    while not sess.stop_event.is_set():
        socketio.sleep(5)
        try:
            frames = sorted((f for f in os.listdir(sess.frames_dir) if f.endswith('.png')),
                            key=lambda f: os.path.getmtime(os.path.join(sess.frames_dir, f)))
        except OSError:
            continue
        for f in frames[:-KEEP_FRAMES]:
            try:
                os.remove(os.path.join(sess.frames_dir, f))
            except OSError:
                pass


@app.route('/')
def play():
    norm = _config['norm_type']
    world = _config['world_type']
    return render_template(
        'human_play.html', norm_type=norm, world_type=world,
        world_name=world.replace('_', ' '),
        norm_description=study._NORM_DATA.get(norm, {}).get('description'))


@socketio.on('disconnect')
def on_disconnect():
    _stop(request.sid)


def _stop(sid):
    with _sessions_lock:
        sess = _sessions.pop(sid, None)
    if sess:
        sess.cleanup()


@socketio.on('start_session')
def on_start_session(data):
    """Start (or restart) play for this connection, set up like the study's
    learning round."""
    sid = request.sid
    _stop(sid)
    sess = PlaySession(sid)
    with _sessions_lock:
        _sessions[sid] = sess

    norm = _config['norm_type']
    world = _config['world_type']
    capacity = study._round_capacity(norm)
    num_no_norm = study._PIPELINE_CONFIG.get('num_no_norm_agents', 1)
    next_id = [capacity + 1]

    with study._setup_lock:
        random.seed(_config['random_seed'] if _config['random_seed'] is not None
                    else random.randrange(2 ** 31))
        subject_positions = study._random_subject_positions(capacity, norm)
        env = study._make_env(norm, sess.frames_dir, capacity, world, sys.maxsize,
                              _config['isometric'], _config['map_type'],
                              _config['interpolation_steps'])
        sess.env = env
        initial_agents = study.build_initial_agents(
            capacity, subject_positions, env.world, norm, None,
            ground_truth='without', subject_kind='human',
            prompt_stage='interactive', num_no_norm_agents=num_no_norm)
    human = next(a for a in initial_agents if isinstance(a, study.HumanAgent))
    human.action_queue = sess.action_queue

    def new_human(pool):
        agent = study.make_agent(f'C{next_id[0]}', env.world, norm, None,
                                 subject_kind='human', ground_truth='without',
                                 prompt_stage='interactive')
        next_id[0] += 1
        agent.action_queue = sess.action_queue
        agent.emit_callback = human.emit_callback  # set by the game loop
        pool.append(agent)

    no_norm_pool = deque()

    def new_scripted(pool):
        norm_agents, no_norm_agents, next_id[0] = study.build_scripted_pool(
            capacity * 20, capacity - len(subject_positions), num_no_norm,
            env.world, norm, None, ground_truth='without',
            first_agent_number=next_id[0])
        pool.extend(norm_agents)
        no_norm_pool.extend(no_norm_agents)

    socketio.start_background_task(_prune_frames, sess)
    thread = threading.Thread(
        target=study._game_loop,
        args=(sess, initial_agents, _EndlessPool(new_human), _EndlessPool(new_scripted),
              no_norm_pool, norm, 'interactive'),
        daemon=True)
    thread.start()
    print(f'[play] started: {world}/{norm} capacity={capacity}')


@socketio.on('player_action')
def on_player_action(data):
    actions = {'up': study.Action.MOVE_UP, 'down': study.Action.MOVE_DOWN,
               'left': study.Action.MOVE_LEFT, 'right': study.Action.MOVE_RIGHT,
               'wait': study.Action.WAIT}
    with _sessions_lock:
        sess = _sessions.get(request.sid)
    if sess:
        sess.action_queue.put(actions.get((data or {}).get('action'), study.Action.WAIT))


@socketio.on('reset_game')
def on_reset_game():
    _stop(request.sid)


def main():
    parser = argparse.ArgumentParser(description='normGym free play in the browser')
    parser.add_argument('--norm_type', type=str, required=True, choices=study.NORM_CHOICES)
    parser.add_argument('--world_type', type=str, default='cafe', choices=study.WORLD_CHOICES)
    parser.add_argument('--map_type', type=str, default='default',
                        choices=['default', 'navigation', 'affordance'])
    parser.add_argument('--capacity', type=int, default=None,
                        help="Max characters on the map (default: the norm's "
                             'benchmark capacity).')
    parser.add_argument('--isometric', action='store_true')
    parser.add_argument('--interpolation_steps', type=int, default=4)
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Seed for the setup (default: a new one each round).')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    if args.capacity is not None and args.capacity < 3:
        parser.error('--capacity must be at least 3 (you enter the map 3rd or later)')

    _config.update(norm_type=args.norm_type, world_type=args.world_type,
                   map_type=args.map_type, isometric=args.isometric,
                   interpolation_steps=args.interpolation_steps,
                   random_seed=args.random_seed)
    # _round_capacity reads the study's config: CLI --capacity, else per norm.
    study._server_config = {'capacity': args.capacity, 'random_seed': args.random_seed}
    print(f'[play] normGym free play on http://{args.host}:{args.port}  '
          f'(norm: {args.norm_type}, world: {args.world_type}; nothing is logged)')
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
