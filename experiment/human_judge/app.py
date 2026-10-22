"""normGym LLM-judge validation study — human ratings of judged model beliefs.

A small multi-page web study (plain Flask, no game engine). Each participant
rates 10 of the judged norm beliefs in norm_beliefs_500.txt. For every item
they see the TRUE rule of the world and an AI player's stated belief about
that rule, together with the LLM judge's 1-7 rating of that belief and its
one-line reason, and answer ONE question: how much they agree with the
judge's rating (1-5).

Two interface modes (--mode / MODE):
  two    wave 1: the participant first gives their OWN 1-7 rating, then the
         judge's rating + reason is revealed and they say how much they agree
         (records carry 'human_likert');
  agree  wave 2 (default): the judge's rating is shown up front and only the
         agreement question is asked ('human_likert' is null).

Flow:
    welcome / consent (Prolific id)  ->  instructions  ->  item 1..10  ->  completion code

Which items a participant gets is the fixed design in study_design.json
(built and verified by make_study_design.py: 30 slots x 10 items, stratified
over the judge's 7 levels and the 4 norm families). Slots are handed out in
arrival order (lowest free slot) and persisted to data/_assignments.json, so a
restart continues the design.

Data: data/<pid>/<pid>_item<k>_<ts>.json per rating and a
data/<pid>/<pid>_summary_<ts>.json when they finish; export_results.py flattens
everything to a CSV and computes the agreement statistics.

Run locally:
    python experiment/human_judge/app.py            # http://localhost:5002
Share a public link:
    SHARE=1 ./experiment/human_judge/run.sh
"""
import argparse
import glob
import io
import json
import os
import secrets
import sys
import threading
import time
import zipfile

from flask import (Flask, jsonify, make_response, redirect, render_template,
                   request, send_file, session as flask_session, url_for)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_study_design import check as _check_design  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, 'data')
_DEFAULT_DESIGN = os.path.join(_HERE, 'study_design.json')
os.makedirs(_DATA_DIR, exist_ok=True)

# Shown on the thank-you page (Prolific completion code) — set per study.
COMPLETION_CODE = os.environ.get('COMPLETION_CODE', 'XXXXXXXX')

# Rubric anchors — the SAME wording the LLM judge was given (see the header of
# norm_beliefs_500.txt), so a human 4/7 means the same thing as the judge's.
LIKERT7 = [
    (1, 'Unrelated or too vague to be useful'),
    (2, ''),
    (3, ''),
    (4, 'Right kind of rule, about half right'),
    (5, ''),
    (6, ''),
    (7, 'Captures the essential rule'),
]
AGREE5 = [
    (1, 'Strongly disagree'),
    (2, 'Disagree'),
    (3, 'Neutral'),
    (4, 'Agree'),
    (5, 'Strongly agree'),
]

app = Flask(__name__, template_folder=os.path.join(_HERE, 'templates'),
            static_folder=os.path.join(_HERE, 'static'))
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

# mode: 'two'   = wave-1 interface: own 1-7 rating first, then the judge is
#                 revealed and the participant says how much they agree (1-5)
#       'agree' = wave-2 interface: judge shown up front, agreement only
_config = {'design_file': _DEFAULT_DESIGN, 'mode': 'agree'}
MODES = ('two', 'agree')


def _ledger_path() -> str:
    """One slot ledger per design file, so waves can be resumed after
    switching between designs: data/_assignments.json for the default design,
    data/_assignments_<design stem>.json for any other."""
    stem = os.path.splitext(os.path.basename(_config['design_file']))[0]
    name = '_assignments.json' if stem == 'study_design' else f'_assignments_{stem}.json'
    return os.path.join(_DATA_DIR, name)
_DESIGN: dict = {}
_assign_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Design + assignment ledger
# ---------------------------------------------------------------------------
def _load_design(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        design = json.load(f)
    _check_design(design)
    return design


def _items_per_participant() -> int:
    return int(_DESIGN['items_per_participant'])


def _safe_pid(pid) -> str:
    return ''.join(c for c in str(pid) if c.isalnum() or c in '-_')[:64] or 'anon'


def _load_assignments() -> dict:
    try:
        with open(_ledger_path(), encoding='utf-8') as f:
            rec = json.load(f)
        if isinstance(rec, dict) and isinstance(rec.get('assigned'), dict):
            return rec
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f'[judge] WARN: could not read {_ledger_path()}: {e}')
    return {'schedule': [], 'next_slot': 0, 'assigned': {}}


def _save_assignments(rec: dict) -> None:
    path = _ledger_path()
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, path)


def _assign(pid: str) -> list:
    """Item list (entry ids, in presentation order) for this participant: the
    lowest slot of the design nobody holds. A returning id keeps its slot; a
    slot freed by /admin/release is refilled first; once every slot is held the
    design wraps around."""
    schedule = [list(r) for r in _DESIGN['schedule']]
    with _assign_lock:
        rec = _load_assignments()
        if rec.get('schedule') and rec['schedule'] != schedule:
            try:
                os.replace(_ledger_path(),
                           _ledger_path().replace('.json', f'_{time.strftime("%Y%m%d_%H%M%S")}.json'))
            except Exception as e:
                print(f'[judge] WARN: could not archive old ledger: {e}')
            print('[judge] note: design changed — starting a fresh assignment ledger.')
            rec = {'schedule': [], 'next_slot': 0, 'assigned': {}}
        prior = rec['assigned'].get(pid)
        if prior:
            return list(prior['items'])
        taken = {int(e.get('slot', -1)) for e in rec['assigned'].values()}
        free = [s for s in range(len(schedule)) if s not in taken]
        slot = free[0] if free else int(rec.get('next_slot', 0))
        items = list(schedule[slot % len(schedule)])
        rec['schedule'] = schedule
        rec['next_slot'] = max(int(rec.get('next_slot', 0)), slot + 1)
        rec['assigned'][pid] = {'slot': slot, 'items': items,
                                'timestamp': time.strftime('%Y%m%d_%H%M%S')}
        _save_assignments(rec)
        print(f'[judge] assigned {pid} -> slot {slot} ({slot % len(schedule) + 1}/{len(schedule)}): {items}')
        return items


# ---------------------------------------------------------------------------
# Per-participant state (disk is the source of truth; the cookie only holds
# the id, so a reload / reconnect resumes where they left off)
# ---------------------------------------------------------------------------
def _pid():
    return flask_session.get('participant_id')


def _pdir(pid: str) -> str:
    d = os.path.join(_DATA_DIR, pid)
    os.makedirs(d, exist_ok=True)
    return d


def _answered(pid: str) -> set:
    """Indices (0-based) of the items this participant has already submitted."""
    out = set()
    for path in glob.glob(os.path.join(_DATA_DIR, pid, f'{pid}_item*_*.json')):
        name = os.path.basename(path)
        try:
            out.add(int(name.split('_item')[1].split('_')[0]))
        except Exception:
            continue
    return out


def _next_index(pid: str) -> int:
    done = _answered(pid)
    n = _items_per_participant()
    for k in range(n):
        if k not in done:
            return k
    return n


def _item_for(pid: str, k: int) -> dict:
    entries = _assign(pid)
    return _DESIGN['items'][str(entries[k])]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route('/')
def welcome():
    return render_template('welcome.html', n_items=_items_per_participant())


@app.route('/begin', methods=['POST'])
def begin():
    if request.form.get('consent') != 'on':
        return redirect(url_for('welcome'))
    raw = (request.form.get('participant_id') or '').strip()
    pid = _safe_pid(raw) if raw else f'anon_{time.strftime("%Y%m%d_%H%M%S")}_{secrets.token_hex(2)}'
    flask_session['participant_id'] = pid
    items = _assign(pid)
    pdir = _pdir(pid)
    with open(os.path.join(pdir, f'{pid}_begin_{time.strftime("%Y%m%d_%H%M%S")}.json'), 'w') as f:
        json.dump({'participant_id': pid, 'raw_id': raw, 'items': items,
                   'mode': _config['mode'], 'design_file': os.path.basename(_config['design_file']),
                   'user_agent': request.headers.get('User-Agent', ''),
                   'time': time.time()}, f, indent=2)
    return redirect(url_for('instructions'))


@app.route('/instructions')
def instructions():
    if not _pid():
        return redirect(url_for('welcome'))
    tpl = 'instructions_two.html' if _config['mode'] == 'two' else 'instructions.html'
    return render_template(tpl, n_items=_items_per_participant(),
                           likert7=LIKERT7, agree5=AGREE5, stage='instructions')


@app.route('/rate/<int:k>', methods=['GET', 'POST'])
def rate(k: int):
    pid = _pid()
    if not pid:
        return redirect(url_for('welcome'))
    n = _items_per_participant()
    nxt = _next_index(pid)
    if k >= n:
        return redirect(url_for('done'))
    if k != nxt:  # already answered or skipped ahead: go to the first open item
        return redirect(url_for('rate', k=nxt) if nxt < n else url_for('done'))
    item = _item_for(pid, k)

    two = _config['mode'] == 'two'
    tpl = 'rate_two.html' if two else 'rate.html'

    if request.method == 'POST':
        try:
            agree = int(request.form.get('agree_rating', ''))
        except ValueError:
            agree = None
        own = None
        if two:
            try:
                own = int(request.form.get('own_rating', ''))
            except ValueError:
                own = None
        if agree not in range(1, 6) or (two and own not in range(1, 8)):
            return render_template(tpl, k=k, n=n, item=item, likert7=LIKERT7, agree5=AGREE5,
                                   error='Please answer every question before continuing.', stage='rate')
        ts = time.strftime('%Y%m%d_%H%M%S')
        rec = {
            'participant_id': pid,
            'index': k,
            'entry': item['entry'],
            'norm': item['norm'],
            'family': item['family'],
            'model': item['model'],
            'belief': item['belief'],
            'ground_truth_shown': item['ground_truth_short'],
            'judge_likert': item['judge_likert'],
            'judge_reason': item['judge_reason'],
            'human_likert': own,             # only asked in mode 'two'
            'agree_with_judge': agree,
            'comment': (request.form.get('comment') or '').strip(),
            'mode': _config['mode'],
            'design_file': os.path.basename(_config['design_file']),
            # client-side timings (ms since page load), as reported by the browser
            'ms_to_own_rating': _int_or_none(request.form.get('ms_own')),
            'ms_to_reveal': _int_or_none(request.form.get('ms_reveal')),
            'ms_to_answer': _int_or_none(request.form.get('ms_answer')),
            'ms_to_submit': _int_or_none(request.form.get('ms_submit')),
            'server_time': time.time(),
            'timestamp': ts,
        }
        with open(os.path.join(_pdir(pid), f'{pid}_item{k:02d}_{ts}.json'), 'w', encoding='utf-8') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        return redirect(url_for('rate', k=k + 1) if k + 1 < n else url_for('done'))

    return render_template(tpl, k=k, n=n, item=item, likert7=LIKERT7, agree5=AGREE5, error=None, stage='rate')


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@app.route('/done')
def done():
    pid = _pid()
    if not pid:
        return redirect(url_for('welcome'))
    n = _items_per_participant()
    if _next_index(pid) < n:
        return redirect(url_for('rate', k=_next_index(pid)))
    pdir = _pdir(pid)
    if not glob.glob(os.path.join(pdir, f'{pid}_summary_*.json')):
        ts = time.strftime('%Y%m%d_%H%M%S')
        ratings = []
        for path in sorted(glob.glob(os.path.join(pdir, f'{pid}_item*_*.json'))):
            with open(path, encoding='utf-8') as f:
                ratings.append(json.load(f))
        with open(os.path.join(pdir, f'{pid}_summary_{ts}.json'), 'w', encoding='utf-8') as f:
            json.dump({'participant_id': pid, 'n_items': len(ratings),
                       'items': _assign(pid), 'ratings': ratings,
                       'completion_code': COMPLETION_CODE, 'timestamp': ts,
                       'server_time': time.time()}, f, indent=2, ensure_ascii=False)
    return render_template('done.html', completion_code=COMPLETION_CODE, n_items=n, stage='done')


@app.route('/health')
def health():
    return jsonify({'ok': True})


# ---------------------------------------------------------------------------
# Admin (token-gated): progress, dropouts, data download
# ---------------------------------------------------------------------------
def _check_admin():
    token = os.environ.get('ADMIN_TOKEN')
    if not token:
        return ('Admin endpoints are disabled. Set the ADMIN_TOKEN env var to enable them.', 404)
    if request.args.get('token') != token:
        return ('Forbidden — wrong or missing ?token=', 403)
    return None


@app.route('/admin/assignments')
def admin_assignments():
    """Who holds which slot and how far they got. /admin/assignments?token=..."""
    denied = _check_admin()
    if denied:
        return denied
    with _assign_lock:
        rec = _load_assignments()
    n = _items_per_participant()
    progress = {}
    for pid, e in rec['assigned'].items():
        d = len(_answered(pid))
        progress[pid] = {'slot': e['slot'], 'answered': d, 'of': n,
                         'finished': bool(glob.glob(os.path.join(_DATA_DIR, pid, f'{pid}_summary_*.json'))),
                         'timestamp': e.get('timestamp')}
    taken = {int(e.get('slot', -1)) for e in rec['assigned'].values()}
    return jsonify({
        'slots': len(_DESIGN['schedule']),
        'participants_assigned': len(rec['assigned']),
        'finished': sum(p['finished'] for p in progress.values()),
        'free_slots': [s for s in range(len(_DESIGN['schedule'])) if s not in taken],
        'progress': progress,
        'released': rec.get('released', []),
    })


@app.route('/admin/release')
def admin_release():
    """Free a dropout's slot so the next arrival takes it.
    /admin/release?token=...&pid=PARTICIPANT_ID"""
    denied = _check_admin()
    if denied:
        return denied
    pid = (request.args.get('pid') or '').strip()
    with _assign_lock:
        rec = _load_assignments()
        entry = rec['assigned'].pop(pid, None)
        if entry is None:
            return jsonify({'released': False, 'error': f'no assignment for {pid!r}'}), 404
        entry = dict(entry, pid=pid, released_at=time.strftime('%Y%m%d_%H%M%S'))
        rec.setdefault('released', []).append(entry)
        _save_assignments(rec)
    print(f'[judge] released slot {entry["slot"]} of {pid}')
    return jsonify({'released': True, 'pid': pid, 'slot': entry['slot']})


def _result_files():
    out = []
    for root, _dirs, files in os.walk(_DATA_DIR):
        for name in files:
            if name == '.gitignore' or name.endswith('.tmp'):
                continue
            path = os.path.join(root, name)
            out.append((os.path.relpath(path, _HERE), path))
    return sorted(out)


@app.route('/admin/results')
def admin_results():
    """All participant data as one zip. /admin/results?token=..."""
    denied = _check_admin()
    if denied:
        return denied
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, path in _result_files():
            zf.write(path, arcname=name)
    mem.seek(0)
    return send_file(mem, mimetype='application/zip', as_attachment=True,
                     download_name=f'judge_study_results_{time.strftime("%Y%m%d_%H%M%S")}.zip')


@app.route('/admin/csv')
def admin_csv():
    """Every rating as one flat CSV row. /admin/csv?token=..."""
    denied = _check_admin()
    if denied:
        return denied
    from export_results import collect_ratings, to_csv
    resp = make_response(to_csv(collect_ratings(_DATA_DIR)))
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename=judge_study_ratings.csv'
    return resp


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='normGym LLM-judge validation study')
    ap.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5002)))
    ap.add_argument('--design_file', default=os.environ.get('DESIGN_FILE', _DEFAULT_DESIGN))
    ap.add_argument('--mode', choices=MODES, default=os.environ.get('MODE', 'agree'),
                    help="'two': own 1-7 rating, then judge revealed + agreement (wave 1); "
                         "'agree': judge shown up front, agreement only (wave 2, default)")
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    _config['design_file'] = args.design_file
    _config['mode'] = args.mode
    _DESIGN = _load_design(args.design_file)
    print(f'[judge] design {args.design_file}: {len(_DESIGN["schedule"])} slots x '
          f'{_DESIGN["items_per_participant"]} items, {_DESIGN["n_items"]} items total; '
          f'mode {args.mode}; ledger {_ledger_path()}')
    if not os.environ.get('ADMIN_TOKEN'):
        print('[judge] note: ADMIN_TOKEN not set — /admin/* endpoints disabled')
    print(f'[judge] serving on http://{args.host}:{args.port}')
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
else:
    # imported (e.g. by a WSGI server or the test client): load the default design
    _config['design_file'] = os.environ.get('DESIGN_FILE', _DEFAULT_DESIGN)
    _config['mode'] = os.environ.get('MODE', 'agree')
    _DESIGN = _load_design(_config['design_file'])
