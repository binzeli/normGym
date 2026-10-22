"""Parse norm_beliefs_500.txt into a list of item dicts (and optionally dump
them as JSON).

The text file is a human-readable dump (see its header) of 500 judged model
beliefs: for each entry the model's stated belief about the hidden norm, the
ground-truth norm description, and the LLM judge's 7-point Likert rating plus
its one-line reason. Entries are separated by a line of '=' characters.

Usage (standalone):
    python experiment/human_judge/parse_beliefs.py                 # prints a summary
    python experiment/human_judge/parse_beliefs.py --out items.json
"""
import argparse
import json
import os
import re
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TXT = os.path.join(_HERE, 'norm_beliefs_500.txt')
# Plain-language versions of the 50 norms (no coordinates / quadrant geometry /
# exact queue paths): what the PARTICIPANT sees as the true rule. The judge saw
# the full text, which stays in 'ground_truth'.
NORM_SHORT_PATH = os.path.join(_HERE, 'norm_short.json')

_SEP = re.compile(r'^={20,}\s*$', re.M)
_RE_ENTRY = re.compile(r'^ENTRY (\d+)\s*$', re.M)
_RE_MODEL = re.compile(r'^MODEL: (.+?)\s*$', re.M)
_RE_NORM = re.compile(r'^NORM: (\S+)\s+\(family: ([^)]+)\)\s*$', re.M)
_RE_STEP = re.compile(r'^INTERACTIVE STEP: (\d+)\s+\(subject ([^)]+)\)\s*$', re.M)
_RE_CONF = re.compile(r'^MODEL CONFIDENCE: (\S+)\s*$', re.M)
_RE_BELIEF = re.compile(r'^MODEL BELIEF:\n(.*?)\n\s*\nGROUND-TRUTH NORM \(([^)]+)\):\n(.*?)\n\s*\nJUDGE LIKERT: (\d)/7\s+\(score ([0-9.]+)\)\s*\nJUDGE REASON: (.*?)\s*$', re.S | re.M)


def parse_entry(block: str) -> dict:
    m_entry = _RE_ENTRY.search(block)
    m_model = _RE_MODEL.search(block)
    m_norm = _RE_NORM.search(block)
    m_step = _RE_STEP.search(block)
    m_conf = _RE_CONF.search(block)
    m_body = _RE_BELIEF.search(block)
    if not all([m_entry, m_model, m_norm, m_step, m_conf, m_body]):
        raise ValueError('malformed entry:\n' + block[:400])
    belief = m_body.group(1).strip()
    return {
        'entry': int(m_entry.group(1)),
        'model': m_model.group(1).strip(),
        'norm': m_norm.group(1),
        'family': m_norm.group(2).strip(),
        'step': int(m_step.group(1)),
        'subject': m_step.group(2).strip(),
        'confidence': float(m_conf.group(1)),
        'belief': belief,
        'belief_is_none': belief.lower() == 'none',
        'ground_truth_norm': m_body.group(2).strip(),
        'ground_truth': m_body.group(3).strip(),
        'judge_likert': int(m_body.group(4)),
        'judge_score': float(m_body.group(5)),
        'judge_reason': ' '.join(m_body.group(6).split()),
    }


def load_norm_short(path: str = NORM_SHORT_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith('_')}


def parse_file(path: str = DEFAULT_TXT, short_path: str = NORM_SHORT_PATH) -> list:
    with open(path, encoding='utf-8') as f:
        text = f.read()
    blocks = [b for b in _SEP.split(text) if _RE_ENTRY.search(b)]
    items = [parse_entry(b) for b in blocks]
    ids = [it['entry'] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate ENTRY numbers in ' + path)
    short = load_norm_short(short_path)
    missing = sorted({it['norm'] for it in items} - set(short))
    if missing:
        raise ValueError(f'{short_path} has no short description for: {missing}')
    for it in items:
        it['ground_truth_short'] = short[it['norm']]
    return items


def summarize(items: list) -> str:
    lines = [f'{len(items)} entries']
    lines.append('per judge Likert: ' + ', '.join(
        f'{k}:{v}' for k, v in sorted(Counter(i['judge_likert'] for i in items).items())))
    lines.append('per family: ' + ', '.join(
        f'{k}:{v}' for k, v in sorted(Counter(i['family'] for i in items).items())))
    lines.append('per model: ' + ', '.join(
        f'{k}:{v}' for k, v in sorted(Counter(i['model'] for i in items).items())))
    lines.append(f"belief == 'none': {sum(i['belief_is_none'] for i in items)}")
    gw = sorted(len(i['ground_truth'].split()) for i in items)
    sw = sorted(len(i['ground_truth_short'].split()) for i in items)
    lines.append(f'true-norm words: full median {gw[len(gw) // 2]} max {gw[-1]}; short median {sw[len(sw) // 2]} max {sw[-1]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--txt', default=DEFAULT_TXT)
    ap.add_argument('--out', default=None, help='write the parsed items as JSON here')
    args = ap.parse_args()
    items = parse_file(args.txt)
    print(summarize(items))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=1, ensure_ascii=False)
        print(f'wrote {len(items)} items -> {args.out}')
