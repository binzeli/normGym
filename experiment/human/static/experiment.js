/* normGym experiment client.
 * Reuses the paced-playback + violation-modal engine from the original
 * runner/templates/index.html, and adds:
 *   - demo mode (autoplay, no keyboard) for the Example page
 *   - per-page phase + participant id sent to the server on start
 *   - a "Continue" button that unlocks only once the round is over
 */
console.log('%cnormGym experiment client', 'color:#3a6ea5;font-weight:bold', window.EXP);

const EXP = window.EXP;
const socket = io();
let gameRunning = false;

// ---- Continue button (unlocks on game_over) ----
function unlockContinue() {
  const btn = document.getElementById('btn-continue');
  if (btn) btn.classList.remove('disabled');
}

// ---- Violation banner ----
// The norm-violation "X" is drawn server-side on exactly ONE keyframe, and that
// keyframe is the last frame of the batch carrying human_violated. _playNext
// therefore displays it FIRST and only then fires this, so the participant sees
// the red ✗ land on the board before the banner appears. Playback is frozen the
// instant the violation fires (so no later frame can displace the X), and the
// banner itself is delayed a beat purely so the two events read as sequential.
// The banner sits above the board rather than over it, so the board stays
// visible. Because it has no click-blocking backdrop, the 'violation-lock' body
// class makes the other controls inert until Continue.
const VIOLATION_BANNER_DELAY_MS = 350;
let violationBannerTimer = null;

function showViolationModal(step) {
  paused = true;
  awaitingInput = false;
  document.getElementById('violation-modal-msg').textContent =
    `You moved against the norm on step ${step}.`;
  clearTimeout(violationBannerTimer);
  violationBannerTimer = setTimeout(() => {
    violationBannerTimer = null;
    document.getElementById('violation-modal').classList.add('show');
    document.body.classList.add('violation-lock');
  }, VIOLATION_BANNER_DELAY_MS);
}
function continueAfterViolation() {
  hideViolationBanner();
  document.getElementById('btn-modal-continue').blur(); // so space/enter can't re-fire it
  // Every Nth violation, the tip takes the violation banner's place instead of
  // play resuming. Playback stays paused until the tip is dismissed.
  if (violationSteps.length > 0 && violationSteps.length % TIP_EVERY_N_VIOLATIONS === 0) {
    showTipBanner();
    return;
  }
  afterBannerDismissed();
}
function hideViolationBanner() {
  clearTimeout(violationBannerTimer);
  violationBannerTimer = null;
  document.getElementById('violation-modal').classList.remove('show');
  document.body.classList.remove('violation-lock');
}

// ---- Tip banner ----
// Re-surfaces the sidebar's playing tip after every TIP_EVERY_N_VIOLATIONS
// violations in a round (3rd, 6th, 9th, …), since a participant racking up
// violations is the one most likely to have stopped reading the sidebar.
const TIP_EVERY_N_VIOLATIONS = 3;

function showTipBanner() {
  paused = true;
  awaitingInput = false;
  document.getElementById('tip-modal').classList.add('show');
  document.body.classList.add('violation-lock');
}
function hideTipBanner() {
  document.getElementById('tip-modal').classList.remove('show');
  document.body.classList.remove('violation-lock');
}
function dismissTip() {
  hideTipBanner();
  document.getElementById('btn-tip-continue').blur();
  afterBannerDismissed();
}

// ---- Mid-round check-in (interactive stage only) ----
// Fires every EXP.midGuessEvery steps (30, 60, 90, …), at the moment the board
// *displays* that step, so the question lines up with what the participant can
// see. Play is paused while it is open; the answer goes to the server on submit.
let midGuessNextStep = 0;     // step at which the next check-in is due
let midGuessPending = false;  // due, but waiting on another banner to close

function maybeShowMidGuess(step) {
  if (EXP.phase !== 'interactive' || !EXP.midGuessEvery) return;
  if (midGuessPending || step < midGuessNextStep) return;
  // Advance to the next multiple strictly past this step, so a step that arrives
  // late (or a skipped multiple) still schedules the following one correctly.
  midGuessNextStep = (Math.floor(step / EXP.midGuessEvery) + 1) * EXP.midGuessEvery;
  midGuessPending = true;
  paused = true;
  awaitingInput = false;
  document.getElementById('guess-step').textContent = step;
  // A violation/tip banner for this same step gets read first; that banner's
  // dismissal calls afterBannerDismissed(), which then opens this. Note the
  // pending-timer check: a violation on THIS step has already been fired by
  // _playNext but its banner is still inside its reveal delay, so it is not
  // 'open' yet — without this the check-in would open on top of it.
  if (!modalIsOpen() && violationBannerTimer === null) showMidGuess();
}
function showMidGuess() {
  const modal = document.getElementById('guess-modal');
  if (!modal) { midGuessPending = false; resumePlayback(); return; }
  paused = true;
  awaitingInput = false;
  modal.classList.add('show');
  document.body.classList.add('violation-lock');
  hideMidGuessError();
  setTimeout(() => document.getElementById('mid-guess-text').focus(), 50);
}
// The check-in is required. The button stays clickable and refuses an empty
// answer, rather than sitting disabled with no explanation.
function hideMidGuessError() {
  const err = document.getElementById('mid-guess-error');
  if (err) err.style.display = 'none';
}
function submitMidGuess() {
  const text = document.getElementById('mid-guess-text');
  const conf = document.getElementById('mid-guess-conf');
  if (text.value.trim() === '') {
    const err = document.getElementById('mid-guess-error');
    if (err) err.style.display = 'block';
    text.focus();
    return;
  }
  hideMidGuessError();
  socket.emit('mid_guess', {
    guess: text.value.trim(),
    confidence: Number(conf.value),
    step: frameSteps.length ? frameSteps[frameSteps.length - 1] : 0,
  });
  midGuessPending = false;
  document.getElementById('guess-modal').classList.remove('show');
  document.body.classList.remove('violation-lock');
  document.getElementById('btn-guess-submit').blur();
  clearMidGuessFields();  // each check-in is answered fresh, not edited
  resumePlayback();
}
function clearMidGuessFields() {
  const text = document.getElementById('mid-guess-text');
  if (!text) return;
  text.value = '';
  document.getElementById('mid-guess-conf').value = 0;
  document.getElementById('mid-guess-conf-val').textContent = '0';
  hideMidGuessError();
}
function resetMidGuess() {
  midGuessNextStep = EXP.midGuessEvery || 0;
  midGuessPending = false;
  const modal = document.getElementById('guess-modal');
  if (!modal) return;
  modal.classList.remove('show');
  clearMidGuessFields();
}
function modalIsOpen() {
  const guess = document.getElementById('guess-modal');
  return document.getElementById('violation-modal').classList.contains('show')
      || document.getElementById('tip-modal').classList.contains('show')
      || (guess !== null && guess.classList.contains('show'));
}

// Interruptions queue rather than stack: if the check-in comes due while a
// violation or tip banner is up, it waits for that banner to be dismissed.
function afterBannerDismissed() {
  if (midGuessPending) { showMidGuess(); return; }
  resumePlayback();
}

// ---- Violation log (right panel) + board badge ----
const violationSteps = [];
function recordViolation(step) {
  violationSteps.push(step);
  document.getElementById('violation-total').textContent = violationSteps.length;
  setViolationBadge(violationSteps.length);
  document.getElementById('violation-empty').style.display = 'none';
  const log = document.getElementById('violation-log');
  const item = document.createElement('div');
  item.className = 'violation-item';
  item.innerHTML = `<span class="v-idx">#${violationSteps.length}</span><span>step ${step}</span>`;
  log.appendChild(item);
  log.scrollTop = log.scrollHeight;
}
function clearViolations() {
  violationSteps.length = 0;
  document.getElementById('violation-total').textContent = '0';
  setViolationBadge(0);
  document.getElementById('violation-log').innerHTML =
    '<div class="violation-empty" id="violation-empty">No violations yet.</div>';
}

// ---- Status helper ----
function setStatus(msg, cls = '') {
  const el = document.getElementById('status-bar');
  el.innerHTML = msg;
  el.className = cls;
}

// ---- Violation badge (play mode only) — norm violations so far this round ----
function setViolationBadge(count) {
  const badge = document.getElementById('violation-badge');
  if (!badge) return;
  document.getElementById('violation-badge-value').textContent = count;
  badge.classList.toggle('hit', count > 0);
}

// ---- Frame history + timeline scrubber ----
const frameHistory = [];
const frameSteps = [];
let maxStep = 0;
let currentIdx = 0;
let liveMode = true;
let reviewing = false;

// Open the replay scrubber: reveal the timeline and pause live play / input.
function openReplay() {
  if (frameHistory.length === 0) return;
  reviewing = true;
  liveMode = false;
  document.getElementById('replay-bar').style.display = 'none';
  document.getElementById('timeline-bar').style.display = 'flex';
  refreshControls();
}
// Close the replay scrubber: jump back to the latest frame and resume play.
function closeReplay() {
  reviewing = false;
  document.getElementById('timeline-bar').style.display = 'none';
  document.getElementById('replay-bar').style.display = 'flex';
  goLive();
}

function showFrameAt(idx) {
  if (idx < 0 || idx >= frameHistory.length) return;
  currentIdx = idx;
  document.getElementById('timeline').value = idx;
  const img = document.getElementById('game-frame');
  const ph = document.getElementById('frame-placeholder');
  ph.style.display = 'none';
  img.style.display = 'block';
  img.src = 'data:image/png;base64,' + frameHistory[idx];
}
function updateTimelineLabel() {
  const step = frameHistory.length ? frameSteps[currentIdx] : 0;
  document.getElementById('timeline-label').textContent = `step ${step} / ${maxStep}`;
}
function refreshControls() {
  document.getElementById('btn-live').classList.toggle('live', liveMode);
  const hasFrames = frameHistory.length > 0;
  document.getElementById('btn-prev').disabled = !hasFrames || currentIdx <= 0;
  document.getElementById('btn-next').disabled = !hasFrames || currentIdx >= frameHistory.length - 1;
}
function pushFrame(b64, step) {
  frameHistory.push(b64);
  frameSteps.push(step);
  if (step > maxStep) maxStep = step;
  const slider = document.getElementById('timeline');
  slider.max = frameHistory.length - 1;
  slider.disabled = false;
  if (liveMode) showFrameAt(frameHistory.length - 1);
  updateTimelineLabel();
  refreshControls();
}
function stepBack() {
  if (currentIdx <= 0) return;
  liveMode = false;
  showFrameAt(currentIdx - 1);
  updateTimelineLabel();
  refreshControls();
}
function stepForward() {
  if (currentIdx >= frameHistory.length - 1) return;
  showFrameAt(currentIdx + 1);
  liveMode = (currentIdx === frameHistory.length - 1);
  updateTimelineLabel();
  refreshControls();
}
function goLive() {
  if (frameHistory.length === 0) return;
  liveMode = true;
  showFrameAt(frameHistory.length - 1);
  updateTimelineLabel();
  refreshControls();
}
document.getElementById('timeline').addEventListener('input', () => {
  const idx = Number(document.getElementById('timeline').value);
  showFrameAt(idx);
  liveMode = (idx === frameHistory.length - 1);
  updateTimelineLabel();
  refreshControls();
});

// ---- Frame animation queue ----
const frameQueue = [];
let animating = false;
let paused = false;
let awaitingInput = false;
const MS_PER_FRAME = 100;

function _playNext() {
  if (paused || frameQueue.length === 0) { animating = false; return; }
  // A violation batch carries the X-mark keyframe as its final frame. It is
  // played like any other batch, so the X is on screen before onViolationFn
  // fires below; that call pauses playback, leaving the X displayed under the
  // banner (queued frames can't advance while paused, so nothing displaces it).
  animating = true;
  const { framesB64, step, applyStateFn, onViolationFn } = frameQueue.shift();
  if (framesB64.length === 0) {
    if (onViolationFn) onViolationFn();
    if (applyStateFn) applyStateFn();
    if (paused) { animating = false; return; }
    _playNext();
    return;
  }
  let idx = 0;
  function _step() {
    if (paused) { animating = false; return; }
    pushFrame(framesB64[idx], step);
    if (idx === framesB64.length - 1) {
      if (onViolationFn) onViolationFn();
      if (applyStateFn) applyStateFn();
      if (paused) { animating = false; return; }
      setTimeout(_playNext, MS_PER_FRAME);
      return;
    }
    idx++;
    setTimeout(_step, MS_PER_FRAME);
  }
  _step();
}
function resumePlayback() {
  paused = false;
  if (!animating) _playNext();
}

// ---- Socket events ----
socket.on('connect', () => console.log('socket connected'));
socket.on('disconnect', () => { setStatus('Disconnected from server.', ''); gameRunning = false; });

socket.on('game_state', (data) => {
  const framesB64 = data.frames_b64 || [];

  function applyState() {
    // Keep the "This is you" sprite in sync with the current avatar's colour
    // (the human is replaced by a new, randomly-coloured character each round).
    if (data.human_class) {
      const youSprite = document.getElementById('you-sprite');
      if (youSprite) youSprite.src = youSprite.src.replace(/[^/]+\.png$/, data.human_class + '_you.png');
    }

    // Prefer the natural-language goal text (sourced from agent_goals.json); for
    // a free-choice goal this describes the choice without revealing a cell.
    if (data.human_goal) {
      const tb = document.getElementById('target-banner');
      document.getElementById('target-banner-text').textContent = data.human_goal;
      tb.classList.add('show');
    } else if (data.human_target && data.human_target.length === 2) {
      const tb = document.getElementById('target-banner');
      document.getElementById('target-banner-text').textContent =
        `row ${data.human_target[0]}, col ${data.human_target[1]}`;
      tb.classList.add('show');
    } else {
      document.getElementById('target-banner').classList.remove('show');
    }

    if (data.terminated || data.truncated) {
      setStatus('Episode ended.', 'gameover');
      gameRunning = false; awaitingInput = false;
    } else if (data.waiting_for_input) {
      setStatus('Your turn — press an arrow key or <strong>Space</strong>', 'waiting');
      awaitingInput = true;
    } else if (EXP.mode === 'demo') {
      setStatus('Watching the characters follow the norm…', 'thinking');
      awaitingInput = false;
    } else {
      setStatus('Characters moving…', 'thinking');
      awaitingInput = false;
    }

    // Last, so it can override awaitingInput/paused set just above. Only for a
    // live round — a finished episode should go to its summary, not a check-in.
    if (!(data.terminated || data.truncated)) maybeShowMidGuess(data.step);
  }

  const violatedStep = data.step;
  // In demo mode there is no human, so human_violated never fires — good, no modal.
  const onViolation = data.human_violated
    ? () => { recordViolation(violatedStep); showViolationModal(violatedStep); }
    : null;

  frameQueue.push({ framesB64, step: data.step, applyStateFn: applyState, onViolationFn: onViolation });
  if (!animating && !paused) _playNext();
});

socket.on('game_over', (data) => {
  const overlay = document.getElementById('finished-overlay');
  const title = document.getElementById('finished-title');
  const msg = document.getElementById('finished-msg');
  if (EXP.mode === 'demo') {
    title.textContent = 'Example finished';
    msg.textContent = 'That is how the game looks. Next you can practice the controls.';
  } else {
    const titles = {
      practice: 'Practice complete',
      interactive: 'Learning round complete',
      test: 'Test round complete',
    };
    title.textContent = titles[EXP.phase] || 'Round complete';
    // The interactive round is exploratory, so don't show the summary for it.
    // Reward is never surfaced to the participant — only the counts are.
    msg.textContent = (EXP.phase === 'interactive')
      ? ''
      : `Steps: ${data.total_steps}  •  Collisions: ${data.total_collisions}  •  ` +
        `Norm violations: ${data.norm_violations}`;
  }
  // Wait until the queued frames finish animating before showing the overlay.
  const showWhenIdle = () => {
    if (animating || frameQueue.length > 0) { setTimeout(showWhenIdle, 150); return; }
    overlay.classList.add('show');
    setStatus(EXP.mode === 'demo' ? 'Example finished.' : 'Round finished.', 'gameover');
    unlockContinue();
  };
  showWhenIdle();
  gameRunning = false;
});

socket.on('reset_ack', () => {});

// ---- Start / replay ----
function clearTimeline() {
  frameQueue.length = 0; animating = false; paused = false;
  hideViolationBanner();
  hideTipBanner();
  resetMidGuess();
  document.getElementById('finished-overlay').classList.remove('show');
  frameHistory.length = 0; frameSteps.length = 0;
  maxStep = 0; currentIdx = 0; liveMode = true;
  const slider = document.getElementById('timeline');
  slider.value = 0; slider.max = 0; slider.disabled = true;
  reviewing = false;
  document.getElementById('timeline-bar').style.display = 'none';
  document.getElementById('replay-bar').style.display = 'flex';
  updateTimelineLabel(); refreshControls(); clearViolations();
  document.getElementById('game-frame').style.display = 'none';
  document.getElementById('frame-placeholder').style.display = 'flex';
}

function startRound() {
  clearTimeline();
  // The task panel only makes sense once there is a goal to show.
  const taskSection = document.getElementById('task-section');
  if (taskSection) taskSection.style.display = '';
  socket.emit('start_session', {
    participant_id: EXP.participantId,
    phase: EXP.phase,
    norm_type: EXP.normType,
    world_type: EXP.worldType,
    block: EXP.block,
    episode: EXP.episode,
  });
  gameRunning = true;
  setStatus(EXP.mode === 'demo' ? 'Loading example…' : 'Starting…', 'thinking');
}

if (EXP.mode === 'play') {
  const startBtn = document.getElementById('btn-start');
  startBtn.addEventListener('click', () => {
    // Test rounds are a single clean run — once started, no restart is allowed.
    if (EXP.phase === 'test') {
      startBtn.style.display = 'none';
    } else {
      startBtn.textContent = 'Restart';
    }
    startRound();
  });
} else {
  // demo: autoplay on load, allow replay
  document.getElementById('btn-replay').addEventListener('click', startRound);
  window.addEventListener('load', () => setTimeout(startRound, 400));
}

// ---- Keyboard (play mode only) ----
const KEY_MAP = { 'ArrowUp': 'up', 'ArrowDown': 'down', 'ArrowLeft': 'left', 'ArrowRight': 'right', ' ': 'wait' };
// Space is the "wait" action AND an ordinary character in the check-in's
// textarea; arrows likewise move the caret and the confidence slider. So bail
// out before preventDefault() whenever the key belongs to a form field or a
// modal is up — otherwise typing a space in the check-in does nothing.
function keyBelongsToField(target) {
  if (!target) return false;
  const tag = target.tagName;
  return tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT' || target.isContentEditable;
}
if (EXP.mode === 'play') {
  document.addEventListener('keydown', (e) => {
    const action = KEY_MAP[e.key];
    if (!action) return;
    if (keyBelongsToField(e.target)) return;
    if (modalIsOpen()) return;
    e.preventDefault();
    if (!gameRunning) return;
    if (reviewing) return;
    if (!awaitingInput) return;
    awaitingInput = false;
    socket.emit('player_action', { action });
  });
}
