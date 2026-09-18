"""
Hangman Solver — Streamlit frontend by Siddhant Kumawat

Modes:
  1. Random word from test_words.txt — BiLSTM solver plays it out
  2. Custom word input — BiLSTM solver plays it out

Guesses animate in-place after the full solve finishes.
"""

from __future__ import annotations

import string
import time

import streamlit as st

from .game import MAX_TRIES
from .solver import GameResult, HangmanSolver, load_solver

st.set_page_config(
    page_title="Hangman Solver — Siddhant Kumawat",
    page_icon="H",
    layout="centered",
    initial_sidebar_state="collapsed",
)

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,600;0,700;1,400&family=Fraunces:opsz,wght@9..144,650;9..144,700&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  direction: ltr !important;
}

.stApp {
  background:
    radial-gradient(ellipse 90% 70% at 8% -10%, #355c48 0%, transparent 52%),
    radial-gradient(ellipse 80% 60% at 110% 110%, #1c332c 0%, transparent 48%),
    linear-gradient(165deg, #0c1612 0%, #12211b 42%, #0a120f 100%);
  color: #e8f0ea;
  direction: ltr !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

.block-container {
  padding-top: 1.4rem;
  padding-bottom: 3rem;
  max-width: 820px;
  direction: ltr !important;
}

.hero { text-align: center; margin-bottom: 1.15rem; }
.kicker {
  display: inline-block;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  font-size: 0.72rem;
  font-weight: 700;
  color: #9ee0b0;
  margin-bottom: 0.45rem;
}
.brand {
  font-family: 'Fraunces', Georgia, serif;
  font-size: clamp(2.35rem, 7vw, 3.5rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #f4efe6;
  margin: 0;
  line-height: 1.05;
}
.tagline {
  color: #a8c0b2;
  font-size: 1.02rem;
  margin: 0.55rem 0 0.35rem 0;
}
.credit { color: #7f9b8c; font-size: 0.88rem; margin: 0; }

.stat-row {
  display: flex;
  justify-content: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  margin: 1rem 0 1.2rem 0;
}
.stat {
  background: rgba(24, 46, 38, 0.72);
  border: 1px solid rgba(158, 224, 176, 0.18);
  border-radius: 999px;
  padding: 0.32rem 0.8rem;
  color: #cfe0d6;
  font-size: 0.8rem;
  font-weight: 600;
}
.stat em { color: #9ee0b0; font-style: normal; }

.board {
  background: linear-gradient(180deg, rgba(28, 52, 44, 0.78), rgba(16, 32, 26, 0.88));
  border: 1px solid rgba(201, 184, 150, 0.18);
  border-radius: 18px;
  padding: 1.1rem 1.15rem 1.2rem;
  box-shadow: 0 18px 50px rgba(0,0,0,0.28);
  margin-bottom: 1rem;
}

.stage {
  display: grid;
  grid-template-columns: 180px 1fr;
  gap: 1.1rem;
  align-items: center;
}

@media (max-width: 640px) {
  .stage { grid-template-columns: 1fr; }
}

.gallows-wrap { display: flex; justify-content: center; }
.gallows-wrap svg {
  width: min(168px, 46vw);
  height: auto;
  filter: drop-shadow(0 10px 20px rgba(0,0,0,0.35));
}

.secret-line {
  text-align: center;
  color: #8fa897;
  font-size: 0.84rem;
  margin: 0 0 0.65rem 0;
  direction: ltr !important;
  unicode-bidi: isolate;
}
.secret-line em { color: #f2ebe0; font-style: normal; font-weight: 700; }

.meta {
  display: flex;
  justify-content: center;
  gap: 0.9rem;
  flex-wrap: wrap;
  color: #8fa897;
  font-size: 0.82rem;
  margin-bottom: 0.7rem;
}
.meta strong { color: #d5e6db; font-weight: 700; }

.tiles {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.32rem;
  margin: 0.15rem 0 0.85rem 0;
  direction: ltr !important;
  unicode-bidi: isolate;
}
.tile {
  min-width: 1.7rem;
  height: 2.15rem;
  padding: 0 0.28rem;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.2rem;
  font-weight: 700;
  text-transform: uppercase;
}
.tile.filled {
  background: #2f9e5a;
  color: #f4fff7;
  box-shadow: 0 6px 14px rgba(47, 158, 90, 0.28);
}
.tile.empty {
  background: rgba(15, 28, 24, 0.7);
  color: #c9b896;
  border: 1px solid rgba(201, 184, 150, 0.28);
}

.kb {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.28rem;
  max-width: 420px;
  margin: 0 auto;
}
.key {
  width: 1.7rem;
  height: 1.7rem;
  border-radius: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  background: rgba(15, 28, 24, 0.55);
  color: #8fa897;
  border: 1px solid rgba(143, 168, 151, 0.22);
}
.key.hit { background: #2f9e5a; color: #eafff1; border-color: transparent; }
.key.miss { background: #d6453a; color: #fff0ee; border-color: transparent; }

.banner {
  text-align: center;
  padding: 0.85rem 1rem;
  margin: 0.2rem 0 0.9rem 0;
  border-radius: 12px;
  font-weight: 700;
}
.banner.won {
  background: rgba(62, 140, 98, 0.25);
  color: #9ee0b0;
  border: 1px solid rgba(110, 190, 130, 0.35);
}
.banner.lost {
  background: rgba(160, 70, 55, 0.28);
  color: #f0b4a8;
  border: 1px solid rgba(200, 100, 80, 0.35);
}

.guess-log { direction: ltr !important; unicode-bidi: isolate; text-align: left; }
.guess-row {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  margin: 0.32rem 0;
  font-size: 0.93rem;
}
.badge {
  display: inline-block;
  min-width: 4.4rem;
  text-align: center;
  padding: 0.16rem 0.5rem;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: 0.04em;
  font-size: 0.72rem;
}
.badge-hit { background: #2f9e5a; color: #eafff1; }
.badge-miss { background: #d6453a; color: #fff0ee; }
.guess-meta { color: #a8bdb0; }

.replay-caption {
  text-align: center;
  color: #8fa897;
  font-size: 0.86rem;
  margin: 0.15rem 0 0.85rem 0;
}
.replay-caption strong { color: #d5e6db; font-weight: 700; }
div[data-testid="stSlider"] { padding-top: 0.15rem; }

.preview-box {
  margin: 0.35rem 0 0.7rem 0;
  padding: 0.7rem 0.9rem;
  background: rgba(30, 52, 44, 0.75);
  border: 1px solid rgba(143, 168, 151, 0.35);
  border-radius: 10px;
  direction: ltr !important;
  unicode-bidi: isolate;
  text-align: left;
  font-family: 'DM Sans', monospace, sans-serif;
  letter-spacing: 0.12em;
  color: #e8f0ea;
}
.preview-label { color: #8fa897; letter-spacing: normal; font-size: 0.78rem; display: block; margin-bottom: 0.25rem; }

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
  direction: ltr !important;
  unicode-bidi: normal !important;
  text-align: left !important;
  writing-mode: horizontal-tb !important;
  background: rgba(30, 52, 44, 0.9) !important;
  color: #e8f0ea !important;
  border: 1px solid rgba(143, 168, 151, 0.4) !important;
  border-radius: 10px !important;
  font-family: Consolas, 'Courier New', monospace !important;
  letter-spacing: 0.08em !important;
}

.stButton > button[kind="primary"] {
  background: #3d7a5a !important;
  border: none !important;
  color: #f2ebe0 !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
  height: 2.7rem !important;
}
.stButton > button[kind="primary"]:hover { background: #4a916c !important; }
.stButton > button[kind="secondary"] {
  background: rgba(30, 52, 44, 0.85) !important;
  border: 1px solid rgba(143, 168, 151, 0.4) !important;
  color: #e8f0ea !important;
  border-radius: 10px !important;
}

.victory {
  text-align: center;
  margin: 0.15rem 0 0.75rem 0;
  animation: victory-pop 0.85s cubic-bezier(0.22, 1.2, 0.36, 1) both;
}
.victory svg {
  width: 92px;
  height: 92px;
  filter: drop-shadow(0 10px 28px rgba(62, 180, 110, 0.35));
  animation: victory-bounce 1.4s ease-in-out infinite;
}
.victory-caption {
  color: #9ee0b0;
  font-weight: 700;
  font-size: 1.08rem;
  margin-top: 0.3rem;
  letter-spacing: 0.04em;
}
@keyframes victory-pop {
  0% { opacity: 0; transform: scale(0.4) rotate(-12deg); }
  60% { opacity: 1; transform: scale(1.12) rotate(4deg); }
  100% { opacity: 1; transform: scale(1) rotate(0deg); }
}
@keyframes victory-bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

.empty-hint {
  text-align: center;
  color: #8fa897;
  margin-top: 0.4rem;
  font-size: 0.92rem;
}
</style>
"""

GALLOWS_SVG = [
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/><path d="M120 94 V150" stroke="#e8dcc8" stroke-width="5" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/><path d="M120 94 V150 M120 110 L90 135" stroke="#e8dcc8" stroke-width="5" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/><path d="M120 94 V150 M120 110 L90 135 M120 110 L150 135" stroke="#e8dcc8" stroke-width="5" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/><path d="M120 94 V150 M120 110 L90 135 M120 110 L150 135 M120 150 L95 190" stroke="#e8dcc8" stroke-width="5" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#d48474" stroke-width="5" fill="none"/><path d="M120 94 V150 M120 110 L90 135 M120 110 L150 135 M120 150 L95 190 M120 150 L145 190" stroke="#d48474" stroke-width="5" stroke-linecap="round"/></svg>""",
]

VICTORY_SMILE = """
<div class="victory">
  <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" aria-label="Victory smile">
    <circle cx="60" cy="60" r="54" fill="#2f9e5a"/>
    <circle cx="42" cy="48" r="7" fill="#0f1c18"/>
    <circle cx="78" cy="48" r="7" fill="#0f1c18"/>
    <path d="M36 70 Q60 96 84 70" stroke="#0f1c18" stroke-width="7" fill="none" stroke-linecap="round"/>
  </svg>
  <div class="victory-caption">Victory!</div>
</div>
"""

STEP_DELAY_SEC = 0.28
BIDI_MARKS = str.maketrans("", "", "\u200e\u200f\u202a\u202b\u202c\u202d\u202e")


@st.cache_resource(show_spinner=False)
def get_solver() -> HangmanSolver:
    return load_solver()


def normalize_word(raw: str) -> str:
    """Strip bidi control marks, lowercase, and keep letters only for solving."""
    word = (raw or "").translate(BIDI_MARKS).strip().lower()
    return "".join(ch for ch in word if ch.isalpha())


def _pattern_and_wrong(result: GameResult, step_idx: int) -> tuple[str, int, int]:
    if step_idx <= 0:
        return "_" * len(result.word), MAX_TRIES, 0
    step = result.steps[step_idx - 1]
    return step.pattern, step.tries_left, MAX_TRIES - step.tries_left


def _tiles_html(pattern: str) -> str:
    tiles = []
    for ch in pattern:
        cls = "filled" if ch != "_" else "empty"
        label = ch if ch != "_" else ""
        tiles.append(f'<span class="tile {cls}">{label}</span>')
    return f'<div class="tiles">{"".join(tiles)}</div>'


def _keyboard_html(result: GameResult, step_idx: int) -> str:
    hits: set[str] = set()
    misses: set[str] = set()
    for step in result.steps[:step_idx]:
        if step.hit:
            hits.add(step.letter)
        else:
            misses.add(step.letter)
    keys = []
    for ch in string.ascii_lowercase:
        cls = "key"
        if ch in hits:
            cls += " hit"
        elif ch in misses:
            cls += " miss"
        keys.append(f'<span class="{cls}">{ch}</span>')
    return f'<div class="kb">{"".join(keys)}</div>'


def render_board(result: GameResult, step_idx: int) -> None:
    pattern, tries_left, wrong = _pattern_and_wrong(result, step_idx)
    svg = GALLOWS_SVG[min(wrong, len(GALLOWS_SVG) - 1)]
    st.markdown(
        f"""
        <div class="board">
          <div class="secret-line">Secret: <em>{result.word}</em></div>
          <div class="stage">
            <div class="gallows-wrap">{svg}</div>
            <div>
              <div class="meta">
                <span>Tries left <strong>{tries_left}</strong> / {MAX_TRIES}</span>
                <span>Guess <strong>{step_idx}</strong> / {result.n_guesses}</span>
                <span>Length <strong>{len(result.word)}</strong></span>
              </div>
              {_tiles_html(pattern)}
              {_keyboard_html(result, step_idx)}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_victory() -> None:
    st.markdown(VICTORY_SMILE, unsafe_allow_html=True)


def render_result_banner(result: GameResult) -> None:
    if result.result == "WIN":
        render_victory()
        st.markdown(
            f'<div class="banner won">Solved — {result.n_guesses} guesses, '
            f"{result.tries_used} wrong</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner lost">Failed — used all {MAX_TRIES} wrong tries '
            f"({result.n_guesses} guesses)</div>",
            unsafe_allow_html=True,
        )


def render_guess_log(result: GameResult, up_to: int) -> None:
    st.markdown("#### Guess log")
    if up_to <= 0:
        return

    lines = ['<div class="guess-log">']
    for s in result.steps[:up_to]:
        if s.hit:
            badge = '<span class="badge badge-hit">HIT</span>'
        else:
            badge = '<span class="badge badge-miss">MISS</span>'
        pat = " ".join(s.pattern)
        lines.append(
            f'<div class="guess-row">'
            f"<strong>{s.n_guess:02d}.</strong> "
            f"<code>'{s.letter}'</code> {badge} "
            f'<span class="guess-meta">{pat} · tries left {s.tries_left}</span>'
            f"</div>"
        )
    lines.append("</div>")
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def start_solve(word: str) -> None:
    solver: HangmanSolver = st.session_state.solver
    with st.spinner("Solving…"):
        result = solver.play(word)
    st.session_state.result = result
    st.session_state.step_idx = 0
    st.session_state.animating = True


def _clamp_step(value: int, max_step: int) -> int:
    return max(0, min(int(max_step), int(value)))


def _nudge_replay(delta: int, max_step: int) -> None:
    current = int(st.session_state.get("step_idx", 0) or 0)
    st.session_state.step_idx = _clamp_step(current + delta, max_step)


def _jump_replay(target: int, max_step: int) -> None:
    st.session_state.step_idx = _clamp_step(target, max_step)


def _step_caption(result: GameResult, step_idx: int) -> str:
    total = int(result.n_guesses)
    if step_idx <= 0:
        return f'Step <strong>0 / {total}</strong> · before any guesses'
    step = result.steps[step_idx - 1]
    kind = "HIT" if step.hit else "MISS"
    return (
        f"Step <strong>{step_idx} / {total}</strong> · "
        f"guessed <strong>'{step.letter}'</strong> · {kind}"
    )


def _render_replay_frame(result: GameResult, step_idx: int, n: int) -> None:
    render_board(result, step_idx)
    if n > 0 and step_idx == n:
        render_result_banner(result)
    render_guess_log(result, up_to=step_idx)


def render_playback(result: GameResult) -> None:
    n = int(result.n_guesses)
    max_step = max(n, 0)

    # Clamp before the keyed slider exists this run. Do not write the same
    # key after st.slider, or Streamlit 1.34+ will snap the thumb back.
    if "step_idx" not in st.session_state:
        st.session_state.step_idx = 0
    elif int(st.session_state.step_idx) != _clamp_step(st.session_state.step_idx, max_step):
        st.session_state.step_idx = _clamp_step(st.session_state.step_idx, max_step)

    st.divider()
    if st.session_state.animating:
        slot = st.empty()
        for idx in range(n + 1):
            with slot.container():
                _render_replay_frame(result, idx, n)
            if idx < n:
                time.sleep(STEP_DELAY_SEC)
        st.session_state.step_idx = n
        st.session_state.animating = False
        st.rerun()

    step_now = _clamp_step(st.session_state.step_idx, max_step)
    first, prev, nxt, last = st.columns(4)
    first.button(
        "⏮ First",
        use_container_width=True,
        disabled=step_now <= 0,
        on_click=_jump_replay,
        args=(0, max_step),
        key="replay_first",
    )
    prev.button(
        "◀ Prev",
        use_container_width=True,
        disabled=step_now <= 0,
        on_click=_nudge_replay,
        args=(-1, max_step),
        key="replay_prev",
    )
    nxt.button(
        "Next ▶",
        use_container_width=True,
        disabled=step_now >= max_step,
        on_click=_nudge_replay,
        args=(1, max_step),
        key="replay_next",
    )
    last.button(
        "Last ⏭",
        use_container_width=True,
        disabled=step_now >= max_step,
        on_click=_jump_replay,
        args=(max_step, max_step),
        key="replay_last",
    )

    if n > 0:
        st.slider(
            "Replay step",
            min_value=0,
            max_value=n,
            step=1,
            key="step_idx",
            help="Drag to a guess, then release. The board, gallows, and log follow this step.",
        )
    else:
        st.caption("No guesses to replay.")

    step_idx = _clamp_step(st.session_state.step_idx, max_step)
    st.markdown(
        f'<p class="replay-caption">{_step_caption(result, step_idx)}</p>',
        unsafe_allow_html=True,
    )
    # Keyed container remounts custom HTML so the board cannot stick on an old step.
    with st.container(key=f"replay_frame_{step_idx}"):
        _render_replay_frame(result, step_idx, n)


def main() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <div class="kicker">BiLSTM solver</div>
          <p class="brand">Hangman</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "solver" not in st.session_state:
        with st.spinner("Loading models and dictionary…"):
            try:
                st.session_state.solver = get_solver()
            except Exception as exc:
                st.error(f"Failed to load solver: {exc}")
                st.stop()

    st.session_state.setdefault("animating", False)
    st.session_state.setdefault("step_idx", 0)

    solver: HangmanSolver = st.session_state.solver
    animating = bool(st.session_state.animating)

    tab_random, tab_custom = st.tabs(["Random test word", "Custom word"])

    with tab_random:
        if st.button(
            "Solve a random word",
            type="primary",
            use_container_width=True,
            disabled=animating,
        ):
            start_solve(solver.random_test_word())
            st.rerun()

    with tab_custom:
        custom = st.text_input(
            "Word to solve",
            placeholder="e.g. placement",
            label_visibility="collapsed",
            key="custom_word",
            disabled=animating,
        )
        word_preview = normalize_word(custom)
        st.markdown(
            f'<div class="preview-box"><span class="preview-label">Solver will use</span>'
            f"{word_preview if word_preview else '—'}</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Solve custom word",
            type="primary",
            use_container_width=True,
            disabled=animating,
        ):
            word = word_preview
            if not word:
                st.warning("Enter a word first.")
            elif not (3 <= len(word) <= 30):
                st.warning("Length must be between 3 and 30.")
            elif len(set(word)) <= 1:
                st.warning("Need more than one distinct letter.")
            else:
                start_solve(word)
                st.rerun()

    result: GameResult | None = st.session_state.get("result")
    if not result:
        st.markdown(
            '<p class="empty-hint">Pick a random held-out word or type one to watch the solver play.</p>',
            unsafe_allow_html=True,
        )
        return

    render_playback(result)


if __name__ == "__main__":
    main()
