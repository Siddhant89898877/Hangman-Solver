"""BiLSTM Hangman solver by Siddhant Kumawat — same fusion logic as the pretrained test notebook."""

from __future__ import annotations

import math
import random
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .bilstm_numpy import NumpyHangmanBiLSTM, softmax
from .game import MAX_TRIES, DATA_DIR, MODELS_DIR, clean_word_list

CHAR_TO_IDX = {chr(i + 97): i + 1 for i in range(26)}
CHAR_TO_IDX["_"] = 27
PAD_IDX = 0
MAX_LEN = 30


def encode_word(word: str, max_len: int = MAX_LEN) -> List[int]:
    enc = [CHAR_TO_IDX.get(c, PAD_IDX) for c in word.lower()]
    return (enc + [PAD_IDX] * max_len)[:max_len]


def compute_conditional_prior(candidates: List[str]) -> Dict[str, float]:
    cnt = Counter("".join(candidates))
    tot = sum(cnt.values())
    if tot == 0:
        return {ch: 0.0 for ch in string.ascii_lowercase}
    return {ch: cnt[ch] / tot for ch in string.ascii_lowercase}


def compute_info_gain(candidates: List[str]) -> Dict[str, float]:
    C = len(candidates)
    if C == 0:
        return {ch: 0.0 for ch in string.ascii_lowercase}
    mem: Dict[str, int] = defaultdict(int)
    for w in candidates:
        for ch in set(w):
            mem[ch] += 1
    ig = {}
    for ch in string.ascii_lowercase:
        yes = mem[ch]
        no = C - yes
        py, pn = yes / C, no / C
        val = 0.0
        if py > 0:
            val += py * (math.log2(C) - math.log2(yes))
        if pn > 0 and no > 0:
            val += pn * (math.log2(C) - math.log2(no))
        ig[ch] = val
    return ig


def _encode_word_pool(words: List[str]) -> np.ndarray:
    if not words:
        return np.zeros((0, 0), dtype=np.uint8)
    length = len(words[0])
    arr = np.empty((len(words), length), dtype=np.uint8)
    for i, word in enumerate(words):
        arr[i] = np.frombuffer(word.encode("ascii"), dtype=np.uint8) - 97
    return arr


def _filter_candidates(
    pool: List[str], encoded: np.ndarray, pattern: str
) -> Tuple[List[str], np.ndarray]:
    if encoded.size == 0:
        return [], encoded
    known = [i for i, ch in enumerate(pattern) if ch != "_"]
    if not known:
        return pool, encoded
    cols = np.asarray(known, dtype=np.intp)
    vals = np.frombuffer("".join(pattern[i] for i in known).encode("ascii"), dtype=np.uint8) - 97
    mask = (encoded[:, cols] == vals).all(axis=1)
    idxs = np.flatnonzero(mask)
    if idxs.size == encoded.shape[0]:
        return pool, encoded
    return [pool[i] for i in idxs], encoded[idxs]


def compute_conditional_prior_arr(encoded: np.ndarray) -> Dict[str, float]:
    if encoded.size == 0:
        return {ch: 0.0 for ch in string.ascii_lowercase}
    counts = np.bincount(encoded.ravel(), minlength=26).astype(np.float64)
    tot = float(counts.sum())
    if tot == 0:
        return {ch: 0.0 for ch in string.ascii_lowercase}
    return {chr(97 + i): float(counts[i] / tot) for i in range(26)}


def compute_info_gain_arr(encoded: np.ndarray) -> Dict[str, float]:
    n_words, length = encoded.shape if encoded.ndim == 2 else (0, 0)
    if n_words == 0:
        return {ch: 0.0 for ch in string.ascii_lowercase}
    presence = np.zeros((n_words, 26), dtype=np.uint8)
    rows = np.arange(n_words)
    for pos in range(length):
        presence[rows, encoded[:, pos]] = 1
    yes = presence.sum(axis=0).astype(np.float64)
    log_c = math.log2(n_words)
    ig = {}
    for i in range(26):
        hit = yes[i]
        miss = n_words - hit
        val = 0.0
        if hit > 0:
            val += (hit / n_words) * (log_c - math.log2(hit))
        if miss > 0:
            val += (miss / n_words) * (log_c - math.log2(miss))
        ig[chr(97 + i)] = val
    return ig


@dataclass
class GuessStep:
    letter: str
    hit: bool
    pattern: str
    tries_left: int
    n_guess: int


@dataclass
class GameResult:
    word: str
    result: str  # WIN | LOSS
    tries_used: int
    n_guesses: int
    steps: List[GuessStep] = field(default_factory=list)
    guessed_letters: str = ""


@dataclass
class HangmanSolver:
    model_all: NumpyHangmanBiLSTM
    model_short: NumpyHangmanBiLSTM
    short_words: List[str]
    train_words: List[str]
    test_words: List[str]
    P_uni: Dict[str, float]
    P_bi: Dict[Tuple[str, str], float]
    short_by_len: Dict[int, List[str]] = field(default_factory=dict)
    short_arr: Dict[int, np.ndarray] = field(default_factory=dict)
    alpha: float = 0.8
    beta: float = 0.0
    gamma: float = 0.2
    w1: float = 0.3
    w2: float = 0.2
    w3: float = 0.5

    def __post_init__(self) -> None:
        if not self.short_by_len:
            by_len: Dict[int, List[str]] = defaultdict(list)
            for w in self.short_words:
                by_len[len(w)].append(w)
            self.short_by_len = dict(by_len)
        if not self.short_arr:
            self.short_arr = {length: _encode_word_pool(words) for length, words in self.short_by_len.items()}

    def guess_letter(self, pattern: str, guessed_letters: Set[str]) -> str:
        pattern = pattern.lower()
        L = len(pattern)
        guessed = set(guessed_letters)

        vowels = sorted(list("eaiou"), key=lambda c: -self.P_uni[c])
        consonants = [c for c in string.ascii_lowercase if c not in vowels]
        consonants.sort(key=lambda c: -self.P_uni[c])

        if L <= 3:
            for ch in vowels:
                if ch not in guessed:
                    return ch
            for ch in consonants:
                if ch not in guessed:
                    return ch
            return "e"

        pool = self.short_by_len.get(L, [])
        encoded = self.short_arr.get(L)
        if encoded is None:
            encoded = _encode_word_pool(pool)
        _, cand_arr = _filter_candidates(pool, encoded, pattern)

        net = self.model_all if L > 16 else self.model_short
        logits = net([encode_word(pattern, MAX_LEN)], [L])
        probs = softmax(logits, axis=1)[0]
        model_p = {string.ascii_lowercase[i]: float(probs[i]) for i in range(26)}

        stat_scores: Dict[str, float] = {}
        if L <= 16:
            cp = compute_conditional_prior_arr(cand_arr)
            for ch in string.ascii_lowercase:
                if ch in guessed:
                    continue
                s1 = self.P_uni.get(ch, 0.0)
                bs = []
                for i, pc in enumerate(pattern):
                    if pc == "_":
                        lp = pattern[i - 1] if i > 0 else None
                        rp = pattern[i + 1] if i < L - 1 else None
                        if lp and lp != "_":
                            bs.append(self.P_bi.get((lp, ch), 0.0))
                        if rp and rp != "_":
                            bs.append(self.P_bi.get((ch, rp), 0.0))
                s2 = sum(bs) / len(bs) if bs else 0.0
                s3 = cp.get(ch, 0.0)
                stat_scores[ch] = (
                    self.w1 * math.log(s1 + 1e-12)
                    + self.w2 * math.log(s2 + 1e-12)
                    + self.w3 * math.log(s3 + 1e-12)
                )

        ig_scores = compute_info_gain_arr(cand_arr) if L <= 16 else {}

        best_score, best_ch = -1e9, None
        for ch in string.ascii_lowercase:
            if ch in guessed:
                continue
            m = math.log(model_p.get(ch, 1e-12))
            st = stat_scores.get(ch, 0.0)
            ig = ig_scores.get(ch, 0.0)
            score = self.alpha * m + (1 - self.alpha) * (self.beta * st + self.gamma * ig)
            if score > best_score:
                best_score, best_ch = score, ch

        return best_ch or "e"

    def play(self, secret_word: str, max_tries: int = MAX_TRIES) -> GameResult:
        secret = secret_word.lower().strip()
        if not secret.isalpha() or not (3 <= len(secret) <= 30):
            raise ValueError("Word must be alphabetic, length 3–30.")
        if len(set(secret)) <= 1:
            raise ValueError("Word must contain more than one distinct letter.")

        guessed: Set[str] = set()
        revealed = ["_"] * len(secret)
        tries_remaining = max_tries
        n_guesses = 0
        steps: List[GuessStep] = []

        while True:
            pattern = "".join(revealed)
            if "_" not in pattern:
                return GameResult(
                    word=secret,
                    result="WIN",
                    tries_used=max_tries - tries_remaining,
                    n_guesses=n_guesses,
                    steps=steps,
                    guessed_letters="".join(sorted(guessed)),
                )
            if tries_remaining <= 0:
                return GameResult(
                    word=secret,
                    result="LOSS",
                    tries_used=max_tries,
                    n_guesses=n_guesses,
                    steps=steps,
                    guessed_letters="".join(sorted(guessed)),
                )

            letter = self.guess_letter(pattern, guessed)
            guessed.add(letter)
            n_guesses += 1

            if letter in secret:
                for i, ch in enumerate(secret):
                    if ch == letter:
                        revealed[i] = ch
                hit = True
            else:
                tries_remaining -= 1
                hit = False

            steps.append(
                GuessStep(
                    letter=letter,
                    hit=hit,
                    pattern="".join(revealed),
                    tries_left=tries_remaining,
                    n_guess=n_guesses,
                )
            )

    def random_test_word(self, rng: Optional[random.Random] = None) -> str:
        r = rng or random
        return r.choice(self.test_words)


def load_word_file(path: Path) -> List[str]:
    raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return clean_word_list(raw)


def load_solver() -> HangmanSolver:
    """Load models + priors. Expensive — cache at app level."""
    train_path = DATA_DIR / "train_words.txt"
    test_path = DATA_DIR / "test_words.txt"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing {test_path}")

    train_words = load_word_file(train_path)
    test_words = load_word_file(test_path)
    short_words = [w for w in train_words if 0 < len(w) <= 16]
    del train_words

    uni_cnt = Counter("".join(short_words))
    total_u = sum(uni_cnt.values())
    P_uni = {ch: uni_cnt[ch] / total_u for ch in string.ascii_lowercase}

    bi_cnt = Counter((a, b) for w in short_words for a, b in zip(w, w[1:]))
    total_b = sum(bi_cnt.values())
    P_bi = {bg: bi_cnt[bg] / total_b for bg in bi_cnt}

    model_all_h5 = MODELS_DIR / "model_all.weights.h5"
    model_short_h5 = MODELS_DIR / "model_short.weights.h5"
    if not model_all_h5.is_file() or not model_short_h5.is_file():
        raise FileNotFoundError(
            f"Missing model weights in {MODELS_DIR} "
            "(expected model_all.weights.h5 and model_short.weights.h5)."
        )

    model_all = NumpyHangmanBiLSTM.from_weights_h5(model_all_h5)
    model_short = NumpyHangmanBiLSTM.from_weights_h5(model_short_h5)

    return HangmanSolver(
        model_all=model_all,
        model_short=model_short,
        short_words=short_words,
        train_words=[],
        test_words=test_words,
        P_uni=P_uni,
        P_bi=P_bi,
    )
