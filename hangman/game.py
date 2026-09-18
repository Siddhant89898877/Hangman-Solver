"""Hangman game engine by Siddhant Kumawat — matches notebook rules (6 wrong tries, a–z only)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DEFAULT_WORD_FILES = (DATA_DIR / "test_words.txt", DATA_DIR / "train_words.txt")
MAX_TRIES = 6
MIN_LEN, MAX_LEN = 3, 30


def clean_word_list(words: List[str], min_length: int = MIN_LEN, max_length: int = MAX_LEN) -> List[str]:
    cleaned = []
    for word in words:
        w = word.strip().lower()
        if min_length <= len(w) <= max_length and w.isalpha() and len(set(w)) > 1:
            cleaned.append(w)
    return cleaned


def load_words(paths: Optional[List[Path]] = None) -> List[str]:
    candidates = paths or list(DEFAULT_WORD_FILES)
    for path in candidates:
        if path.is_file():
            raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            cleaned = clean_word_list(raw)
            if cleaned:
                return cleaned
    raise FileNotFoundError(
        "No word list found. Place test_words.txt or train_words.txt in the data/ folder."
    )


@dataclass
class HangmanGame:
    secret: str
    max_tries: int = MAX_TRIES
    guessed: Set[str] = field(default_factory=set)
    wrong_guesses: int = 0
    status: str = "playing"  # playing | won | lost

    def __post_init__(self) -> None:
        self.secret = self.secret.lower()
        if not self.secret.isalpha():
            raise ValueError("Secret word must be alphabetic")

    @classmethod
    def new(cls, words: List[str], max_tries: int = MAX_TRIES, rng: Optional[random.Random] = None) -> "HangmanGame":
        r = rng or random
        return cls(secret=r.choice(words), max_tries=max_tries)

    @property
    def tries_left(self) -> int:
        return self.max_tries - self.wrong_guesses

    @property
    def pattern(self) -> str:
        return "".join(c if c in self.guessed else "_" for c in self.secret)

    @property
    def pattern_display(self) -> str:
        return " ".join(self.pattern)

    @property
    def correct_letters(self) -> Set[str]:
        return {c for c in self.guessed if c in self.secret}

    @property
    def missed_letters(self) -> Set[str]:
        return {c for c in self.guessed if c not in self.secret}

    def guess(self, letter: str) -> dict:
        """Apply a letter guess. Returns a result dict for the UI."""
        letter = letter.lower().strip()
        if self.status != "playing":
            return {"ok": False, "reason": "game_over", "hit": False}

        if len(letter) != 1 or not letter.isalpha():
            return {"ok": False, "reason": "invalid", "hit": False}

        if letter in self.guessed:
            return {"ok": False, "reason": "already", "hit": False}

        self.guessed.add(letter)
        hit = letter in self.secret

        if not hit:
            self.wrong_guesses += 1

        if "_" not in self.pattern:
            self.status = "won"
        elif self.tries_left <= 0:
            self.status = "lost"

        return {
            "ok": True,
            "reason": "hit" if hit else "miss",
            "hit": hit,
            "letter": letter,
            "status": self.status,
            "pattern": self.pattern,
            "tries_left": self.tries_left,
        }

    def to_state(self) -> dict:
        return {
            "secret": self.secret if self.status != "playing" else None,
            "pattern": self.pattern,
            "pattern_display": self.pattern_display,
            "guessed": sorted(self.guessed),
            "correct": sorted(self.correct_letters),
            "missed": sorted(self.missed_letters),
            "tries_left": self.tries_left,
            "wrong_guesses": self.wrong_guesses,
            "max_tries": self.max_tries,
            "status": self.status,
            "word_length": len(self.secret),
        }
