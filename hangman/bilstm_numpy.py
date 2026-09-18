"""NumPy inference for the Keras HangmanBiLSTM weights.

Author: Siddhant Kumawat

The saved ``.weights.h5`` files are Keras-native checkpoints
(embedding + 2× BiLSTM + dense). Inference runs in NumPy so TensorFlow
is not required at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import h5py
import numpy as np

Array = np.ndarray
TokenBatch = Union[Sequence[int], Sequence[Sequence[int]], Array]


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def softmax(logits: Array, axis: int = -1) -> Array:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass(frozen=True)
class LSTMWeights:
    kernel: Array  # (input_dim, 4 * units)
    recurrent: Array  # (units, 4 * units)
    bias: Array  # (4 * units,)

    @property
    def units(self) -> int:
        return int(self.recurrent.shape[0])


def _read_lstm_cell(group: h5py.Group) -> LSTMWeights:
    vars_grp = group["cell"]["vars"]
    return LSTMWeights(
        kernel=np.array(vars_grp["0"], dtype=np.float32),
        recurrent=np.array(vars_grp["1"], dtype=np.float32),
        bias=np.array(vars_grp["2"], dtype=np.float32),
    )


def _lstm_sequence(inputs: Array, mask: Array, weights: LSTMWeights, go_backwards: bool) -> Array:
    """inputs: (T, D), mask: (T,) bool. Returns hidden states (T, units)."""
    time_steps, _ = inputs.shape
    units = weights.units
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    outputs = np.zeros((time_steps, units), dtype=np.float32)
    indices = range(time_steps - 1, -1, -1) if go_backwards else range(time_steps)

    for t in indices:
        if not mask[t]:
            continue
        z = inputs[t] @ weights.kernel + h @ weights.recurrent + weights.bias
        z_i, z_f, z_c, z_o = np.split(z, 4)
        i = _sigmoid(z_i)
        f = _sigmoid(z_f)
        o = _sigmoid(z_o)
        c = f * c + i * np.tanh(z_c)
        h = o * np.tanh(c)
        outputs[t] = h
    return outputs


def _bilstm_sequence(inputs: Array, mask: Array, forward: LSTMWeights, backward: LSTMWeights) -> Array:
    fwd = _lstm_sequence(inputs, mask, forward, go_backwards=False)
    bwd = _lstm_sequence(inputs, mask, backward, go_backwards=True)
    return np.concatenate([fwd, bwd], axis=-1)


@dataclass
class NumpyHangmanBiLSTM:
    embedding: Array
    lstm_layers: List[Tuple[LSTMWeights, LSTMWeights]]
    dense_kernel: Array
    dense_bias: Array

    @classmethod
    def from_weights_h5(cls, path: Union[str, Path]) -> "NumpyHangmanBiLSTM":
        path = Path(path)
        with h5py.File(path, "r") as f:
            embedding = np.array(f["embedding"]["vars"]["0"], dtype=np.float32)
            layer_names = sorted(f["layers"].keys(), key=lambda n: (n != "bidirectional", n))
            lstm_layers: List[Tuple[LSTMWeights, LSTMWeights]] = []
            for name in layer_names:
                grp = f["layers"][name]
                lstm_layers.append(
                    (_read_lstm_cell(grp["forward_layer"]), _read_lstm_cell(grp["backward_layer"]))
                )
            dense_kernel = np.array(f["classifier"]["vars"]["0"], dtype=np.float32)
            dense_bias = np.array(f["classifier"]["vars"]["1"], dtype=np.float32)
        return cls(
            embedding=embedding,
            lstm_layers=lstm_layers,
            dense_kernel=dense_kernel,
            dense_bias=dense_bias,
        )

    def __call__(
        self,
        token_ids: TokenBatch,
        lengths: Union[int, Sequence[int], Array],
        training: bool = False,
    ) -> Array:
        del training
        ids = np.asarray(token_ids, dtype=np.int32)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2:
            raise ValueError(f"token_ids must be rank 1 or 2, got shape {ids.shape}")

        lens = np.asarray(lengths, dtype=np.int32).reshape(-1)
        if lens.shape[0] != ids.shape[0]:
            raise ValueError("lengths batch size must match token_ids")

        logits = np.stack([self._forward_one(row, int(length)) for row, length in zip(ids, lens)])
        return logits

    def _forward_one(self, token_ids: Array, length: int) -> Array:
        length = max(1, min(int(length), int(token_ids.shape[0])))
        token_ids = token_ids[:length]
        mask = token_ids != 0
        # Guard against a mismatch between the provided length and pad tokens.
        if not mask.any():
            mask = np.zeros_like(mask, dtype=bool)
            mask[0] = True
            length = 1
        hidden = self.embedding[token_ids]
        for forward, backward in self.lstm_layers:
            hidden = _bilstm_sequence(hidden, mask, forward, backward)
        final = hidden[length - 1]
        return final @ self.dense_kernel + self.dense_bias
