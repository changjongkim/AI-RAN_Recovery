"""Iterative detection and decoding (IDD) on top of the NVlabs baseline PUSCH receiver (Sionna 0.18).

The conventional channel estimate (LS, linear interpolation) is kept.  Detection is MMSE with parallel interference
cancellation (Sionna MMSEPICDetector), which takes the decoder's extrinsic LLRs as prior.  One round: detect -> layer
demap -> descramble -> undo the code-block interleaving -> LDPC with soft codeword output -> extrinsic = a posteriori -
input -> back to the transmitted bit order -> scramble -> prior of the next detection.  After the last round the TB
decoder of the baseline (same BP iterations) gives the bits and the CRC.  With zero rounds this is MMSE-PIC without a
prior, i.e. the LMMSE receiver.
Feedback (found on CPU runs, 2026-10-15): (1) the extrinsic of a block that does not converge can be large and wrong;
fed back at full size it makes the PIC detector cancel wrong symbols of that user and break the other user; (2) Sionna's
LDPC decoder clips its messages at +-20 while channel LLRs reach 30-60, so output minus input flips the sign of strong
bits.  Hence by default (gate) only code blocks whose CB CRC passes are fed back, re-encoded from the decoded bits as
known bits (+-20), and the others get at most +-fail_clip (0: nothing): soft-input successive interference
cancellation of the users that decode.  A TB counts as decoded in the first round whose CRC passes (a receiver stops
there), so the result is never worse than round 0.

Two checks run when the layer is built and stop the run if they fail: the layer demapper of a one-layer PUSCH is a plain
reshape (so the prior can be put back by reshaping), and the soft output of the LDPC decoder has the sign convention of
its input (so extrinsic = output - input).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sionna.fec.ldpc import LDPC5GDecoder
from sionna.ofdm import MMSEPICDetector


class IddPusch(tf.keras.layers.Layer):
    """Replaces the PUSCHReceiver inside an NVlabs BaselineReceiver (baseline_lslin_lmmse); returns (bits, TB CRC)."""

    def __init__(self, pusch_rx, rounds: int, num_bp_iter: int, num_bits_per_symbol: int,
                 cn_type: str = "boxplus-phi", prior_clip: float = 0.0, prior_scale: float = 1.0,
                 gate: bool = True, fail_clip: float = 0.0) -> None:
        super().__init__()
        self._est = pusch_rx._channel_estimator
        self._demap = pusch_rx._layer_demapper
        self._tbd = pusch_rx._tb_decoder
        self._tbe = self._tbd._tb_encoder
        self._descr = self._tbd._descrambler
        self._rounds = rounds
        self._clip, self._scale = prior_clip, prior_scale
        self._gate, self._fail_clip = gate, fail_clip
        self._known = 20.0                       # prior of a bit of a code block whose CRC passed
        self._llr_max = 20.0                     # Sionna's LDPC message limit
        self._pic = MMSEPICDetector("bit", pusch_rx._resource_grid, pusch_rx._stream_management, "maxlog",
                                    num_iter=1, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol)
        self._cw = LDPC5GDecoder(self._tbe.ldpc_encoder, num_iter=num_bp_iter, cn_type=cn_type, hard_out=False,
                                 return_infobits=False)
        self._info = LDPC5GDecoder(self._tbe.ldpc_encoder, num_iter=num_bp_iter, cn_type=cn_type, hard_out=True,
                                   return_infobits=True)
        self._cb_crc = self._tbd._cb_crc_decoder
        tbe = self._tbe
        self._n_cb = int(tbe.ldpc_encoder.n)
        self._num_cbs = int(tbe.num_cbs)
        self._coded = int(np.sum(tbe.cw_lengths))
        self._fill = self._n_cb * self._num_cbs - self._coded
        self._perm = tf.constant(tbe._output_perm, tf.int32)
        self._perm_inv = tf.constant(tbe.output_perm_inv, tf.int32)
        self._sign = self._check_decoder_sign()
        self._checked_demapper = False

    def _check_decoder_sign(self) -> float:
        enc = self._tbe.ldpc_encoder
        u = tf.cast(tf.random.uniform([4, enc.k], 0, 2, tf.int32), tf.float32)
        c = enc(u)
        llr = 6.0 * (2.0 * c - 1.0)                     # Sionna: positive LLR for bit 1
        llr = llr + tf.random.normal(tf.shape(llr), stddev=4.0)
        post = self._cw(llr)
        agree = float(tf.reduce_mean(tf.cast(tf.equal(post > 0, c > 0.5), tf.float32)))
        assert agree > 0.99 or agree < 0.01, f"LDPC soft output agrees with the codeword in {agree:.3f} of the bits"
        return 1.0 if agree > 0.99 else -1.0

    def _extrinsic(self, llr: tf.Tensor) -> tf.Tensor:
        """llr: [batch, num_tx, coded bits] in transmitted order -> prior of the next detection, same order.

        gate (default): code blocks whose CB CRC passes are re-encoded from the decoded bits and fed back as known bits
        (+-known); the others get the extrinsic LLRs clipped to +-fail_clip (0: nothing).  Without gate: the extrinsic
        LLRs of every block.  The extrinsic is the decoder output minus its input clipped to the decoder's own limit
        (Sionna clips messages at +-20; without the clip, a channel LLR of 45 against an output of 20 gives -25)."""
        x = self._descr(llr) if self._descr is not None else llr
        b, t = tf.shape(x)[0], self._tbe.num_tx
        x = tf.concat([x, tf.zeros([b, t, self._fill], x.dtype)], axis=-1)
        x = tf.gather(x, self._perm_inv, axis=-1)
        x = tf.reshape(x, [b, t, self._num_cbs, self._n_cb])
        ext = self._sign * self._cw(x) - tf.clip_by_value(x, -self._llr_max, self._llr_max)
        if self._gate:
            u = self._info(x)                                                # [batch, num_tx, num_cbs, k]
            if self._cb_crc is not None:
                _, ok = self._cb_crc(u)                                      # [batch, num_tx, num_cbs, 1]
            else:
                _, tb_ok = self._tbd(llr)
                ok = tf.reshape(tb_ok, [b, t, 1, 1])
            known = self._known * (2.0 * self._tbe.ldpc_encoder(u) - 1.0)
            weak = tf.clip_by_value(ext, -self._fail_clip, self._fail_clip)
            ext = tf.where(tf.cast(ok, tf.bool), known, weak)
        ext = tf.gather(tf.reshape(ext, [b, t, -1]), self._perm, axis=-1)[..., :self._coded]
        return self._descr(ext) if self._descr is not None else ext

    def call(self, inputs):
        """Rounds stop counting for a TB once its CRC passes (a receiver stops there), so the result is never worse
        than round 0 (LMMSE)."""
        y, no = inputs
        h_hat, err_var = self._est([y, no])
        prior = tf.zeros(self._pic_shape(y), tf.float32)
        best_b = best_crc = None
        for r in range(self._rounds + 1):
            llr_det = self._pic([y, h_hat, prior, err_var, no])            # [batch, num_tx, 1, coded bits]
            if not self._checked_demapper:
                flat = self._demap(llr_det)
                assert flat.shape[-1] == llr_det.shape[-1] and bool(tf.reduce_all(tf.equal(flat, llr_det[:, :, 0, :]))), \
                    "the layer demapper is not a reshape"
                self._checked_demapper = True
            llr = self._demap(llr_det)
            b_hat, crc = self._tbd(llr)
            if best_b is None:
                best_b, best_crc = b_hat, crc
            else:
                take = tf.logical_and(crc, tf.logical_not(best_crc))
                best_b = tf.where(take[..., None], b_hat, best_b)
                best_crc = tf.logical_or(best_crc, crc)
            if r < self._rounds:
                ext = self._scale * self._extrinsic(llr)
                if self._clip > 0:
                    ext = tf.clip_by_value(ext, -self._clip, self._clip)
                prior = ext[:, :, None, :]
        return best_b, best_crc

    def _pic_shape(self, y):
        return [tf.shape(y)[0], self._tbe.num_tx, 1, self._coded]
