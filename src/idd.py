"""Iterative detection and decoding (IDD) on top of the NVlabs baseline PUSCH receiver (Sionna 0.18).

The conventional channel estimate (LS, linear interpolation) is kept.  Detection is MMSE with parallel interference
cancellation (Sionna MMSEPICDetector), which takes the decoder's extrinsic LLRs as prior.  One round: detect -> layer
demap -> descramble -> undo the code-block interleaving -> LDPC with soft codeword output -> extrinsic = a posteriori -
input -> back to the transmitted bit order -> scramble -> prior of the next detection.  After the last round the TB
decoder of the baseline (same BP iterations) gives the bits and the CRC.  With zero rounds this is MMSE-PIC without a
prior, i.e. the LMMSE receiver.

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
                 cn_type: str = "boxplus-phi") -> None:
        super().__init__()
        self._est = pusch_rx._channel_estimator
        self._demap = pusch_rx._layer_demapper
        self._tbd = pusch_rx._tb_decoder
        self._tbe = self._tbd._tb_encoder
        self._descr = self._tbd._descrambler
        self._rounds = rounds
        self._pic = MMSEPICDetector("bit", pusch_rx._resource_grid, pusch_rx._stream_management, "maxlog",
                                    num_iter=1, constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol)
        self._cw = LDPC5GDecoder(self._tbe.ldpc_encoder, num_iter=num_bp_iter, cn_type=cn_type, hard_out=False,
                                 return_infobits=False)
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
        """llr: [batch, num_tx, coded bits] in transmitted order -> extrinsic LLRs of the decoder, same order."""
        x = self._descr(llr) if self._descr is not None else llr
        b, t = tf.shape(x)[0], self._tbe.num_tx
        x = tf.concat([x, tf.zeros([b, t, self._fill], x.dtype)], axis=-1)
        x = tf.gather(x, self._perm_inv, axis=-1)
        x = tf.reshape(x, [b, t, self._num_cbs, self._n_cb])
        ext = self._sign * self._cw(x) - x
        ext = tf.gather(tf.reshape(ext, [b, t, -1]), self._perm, axis=-1)[..., :self._coded]
        return self._descr(ext) if self._descr is not None else ext

    def call(self, inputs):
        y, no = inputs
        h_hat, err_var = self._est([y, no])
        llr_det = None
        prior = None
        for r in range(self._rounds + 1):
            if prior is None:
                shape = self._pic_shape(y)
                prior = tf.zeros(shape, tf.float32)
            llr_det = self._pic([y, h_hat, prior, err_var, no])            # [batch, num_tx, 1, coded bits]
            if not self._checked_demapper:
                flat = self._demap(llr_det)
                assert flat.shape[-1] == llr_det.shape[-1] and bool(tf.reduce_all(tf.equal(flat, llr_det[:, :, 0, :]))), \
                    "the layer demapper is not a reshape"
                self._checked_demapper = True
            if r < self._rounds:
                prior = self._extrinsic(self._demap(llr_det))[:, :, None, :]
        return self._tbd(self._demap(llr_det))

    def _pic_shape(self, y):
        return [tf.shape(y)[0], self._tbe.num_tx, 1, self._coded]
