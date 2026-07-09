"""
Sequence-mixing blocks for Tiberius residual-stream backbones.

These layers are drop-in replacements for the biLSTM/LRU stack in
`lstm_residual_stream_model`. Each block is a complete pre-norm residual
unit: LayerNorm -> mixer -> gated FFN, with a residual add after each
of the two sub-layers. Input and output are both (B, L, d_model), so N
blocks stacked with `x = block_i(x)` form a residual-stream backbone.

Blocks:
    SlidingWindowAttentionBlock -- bidirectional block-local multi-head
        attention with ALiBi. Memory O(L * window * H), not O(L^2).
    DiagonalSSMBlock            -- S4D-style diagonal LTI state-space
        mixer. FFT-conv training / recurrent inference duality.
    BorderGatedSelfAttention    -- global multi-head self-attention whose
        key logits are biased by a predicted per-position "exon-border"
        score, so queries preferentially attend to positions the model
        thinks are exon borders (graph message-passing over borders).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization


class SlidingWindowAttentionBlock(tf.keras.layers.Layer):
    """
    Bidirectional block-local multi-head self-attention with ALiBi bias.

    The sequence is split into non-overlapping blocks of length `block_size`;
    each query block attends to itself and `halo_blocks` neighboring blocks
    on each side. Effective receptive field per position is
    `(2*halo_blocks + 1) * block_size`. Attention memory scales as
    O(L * window * H), independent of L beyond the linear factor.

    Uses only relative position information (ALiBi), so length
    generalization is by construction.

    Args:
        d_model:            input/output width.
        num_heads:          attention heads. Must divide d_model.
        block_size:         block length used to tile the sequence.
        halo_blocks:        neighbor blocks on each side in the k/v window.
        ffn_mult:           gated-FFN hidden width as a multiple of d_model.
        dropout:            attention and FFN dropout.
        use_alibi:          add ALiBi symmetric linear-distance bias per head.
        zero_init_residual: zero-init the two output projections so the block
                            is identity at initialization.
    """

    def __init__(self, d_model, num_heads=8, block_size=64, halo_blocks=1,
                 ffn_mult=4, dropout=0.0, use_alibi=True,
                 zero_init_residual=True, **kw):
        super().__init__(**kw)
        if d_model % num_heads != 0:
            raise ValueError(
                f'd_model ({d_model}) must be divisible by num_heads ({num_heads})'
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.halo_blocks = halo_blocks
        self.window_blocks = 2 * halo_blocks + 1
        self.window_size = self.window_blocks * block_size
        self.ffn_mult = ffn_mult
        self.dropout_rate = dropout
        self.use_alibi = use_alibi

        proj_init = 'zeros' if zero_init_residual else 'glorot_uniform'
        self.ln_attn = LayerNormalization(name='ln_attn')
        self.qkv = Dense(3 * d_model, use_bias=False, name='qkv')
        self.attn_out = Dense(d_model, kernel_initializer=proj_init,
                              name='attn_out')
        self.attn_drop = Dropout(dropout, name='attn_drop')

        self.ln_ffn = LayerNormalization(name='ln_ffn')
        # Gated FFN (GLU with GELU gate): project to 2 * ffn_mult * d, split.
        self.ffn_in = Dense(2 * ffn_mult * d_model, name='ffn_in')
        self.ffn_out = Dense(d_model, kernel_initializer=proj_init,
                             name='ffn_out')
        self.ffn_drop = Dropout(dropout, name='ffn_drop')

    def build(self, input_shape):
        if self.use_alibi:
            slopes = [2.0 ** (-8.0 * (i + 1) / self.num_heads)
                      for i in range(self.num_heads)]
            self.alibi_slopes = tf.constant(slopes, dtype=tf.float32)
        super().build(input_shape)

    def _window_attention(self, x, training):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        bs = self.block_size
        hb = self.halo_blocks
        H = self.num_heads
        Dh = self.head_dim

        # Pad sequence to a multiple of block_size.
        pad = (-L) % bs
        x_padded = tf.pad(x, [[0, 0], [0, pad], [0, 0]])
        Lp = L + pad
        nblocks = Lp // bs

        # qkv projection and split into heads.
        qkv = self.qkv(x_padded)
        qkv = tf.reshape(qkv, (B, Lp, 3, H, Dh))
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # Tile into blocks: (B, nblocks, block, H, Dh).
        q_blk = tf.reshape(q, (B, nblocks, bs, H, Dh))
        k_blk = tf.reshape(k, (B, nblocks, bs, H, Dh))
        v_blk = tf.reshape(v, (B, nblocks, bs, H, Dh))

        # Build key/value with halo by padding the block axis and taking
        # `window_blocks` contiguous slices — each query block sees itself
        # plus `halo_blocks` neighbors on each side.
        pad_blocks = tf.zeros((B, hb, bs, H, Dh), dtype=k_blk.dtype)
        k_pad = tf.concat([pad_blocks, k_blk, pad_blocks], axis=1)
        v_pad = tf.concat([pad_blocks, v_blk, pad_blocks], axis=1)
        k_win = tf.concat(
            [k_pad[:, i:i + nblocks] for i in range(self.window_blocks)],
            axis=2,
        )  # (B, nblocks, window, H, Dh)
        v_win = tf.concat(
            [v_pad[:, i:i + nblocks] for i in range(self.window_blocks)],
            axis=2,
        )

        scale = 1.0 / tf.math.sqrt(tf.cast(Dh, x.dtype))
        # (B, nblocks, H, block, window)
        scores = tf.einsum('bnqhd,bnkhd->bnhqk', q_blk, k_win) * scale

        # ALiBi: symmetric linear penalty on |q_pos - k_pos| per head.
        if self.use_alibi:
            q_pos = tf.cast(tf.range(bs), scores.dtype)
            k_pos = (tf.cast(tf.range(self.window_size), scores.dtype)
                     - tf.cast(hb * bs, scores.dtype))
            rel = k_pos[None, :] - q_pos[:, None]  # (block, window)
            slopes = tf.cast(self.alibi_slopes, scores.dtype)
            alibi = -tf.abs(rel)[None, None, None, :, :] * \
                    slopes[None, None, :, None, None]
            scores = scores + alibi

        # Mask keys that fall outside the true (unpadded) sequence.
        block_idx = tf.range(nblocks, dtype=tf.int32)[:, None]
        k_global = (block_idx * bs
                    + tf.range(self.window_size)[None, :]
                    - hb * bs)  # (nblocks, window)
        valid = tf.logical_and(k_global >= 0, k_global < L)
        neg_inf = tf.cast(-1e9, scores.dtype)
        mask = tf.cast(valid, scores.dtype)[None, :, None, None, :]
        scores = scores * mask + (1.0 - mask) * neg_inf

        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.attn_drop(attn, training=training)
        out = tf.einsum('bnhqk,bnkhd->bnqhd', attn, v_win)
        out = tf.reshape(out, (B, Lp, self.d_model))
        return out[:, :L, :]

    def call(self, x, training=None):
        h = self.ln_attn(x)
        h = self._window_attention(h, training)
        h = self.attn_out(h)
        x = x + h

        h = self.ln_ffn(x)
        a, b = tf.split(self.ffn_in(h), 2, axis=-1)
        h = a * tf.nn.gelu(b)
        h = self.ffn_drop(h, training=training)
        h = self.ffn_out(h)
        return x + h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'block_size': self.block_size,
            'halo_blocks': self.halo_blocks,
            'ffn_mult': self.ffn_mult,
            'dropout': self.dropout_rate,
            'use_alibi': self.use_alibi,
        })
        return cfg


class DiagonalSSMBlock(tf.keras.layers.Layer):
    """
    S4D-style diagonal LTI state-space mixer, bidirectional.

    Per channel: h_t = A_bar * h_{t-1} + B_bar * x_t,  y_t = 2*Re(C * h_t) + D * x_t.
    A is diagonal complex with Re(A) < 0 (softplus parameterization) so the
    system is stable. A_bar = exp(dt * A), B_bar = dt * B (Euler discretization;
    ZOH is a drop-in swap if needed later).

    Two run modes with equivalent mathematics:
      * mode="conv":      FFT convolution with the causal kernel
                          K[t] = 2 * Re(sum_n C_n * B_bar_n * A_bar_n^t).
                          Parallel, O(L log L) time, O(L) memory per channel.
                          Kernel materialization needs O(D * N * L) complex
                          during forward -- use "recurrent" for very long L.
      * mode="recurrent": explicit recurrence via tf.scan. O(N) state memory,
                          O(L) time. Intended for inference on long sequences
                          (e.g. whole-chromosome). Not recommended for training.

    Bidirectional: an independent SSM runs on reversed input; outputs summed.

    Args:
        d_model:            input/output width.
        state_size:         N, diagonal state size per channel.
        bidirectional:      run a second SSM on reversed input and sum.
        dt_min, dt_max:     init range for the learnable log-timestep.
        mode:               "conv" or "recurrent" (see above).
        ffn_mult:           gated-FFN hidden width as a multiple of d_model.
        dropout:            SSM output and FFN dropout.
        zero_init_residual: zero-init the two output projections.
    """

    def __init__(self, d_model, state_size=32, bidirectional=True,
                 dt_min=0.001, dt_max=0.1, mode='conv',
                 ffn_mult=4, dropout=0.0, zero_init_residual=True, **kw):
        super().__init__(**kw)
        if mode not in ('conv', 'recurrent'):
            raise ValueError(f"mode must be 'conv' or 'recurrent', got {mode!r}")
        self.d_model = d_model
        self.state_size = state_size
        self.bidirectional = bidirectional
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.mode = mode
        self.ffn_mult = ffn_mult
        self.dropout_rate = dropout

        proj_init = 'zeros' if zero_init_residual else 'glorot_uniform'
        self.ln_ssm = LayerNormalization(name='ln_ssm')
        self.ssm_in = Dense(d_model, name='ssm_in')
        self.ssm_out = Dense(d_model, kernel_initializer=proj_init,
                             name='ssm_out')
        self.ssm_drop = Dropout(dropout, name='ssm_drop')

        self.ln_ffn = LayerNormalization(name='ln_ffn')
        self.ffn_in = Dense(2 * ffn_mult * d_model, name='ffn_in')
        self.ffn_out = Dense(d_model, kernel_initializer=proj_init,
                             name='ffn_out')
        self.ffn_drop = Dropout(dropout, name='ffn_drop')

    def _init_direction(self, suffix):
        D, N = self.d_model, self.state_size
        # softplus^-1(0.5): initial |Re(A)| ~= 0.5.
        A_re_init = float(np.log(np.expm1(0.5)))
        # S4D-Lin imaginary init: Im(A_n) = pi * n.
        A_im_init = np.tile(np.pi * np.arange(1, N + 1, dtype='float32'), (D, 1))
        w = {}
        w['A_re_raw'] = self.add_weight(
            name=f'A_re_raw{suffix}', shape=(D, N),
            initializer=tf.keras.initializers.Constant(A_re_init))
        w['A_im'] = self.add_weight(
            name=f'A_im{suffix}', shape=(D, N),
            initializer=tf.keras.initializers.Constant(A_im_init))
        w['log_dt'] = self.add_weight(
            name=f'log_dt{suffix}', shape=(D,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=float(np.log(self.dt_min)),
                maxval=float(np.log(self.dt_max))))
        std = 0.5 ** 0.5
        for wname in ('B_re', 'B_im', 'C_re', 'C_im'):
            w[wname] = self.add_weight(
                name=f'{wname}{suffix}', shape=(D, N),
                initializer=tf.keras.initializers.RandomNormal(stddev=std))
        return w

    def build(self, input_shape):
        self.fwd = self._init_direction(suffix='_f')
        if self.bidirectional:
            self.bwd = self._init_direction(suffix='_b')
        self.D_skip = self.add_weight(
            name='D_skip', shape=(self.d_model,), initializer='ones')
        super().build(input_shape)

    def _params(self, direction):
        w = self.bwd if direction == 'b' else self.fwd
        A_re = -tf.nn.softplus(w['A_re_raw'])  # Re(A) < 0
        A = tf.complex(A_re, w['A_im'])
        B = tf.complex(w['B_re'], w['B_im'])
        C = tf.complex(w['C_re'], w['C_im'])
        dt = tf.exp(w['log_dt'])
        return A, B, C, dt

    def _kernel(self, L, direction):
        A, B, C, dt = self._params(direction)
        dt_c = tf.complex(dt, tf.zeros_like(dt))  # (D,)
        dtA = dt_c[:, None] * A                    # (D, N)
        B_bar = dt_c[:, None] * B                  # (D, N)  Euler
        t_f = tf.cast(tf.range(L), tf.float32)
        t_c = tf.complex(t_f, tf.zeros_like(t_f))  # (L,)
        # A_bar^t = exp(t * dtA). Intermediate is (D, N, L) complex64.
        A_bar_t = tf.exp(t_c[None, None, :] * dtA[:, :, None])
        K_c = tf.einsum('dn,dn,dnl->dl', C, B_bar, A_bar_t)  # (D, L)
        return 2.0 * tf.math.real(K_c)  # (D, L)

    def _conv1d_fft(self, x, K):
        # x: (B, L, D), K: (D, L) -> (B, L, D)
        L = tf.shape(x)[1]
        n = 2 * L
        n_len = tf.reshape(n, [1])
        x_t = tf.transpose(x, [0, 2, 1])           # (B, D, L)
        X = tf.signal.rfft(x_t, fft_length=n_len)  # (B, D, n/2+1)
        Kf = tf.signal.rfft(K, fft_length=n_len)   # (D, n/2+1)
        y_t = tf.signal.irfft(X * Kf[None, :, :],
                              fft_length=n_len)    # (B, D, n)
        y_t = y_t[:, :, :L]
        return tf.transpose(y_t, [0, 2, 1])        # (B, L, D)

    def _ssm_conv(self, x):
        L = tf.shape(x)[1]
        K = self._kernel(L, direction='f')
        y = self._conv1d_fft(x, K)
        if self.bidirectional:
            Kb = self._kernel(L, direction='b')
            y_b = self._conv1d_fft(tf.reverse(x, axis=[1]), Kb)
            y = y + tf.reverse(y_b, axis=[1])
        return y + x * self.D_skip[None, None, :]

    def _ssm_recurrent_dir(self, x, direction):
        A, B, C, dt = self._params(direction)
        dt_c = tf.complex(dt, tf.zeros_like(dt))
        A_bar = tf.exp(dt_c[:, None] * A)          # (D, N)
        B_bar = dt_c[:, None] * B                   # (D, N)

        if direction == 'b':
            x = tf.reverse(x, axis=[1])
        x_t = tf.transpose(x, [1, 0, 2])            # (L, B, D)
        x_c = tf.complex(x_t, tf.zeros_like(x_t))
        B_size = tf.shape(x)[0]
        h0 = tf.zeros((B_size, self.d_model, self.state_size),
                      dtype=tf.complex64)

        def step(h_prev, x_step):
            Bx = B_bar[None, :, :] * x_step[:, :, None]  # (B, D, N)
            return A_bar[None, :, :] * h_prev + Bx

        hs = tf.scan(step, x_c, initializer=h0)     # (L, B, D, N)
        y_c = tf.einsum('dn,lbdn->lbd', C, hs)
        y = 2.0 * tf.math.real(y_c)                  # (L, B, D)
        y = tf.transpose(y, [1, 0, 2])               # (B, L, D)
        if direction == 'b':
            y = tf.reverse(y, axis=[1])
        return y

    def _ssm_recurrent(self, x):
        y = self._ssm_recurrent_dir(x, 'f')
        if self.bidirectional:
            y = y + self._ssm_recurrent_dir(x, 'b')
        return y + x * self.D_skip[None, None, :]

    def call(self, x, training=None):
        h = self.ln_ssm(x)
        h = self.ssm_in(h)
        h = self._ssm_conv(h) if self.mode == 'conv' else self._ssm_recurrent(h)
        h = self.ssm_drop(h, training=training)
        h = self.ssm_out(h)
        x = x + h

        h = self.ln_ffn(x)
        a, b = tf.split(self.ffn_in(h), 2, axis=-1)
        h = a * tf.nn.gelu(b)
        h = self.ffn_drop(h, training=training)
        h = self.ffn_out(h)
        return x + h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'd_model': self.d_model,
            'state_size': self.state_size,
            'bidirectional': self.bidirectional,
            'dt_min': self.dt_min,
            'dt_max': self.dt_max,
            'mode': self.mode,
            'ffn_mult': self.ffn_mult,
            'dropout': self.dropout_rate,
        })
        return cfg


class BorderGatedSelfAttention(tf.keras.layers.Layer):
    """
    Global multi-head self-attention biased by a predicted per-position
    "exon-border" score.

    A small internal head (LN -> Dense(hidden, relu) -> Dense(1)) predicts
    an unactivated border logit for every position. That logit is added
    as an additive bias along the KEY axis of the attention score matrix
    before softmax, so all query positions attend more strongly to
    positions the model thinks are exon borders. This gives a soft,
    fully differentiable "graph message-passing over exon borders" without
    hard Top-K selection.

    Shape: (B, L, d_model) -> (B, L, d_model). Attention is full O(L^2)
    over the pooled sequence length. Intended for use at pooled
    resolution (e.g. after Tiberius's pool_size=9 reshape).

    The per-position border logits are exposed as an auxiliary tensor
    via the layer's `border_logits` argument in `call(..., return_border=True)`,
    so a builder can optionally attach an auxiliary supervision loss.
    """

    def __init__(self, d_model, num_heads=8, dropout=0.0,
                 zero_init_proj=True, border_hidden=None, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads "
                f"({num_heads})."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout
        self.zero_init_proj = zero_init_proj
        self.border_hidden = border_hidden or max(d_model // 2, 32)

    def build(self, input_shape):
        self.q_proj = Dense(self.d_model, name='q_proj')
        self.k_proj = Dense(self.d_model, name='k_proj')
        self.v_proj = Dense(self.d_model, name='v_proj')
        proj_init = 'zeros' if self.zero_init_proj else 'glorot_uniform'
        self.o_proj = Dense(self.d_model, kernel_initializer=proj_init,
                            name='o_proj')
        self.border_ln = LayerNormalization(name='border_ln')
        self.border_hidden_layer = Dense(self.border_hidden,
                                         activation='relu',
                                         name='border_hidden')
        self.border_out = Dense(1, name='border_logits')
        self.attn_dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=None, return_border=False):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]

        # Per-position border logits (pre-sigmoid) [B, L, 1]
        b = self.border_ln(x)
        b = self.border_hidden_layer(b)
        border_logits = self.border_out(b)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def _split(t):
            t = tf.reshape(t, [B, L, self.num_heads, self.head_dim])
            return tf.transpose(t, [0, 2, 1, 3])  # [B, H, L, dh]

        q = _split(q)
        k = _split(k)
        v = _split(v)

        scale = tf.math.rsqrt(tf.cast(self.head_dim, x.dtype))
        scores = tf.matmul(q, k, transpose_b=True) * scale  # [B, H, L, L]

        # Additive border bias broadcast over queries and heads:
        # border_logits: [B, L, 1] -> [B, 1, 1, L] (bias on KEY axis).
        border_bias = tf.transpose(border_logits, [0, 2, 1])  # [B, 1, L]
        border_bias = border_bias[:, tf.newaxis, :, :]        # [B, 1, 1, L]
        scores = scores + tf.cast(border_bias, scores.dtype)

        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.attn_dropout(attn, training=training)

        out = tf.matmul(attn, v)                     # [B, H, L, dh]
        out = tf.transpose(out, [0, 2, 1, 3])        # [B, L, H, dh]
        out = tf.reshape(out, [B, L, self.d_model])
        out = self.o_proj(out)

        if return_border:
            return out, border_logits
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate,
            'zero_init_proj': self.zero_init_proj,
            'border_hidden': self.border_hidden,
        })
        return cfg
