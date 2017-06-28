"""Conatains custom layers not implemented in Lasagne."""

from theano.tensor import switch, expm1
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T

from lasagne.layers.base import Layer


def selu(x):
    """
    Scaled exponential linear units as proposed in [1].

    [1] - https://arxiv.org/pdf/1706.02515.pdf
    """
    alpha = 1.6732632423543772848170429916717
    lam = 1.0507009873554804934193349852946
    return lam * switch(x >= 0.0, x, alpha * expm1(x))


class AlphaDropoutLayer(Layer):
    """Dropout layer.

    Sets values to zero with probability p. Will also converge the remaining
    neurons to mean 0 and a variance of 1 unit. See notes for disabling dropout
    during testing.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If ``True`` (the default), scale the input by ``1 / (1 - p)`` when
        dropout is enabled, to keep the expected output mean the same.
    shared_axes : tuple of int
        Axes to share the dropout mask over. By default, each value can be
        dropped individually. ``shared_axes=(0,)`` uses the same mask across
        the batch. ``shared_axes=(2, 3)`` uses the same mask across the
        spatial dimensions of 2D feature maps.

    Notes
    -----
    The alpha dropout layer is a regularizer that randomly sets input values to
    zero and also applies a normalization function to bring the mean to 0
    and the variance to 1; see [1]_for why this might improve
    generalization. The behaviour of the layer depends on the ``deterministic``
    keyword argument passed to :func:`lasagne.layers.get_output`. If ``True``,
    the layer behaves deterministically, and passes on the input unchanged. If
    ``False`` or not specified, dropout (and possibly scaling) is enabled.
    Usually, you would use ``deterministic=False`` at train time and
    ``deterministic=True`` at test time.
    References
    ----------
    .. [1] Klambauer, G., Unterthiner, T., Mayr, A., Hochreiter, S. (2017):
           Self-Normalizing Neural Networks. arXiv preprint: 1706.02515
    """

    def __init__(self, incoming, p=0.1, rescale=True, shared_axes=(),
                 **kwargs):
        """Class initialization."""
        super(AlphaDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams()
        self.p = p
        self.alpha = -1.7580993408473766
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """Apply alpha dropout."""
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))

            mask = self._srng.uniform(mask_shape,
                                      dtype=input.dtype) < retain_prob

            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)

            a = T.pow(retain_prob + self.alpha ** 2 * retain_prob *
                      (1 - retain_prob), -0.5)

            b = -a * (1 - retain_prob) * self.alpha

            return a * (input * mask + self.alpha * (1 - mask)) + b

if __name__ == "__main__":
    pass
