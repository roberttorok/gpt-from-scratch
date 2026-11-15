from math import sqrt
import ctypes, os, random

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkernel.so")
)

F, I, R, PI = (
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_int),
)
_lib.mat_mul.argtypes = [F, I, I, I, I, I, F, I, I, I, I, I, F, I, I, I]
_lib.mat_mul.restype = None

_lib.mat_mul_add.argtypes = [F, I, I, I, I, I, F, I, I, I, I, I, F, I, I, I]
_lib.mat_mul_add.restype = None

_lib.scale.argtypes = [F, I, I, I, I, I, F, I, I, I, R]
_lib.scale.restype = None

_lib.scale_bw.argtypes = [F, I, I, I, I, I, F, I, I, I, R]
_lib.scale_bw.restype = None

_lib.softmax.argtypes = [F, I, I, I, I, I, F, I, I, I]
_lib.softmax.restype = None

_lib.softmax_bw.argtypes = [F, F, I, I, I, I, I, F, I, I, I]
_lib.softmax_bw.restype = None

_lib.casual_mask.argtypes = [F, I, I, I, I, I, F, I, I, I]
_lib.casual_mask.restype = None

_lib.add.argtypes = [F, I, I, I, I, I, F, I, I, I, I, I, F, I, I, I]
_lib.add.restype = None

_lib.casual_mask_bw.argtypes = [F, I, I, I, I, I, F, I, I, I]
_lib.casual_mask_bw.restype = None

_lib.add_bw.argtypes = [F, I, I, I, I, I, F, I, I, I]
_lib.add_bw.restype = None

_lib.relu.argtypes = [F, I, I, I, I, I, F, I, I, I]
_lib.relu.restype = None


_lib.relu_bw.argtypes = [F, F, I, I, I, I, I, F, I, I, I]
_lib.relu_bw.restype = None

_lib.rms_norm.argtypes = [F, I, I, I, I, I, F, I, I, I]
_lib.rms_norm.restype = None

_lib.rms_norm_bw.argtypes = [F, F, I, I, I, I, I, F, I, I, I]
_lib.rms_norm_bw.restype = None

_lib.cross_entropy.argtypes = [F, I, I, I, I, I, F, I, I, I, PI, F]
_lib.cross_entropy.restype = None

_lib.cross_entropy_bw.argtypes = [F, I, I, I, I, I, PI, F, I, I, I, R]
_lib.cross_entropy_bw.restype = None

_lib.embed.argtypes = [F, I, I, I, I, I, F, I, I, I, I, I, PI]
_lib.embed.restype = None

_lib.embed_bw.argtypes = [F, I, I, I, I, I, F, I, I, I, I, I, PI]
_lib.embed_bw.restype = None

_lib.learn_simple.argtypes = [F, F, I, I, I, I, I, R]
_lib.learn_simple.restype = None


class Tensor:
    def __init__(
        self,
        rows,
        cols,
        data=None,
        grad=None,
        offset=0,
        strides=None,
        parents=None,
        rand=False,
    ):
        self.rows = rows
        self.cols = cols
        self.data = data if data is not None else (ctypes.c_float * (rows * cols))()
        self.grad = grad if grad is not None else (ctypes.c_float * (rows * cols))()
        self.offset = offset if offset is not None else 0
        self.strides = strides if strides is not None else (cols, 1)
        self.parents = parents if parents is not None else []
        self.backward_fn = None

        if rand and data is None:
            size = rows * cols
            for i in range(size):
                self.data[i] = random.uniform(-1, 1) * sqrt(3.0 / rows)

    def col_view(self, a, b):
        # offset + x * stride[0] + (y * strides[1])
        return Tensor(
            rows=self.rows,
            cols=(b - a),
            data=self.data,
            grad=self.grad,
            offset=(self.offset + (a * self.strides[1])),
            strides=self.strides,
            parents=[self],
        )

    def T(self):
        return Tensor(
            rows=self.cols,
            cols=self.rows,
            data=self.data,
            grad=self.grad,
            offset=(self.offset),
            strides=(self.strides[1], self.strides[0]),
            parents=[self],
        )


def mat_mul(a: Tensor, b: Tensor, result: Tensor = None):
    if a.cols != b.rows:
        raise ValueError("matrix dimensions do not match")

    if result is None:
        result = Tensor(rows=a.rows, cols=b.cols, parents=[a, b])
    else:
        result.parents = [a, b]

    _lib.mat_mul(
        a.data,
        a.rows,
        a.cols,
        a.offset,
        a.strides[0],
        a.strides[1],
        b.data,
        b.rows,
        b.cols,
        b.offset,
        b.strides[0],
        b.strides[1],
        result.data,
        result.offset,
        result.strides[0],
        result.strides[1],
    )

    def backward_fn():
        b_t = b.T()
        res = result
        _lib.mat_mul_add(
            res.grad,
            res.rows,
            res.cols,
            res.offset,
            res.strides[0],
            res.strides[1],
            b_t.data,
            b_t.rows,
            b_t.cols,
            b_t.offset,
            b_t.strides[0],
            b_t.strides[1],
            a.grad,
            a.offset,
            a.strides[0],
            a.strides[1],
        )

        a_t = a.T()
        _lib.mat_mul_add(
            a_t.data,
            a_t.rows,
            a_t.cols,
            a_t.offset,
            a_t.strides[0],
            a_t.strides[1],
            res.grad,
            res.rows,
            res.cols,
            res.offset,
            res.strides[0],
            res.strides[1],
            b.grad,
            b.offset,
            b.strides[0],
            b.strides[1],
        )

    result.backward_fn = backward_fn
    return result


def scale(src: Tensor, value):
    result = Tensor(rows=src.rows, cols=src.cols, parents=[src])

    _lib.scale(
        src.data,
        src.rows,
        src.cols,
        src.offset,
        src.strides[0],
        src.strides[1],
        result.data,
        result.offset,
        result.strides[0],
        result.strides[1],
        value,
    )

    def backward_fn():
        _lib.scale_bw(
            src.grad,
            src.rows,
            src.cols,
            src.offset,
            src.strides[0],
            src.strides[1],
            result.grad,
            result.offset,
            result.strides[0],
            result.strides[1],
            value,
        )

    result.backward_fn = backward_fn

    return result


def softmax(src: Tensor):
    result = Tensor(rows=src.rows, cols=src.cols, parents=[src])
    _lib.softmax(
        src.data,
        src.rows,
        src.cols,
        src.offset,
        src.strides[0],
        src.strides[1],
        result.data,
        result.offset,
        result.strides[0],
        result.strides[1],
    )

    def backward_fn():
        _lib.softmax_bw(
            result.data,
            result.grad,
            result.rows,
            result.cols,
            result.offset,
            result.strides[0],
            result.strides[1],
            src.grad,
            src.offset,
            src.strides[0],
            src.strides[1],
        )

    result.backward_fn = backward_fn
    return result


def casual_mask(src: Tensor):
    result = Tensor(rows=src.rows, cols=src.cols, parents=[src])
    _lib.casual_mask(
        src.data,
        src.rows,
        src.cols,
        src.offset,
        src.strides[0],
        src.strides[1],
        result.data,
        result.offset,
        result.strides[0],
        result.strides[1],
    )

    def backward_fn():
        _lib.casual_mask_bw(
            src.grad,
            src.rows,
            src.cols,
            src.offset,
            src.strides[0],
            src.strides[1],
            result.grad,
            result.offset,
            result.strides[0],
            result.strides[1],
        )

    result.backward_fn = backward_fn
    return result


def add(a: Tensor, b: Tensor):
    result = Tensor(rows=a.rows, cols=b.cols, parents=[a, b])

    _lib.add(
        a.data,
        a.rows,
        a.cols,
        a.offset,
        a.strides[0],
        a.strides[1],
        b.data,
        b.rows,
        b.cols,
        b.offset,
        b.strides[0],
        b.strides[1],
        result.data,
        result.offset,
        result.strides[0],
        result.strides[1],
    )

    def backward_fn():
        _lib.add_bw(
            a.grad,
            a.rows,
            a.cols,
            a.offset,
            a.strides[0],
            a.strides[1],
            result.grad,
            result.offset,
            result.strides[0],
            result.strides[1],
        )
        _lib.add_bw(
            b.grad,
            b.rows,
            b.cols,
            b.offset,
            b.strides[0],
            b.strides[1],
            result.grad,
            result.offset,
            result.strides[0],
            result.strides[1],
        )

    result.backward_fn = backward_fn

    return result


def relu(src: Tensor):
    result = Tensor(rows=src.rows, cols=src.cols, parents=[src])
    _lib.relu(
        src.data,
        src.rows,
        src.cols,
        src.offset,
        src.strides[0],
        src.strides[1],
        result.data,
        result.offset,
        result.strides[0],
        result.strides[1],
    )

    def backward_fn():
        _lib.relu_bw(
            src.data,
            src.grad,
            src.rows,
            src.cols,
            src.offset,
            src.strides[0],
            src.strides[1],
            result.grad,
            result.offset,
            result.strides[0],
            result.strides[1],
        )

    result.backward_fn = backward_fn
    return result


def rms_norm(src: Tensor):
    result = Tensor(rows=src.rows, cols=src.cols, parents=[src])
    _lib.rms_norm(
        src.data,
        src.rows,
        src.cols,
        src.offset,
        src.strides[0],
        src.strides[1],
        result.data,
        result.offset,
        result.strides[0],
        result.strides[1],
    )

    def backward_fn():
        _lib.rms_norm_bw(
            src.data,
            src.grad,
            src.rows,
            src.cols,
            src.offset,
            src.strides[0],
            src.strides[1],
            result.grad,
            result.offset,
            result.strides[0],
            result.strides[1],
        )

    result.backward_fn = backward_fn
    return result


def cross_entropy(logits: Tensor, targets):
    buf = Tensor(
        rows=logits.rows, cols=logits.cols
    )  # Throwable, only needed for memory
    result = Tensor(rows=1, cols=1, parents=[logits])  # l
    c_targets = (ctypes.c_int * len(targets))(*targets)
    _lib.cross_entropy(
        logits.data,
        logits.rows,
        logits.cols,
        logits.offset,
        logits.strides[0],
        logits.strides[1],
        buf.data,
        buf.offset,
        buf.strides[0],
        buf.strides[1],
        c_targets,
        result.data,
    )

    def backward_fn():
        _lib.cross_entropy_bw(
            buf.data,
            buf.rows,
            buf.cols,
            buf.offset,
            buf.strides[0],
            buf.strides[1],
            c_targets,
            logits.grad,
            logits.offset,
            logits.strides[0],
            logits.strides[1],
            result.grad[0] / logits.rows,  # mean over rows, times the seed (1.0)
        )

    result.backward_fn = backward_fn

    return result


def embed(tokens: Tensor, ids):
    # tokens: the V x E embedding table, ids: list of T ints -> result is T x E
    for t in ids:
        if t < 0 or t >= tokens.rows:
            raise ValueError("token id out of range")
    c_ids = (ctypes.c_int * len(ids))(*ids)
    result = Tensor(rows=len(ids), cols=tokens.cols, parents=[tokens])
    _lib.embed(
        tokens.data,
        tokens.rows,
        tokens.cols,
        tokens.offset,
        tokens.strides[0],
        tokens.strides[1],
        result.data,
        result.rows,
        result.cols,
        result.offset,
        result.strides[0],
        result.strides[1],
        c_ids,
    )

    def backward_fn():
        _lib.embed_bw(
            tokens.grad,
            tokens.rows,
            tokens.cols,
            tokens.offset,
            tokens.strides[0],
            tokens.strides[1],
            result.grad,
            result.rows,
            result.cols,
            result.offset,
            result.strides[0],
            result.strides[1],
            c_ids,
        )

    result.backward_fn = backward_fn
    return result


def learn_simple(all_params: list[Tensor], lr):
    for param in all_params:
        _lib.learn_simple(
            param.data,
            param.grad,
            param.rows,
            param.cols,
            param.offset,
            param.strides[0],
            param.strides[1],
            lr,
        )
