import struct, sys

from profile import perf_end, perf_start
from tensor import (
    Tensor,
    add,
    casual_mask,
    cross_entropy,
    embed,
    learn_simple,
    mat_mul,
    relu,
    rms_norm,
    scale,
    softmax,
)
import math, random


class Config:
    def __init__(self):
        self.learning_steps = 11000
        self.vocab_size = 100
        self.embedding_len = 256
        self.hidden_layers = 4
        self.heads = 4
        self.head_dim = self.embedding_len // self.heads
        self.vocalbulary = []
        self.learning_context_window = 64
        self.export_steps = 1000
        self.lr = 0.05


class Layer:
    def __init__(self, config: Config):
        expansion = 4
        self.q = Tensor(rows=config.embedding_len, cols=config.embedding_len, rand=True)
        self.k = Tensor(rows=config.embedding_len, cols=config.embedding_len, rand=True)
        self.v = Tensor(rows=config.embedding_len, cols=config.embedding_len, rand=True)
        self.wo = Tensor(
            rows=config.embedding_len, cols=config.embedding_len, rand=True
        )
        self.ff1 = Tensor(
            rows=config.embedding_len, cols=config.embedding_len * expansion, rand=True
        )
        self.ff2 = Tensor(
            rows=config.embedding_len * expansion, cols=config.embedding_len, rand=True
        )


class Model:
    def __init__(self, config: Config):
        self.layers = [Layer(config) for _ in range(config.hidden_layers)]
        self.lm_head = Tensor(
            rows=config.embedding_len, cols=config.vocab_size, rand=True
        )
        self.embeddings = Tensor(
            rows=config.vocab_size, cols=config.embedding_len, rand=True
        )
        self.pos_embeddings = Tensor(
            rows=config.learning_context_window, cols=config.embedding_len, rand=True
        )
        self.config = config
        self.all_params = []
        self.all_params.extend([self.embeddings, self.pos_embeddings, self.lm_head])
        for i in range(config.hidden_layers):
            self.all_params.extend(
                [
                    self.layers[i].ff1,
                    self.layers[i].ff2,
                    self.layers[i].k,
                    self.layers[i].q,
                    self.layers[i].v,
                    self.layers[i].wo,
                ]
            )

    def encode(self, text):
        ids = []
        for char in text:
            ids.append(self.config.vocalbulary.index(char))
        return ids

    def decode_single(self, id):
        return self.config.vocalbulary[id]

    def export(self, filename):
        c = self.config
        with open(filename, "wb") as f:
            vocab_bytes = "".join(c.vocalbulary).encode("utf-8")
            f.write(
                struct.pack(
                    "<6i",
                    c.embedding_len,
                    c.hidden_layers,
                    c.heads,
                    c.learning_context_window,
                    c.vocab_size,
                    len(vocab_bytes),
                )
            )
            f.write(vocab_bytes)
            for p in self.all_params:
                f.write(bytes(p.data))
            print(f"Exported to {filename}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            emb, layers, heads, ctx, vocab_size, vocab_len = struct.unpack(
                "<6i", f.read(6 * 4)
            )
            config = Config()
            config.embedding_len = emb
            config.heads = heads
            config.hidden_layers = layers
            config.learning_context_window = ctx
            config.head_dim = emb // heads
            config.vocab_size = vocab_size
            config.vocalbulary = list(f.read(vocab_len).decode("utf-8"))
            model = Model(config)
            for p in model.all_params:
                f.readinto(p.data)
        return model


def backpropagation(tensor: Tensor):
    seen = set()
    order = []

    def visit(node: Tensor):
        if node not in seen:
            seen.add(node)
            for p in node.parents:
                visit(p)
            order.append(node)

    visit(tensor)

    tensor.grad[tensor.offset] = 1.0

    for node in reversed(order):
        if node.backward_fn:
            node.backward_fn()


def gpt(model: Model, x: Tensor):
    model_config = model.config
    for j in range(model_config.hidden_layers):
        x_resid = x
        x = rms_norm(x)
        layer = model.layers[j]
        q = mat_mul(x, layer.q)
        k = mat_mul(x, layer.k)
        v = mat_mul(x, layer.v)
        attn_out = Tensor(rows=x.rows, cols=model_config.embedding_len)
        part_results = []
        for h in range(model_config.heads):
            q_h = q.col_view(h * model_config.head_dim, (h + 1) * model_config.head_dim)
            k_h = k.col_view(h * model_config.head_dim, (h + 1) * model_config.head_dim)
            v_h = v.col_view(h * model_config.head_dim, (h + 1) * model_config.head_dim)

            score = mat_mul(q_h, k_h.T())
            scaled = scale(score, 1.0 / math.sqrt(model_config.head_dim))
            attn_out_head = attn_out.col_view(
                h * model_config.head_dim, (h + 1) * model_config.head_dim
            )
            part_results.append(
                mat_mul(softmax(casual_mask(scaled)), v_h, attn_out_head)
            )
        attn_out.parents = part_results

        x = add(x_resid, mat_mul(attn_out, layer.wo))

        # forward
        x_resid = x
        x = rms_norm(x)
        x = mat_mul(x, layer.ff1)
        x = relu(x)
        x = mat_mul(x, layer.ff2)
        x = add(x_resid, x)

    # return logits
    return mat_mul(rms_norm(x), model.lm_head)

def train(file, config: Config):
    with open(file) as f:
        content = f.read()
    config.vocalbulary = list(sorted(set(content)))
    config.vocab_size = len(config.vocalbulary)

    model = Model(config=config)

    # collect learnable parameters
    all_params: list[Tensor] = []
    all_params.extend([model.embeddings, model.pos_embeddings, model.lm_head])
    for i in range(config.hidden_layers):
        all_params.extend(
            [
                model.layers[i].ff1,
                model.layers[i].ff2,
                model.layers[i].k,
                model.layers[i].q,
                model.layers[i].v,
                model.layers[i].wo,
            ]
        )

    for i in range(config.learning_steps):
        perf_start("train")

        # select a random context window for learning
        start_offset = random.randint(
            0, len(content) - config.learning_context_window - 1
        )
        sequence = content[start_offset : start_offset + config.learning_context_window]
        targets = model.encode(
            content[
                start_offset + 1 : start_offset + 1 + config.learning_context_window
            ]
        )

        x = add(embed(model.embeddings, model.encode(sequence)), model.pos_embeddings)
        logits = gpt(model, x)
        loss = cross_entropy(logits, targets)
        backpropagation(loss)
        learn_simple(all_params, config.lr)

        elapsed = perf_end("train", quiet=True)
        print(f"step: {i}, l: {loss.data[0]}, took {elapsed} s")

        if i % config.export_steps == 0:
            model.export(f"cp{i}.bin")

class InferenceConfig:
    def __init__(self):
        self.temperature = 0.5
        self.max_tokens = 100
        self.ctx_window = ""

def inference(file, inference_config: InferenceConfig):
    model = Model.load(file)
    model_config = model.config

    context_window = list(inference_config.ctx_window)
    for i in range(inference_config.max_tokens):
        if len(context_window) > model_config.vocab_size:
            context_window = context_window[-model_config.vocab_size:]

        x = add(
            embed(model.embeddings, model.encode(context_window)), model.pos_embeddings
        )
        logits = gpt(model, x)

        last_row = []
        for j in range(logits.cols):
            last_row.append(
                logits.data[
                    logits.offset
                    + ((logits.rows - 1) * logits.strides[0])
                    + (j * logits.strides[1])
                ]
            )
        # now pick the next token...
        temp = inference_config.temperature
        scaled = [z / temp for z in last_row]
        m = max(scaled)
        exps = [math.exp(s - m) for s in scaled]
        total = sum(exps)
        probs = [e / total for e in exps]
        r = random.random()
        cumulative = 0.0
        next_id = len(probs) - 1
        for index, p in enumerate(probs):
            cumulative += p
            if cumulative > r:
                next_id = index
                break
        char = model.decode_single(next_id)
        context_window.append(char)
        print(char, end="", flush=True)


def help():
    c = Config()
    ic = InferenceConfig()
    print(f"""Usage:
  python3 gpt.py --train <data_file> [options]
  python3 gpt.py --chat <model_file> [options]

Train options:
  --steps <n>           number of training steps        (default {c.learning_steps})
  --embedding_len <n>   embedding size per token        (default {c.embedding_len})
  --heads <n>           attention heads per layer       (default {c.heads})
  --hidden_layers <n>   number of transformer layers    (default {c.hidden_layers})

Chat options:
  --prompt <text>       text to start generating from   (default {ic.ctx_window!r})
  --max_tokens <n>      characters to generate          (default {ic.max_tokens})
  --temp <x>            sampling temperature            (default {ic.temperature})

Examples:
  python3 gpt.py --train data/names.txt --steps 3000 --embedding_len 64
  python3 gpt.py --chat test.bin --prompt "Ka" --temp 0.8""")

def get_param(argv, param, default):
    for i, e in enumerate(argv):
        if e == param:
            return argv[i + 1]
    return default

def main():
    if len(sys.argv) == 1:
        help()
        return

    if sys.argv[1] == "--train":
        config = Config()
        config.learning_steps = int(get_param(sys.argv, "--steps", config.learning_steps))
        config.embedding_len = int(get_param(sys.argv, "--embedding_len", config.embedding_len))
        config.heads = int(get_param(sys.argv, "--heads", config.heads))
        config.hidden_layers = int(get_param(sys.argv, "--hidden_layers", config.hidden_layers))
        config.export_steps = int(get_param(sys.argv, "--export_steps", config.export_steps))
        config.head_dim = config.embedding_len // config.heads
        config.lr = float(get_param(sys.argv, "--lr", config.lr))
        train(sys.argv[2], config)
    elif sys.argv[1] == "--chat":
        config = InferenceConfig()
        config.ctx_window = str(get_param(sys.argv, "--prompt", config.ctx_window))
        config.max_tokens = int(get_param(sys.argv, "--max_tokens", config.max_tokens))
        config.temperature = float(get_param(sys.argv, "--temp", config.temperature))
        inference(sys.argv[2], config)
    else:
        help()

if __name__ == "__main__":
    main()

