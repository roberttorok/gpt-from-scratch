# GPT from sratch in python with custom kernels

A small character-level GPT I wrote to learn how these models actually work.

No PyTorch, no NumPy, no dependencies at all. Just Python's standard library
and one C file. Python drives everything (model, autograd, training loop),
and the heavy math runs in hand-written C kernels called through `ctypes`.

## What's in here

- `kernel.c` - the C kernels: matmul, softmax, causal mask, RMSNorm, ReLU,
  cross entropy, embeddings, and the backward pass for each one
- `tensor.py` - a tiny tensor (flat float buffer + strides) and a wrapper per op
- `gpt.py` - the model, backprop, training and text generation
- `profile.py` - a small timing helper

## Running it

Build the kernel (needs gcc, Linux):

    ./build.sh

Get some training data (tiny Shakespeare, ~1 MB):

    mkdir -p data
    wget -O data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Train on it:

    python3 gpt.py --train data/shakespeare.txt --steps 3000 --embedding_len 64

Checkpoints are saved to `exports/`. Generate text from one:

    python3 gpt.py --chat exports/cp2000.bin --prompt "ROMEO:"

Run `python3 gpt.py` with no arguments to see all options.

## Data

No training data is included in the repo. Any plain text file works, the
vocabulary is just the set of characters in the file.

I tested it on four small (~1-2 MB) datasets: fantasy names, DOOM 3 source code,
English Wikipedia articles about cities, and tiny Shakespeare.

After a few thousand steps on the names it comes up with stuff like
"Sorakolokae", "Riolvatora" and "Athaminokus".

## Why this and not microgpt-c?

I also have [microgpt-c](https://github.com/roberttorok/microgpt-c), which
is the same idea in pure C. So why do it again?

- Python is the language of AI, so I wanted to do it there too.
- The GPT code is much easier to read in Python. In C the model gets
  buried under memory management and loops.
- I wanted to learn how Python calls into C. Here it's done with plain `ctypes`,
  no build system, no wrappers.
- I wanted to learn how kernels work. The split here is the same one the
  big frameworks use: Python describes the model, and small C functions
  do the number crunching.
