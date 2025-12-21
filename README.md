# Fast STOI

Rust fast STOI implementation with python bindings.

Inspired from [pystoi](https://github.com/mpariente/pystoi)
and [torch_stoi](https://github.com/mpariente/pytorch_stoi).

The implementation follows the original `STOI` formula with a maximum
error of `1e-7` on random gaussian noise.
It is much faster than `pystoi`, and even faster than the simplified `torch_stoi`
version (which uses a much lighter resampling).

- `fast-stoi/`: Rust implementation
- `fast-stoi-python/`: python bindings available as the `fast_stoi` package

## Installation

Rust:

```bash
cargo add fast-stoi
```

Python:

```bash
pip install fast_stoi
```

## Usage

Compute STOI from numpy data

```python
import numpy as np
from fast_stoi import stoi

x = np.random.random(24_000).astype(np.float32)
y = np.random.random(24_000).astype(np.float32)

score = stoi(x, y, fs_sig=8_000, extended=False)

```

> [!NOTE]
> You can pass 2D arrays of batched waveforms to leverage
> rust multithreading

Compute STOI with the torch wrapper.

```python
import torch
from fast_stoi import STOI

stoi = STOI(fs_sig=8_000, extended=False)

x = torch.randn(24_000)
y = torch.randn(24_000)

score = stoi(x, y)
```

## Optimizations

- use [`faer`](https://github.com/sarah-quinones/faer-rs) for fast operations and **simd**
- use `f32` internally for even faster vectorization than `f64`
  _(`pystoi` uses the default `np.float64` internally)_
- abuse **cache locality** with `faer`'s column-major storage layout
- limit allocations and copies
- use `rayon` for parallelism at `rust` level _(whose low overhead makes
  it actually work compared to python's `multiprocessing` for this relatively
  fast computation)_

## Benchmarks

Run on a plugged-in Lenovo Yoga Slim 7 Pro X laptop
(AMD Ryzen 7 6800HS Creator Edition cpu).

Parameters:

- 3s audio samples at 8000Hz as f32
- batches of 16 elements

The `torch_stoi` and `pystoi` version are run without parallelism
on the batched benchmarks (the overhead of `multiprocessing` is too high).

`torch_stoi` is run on CPU only.

Standard STOI:

| Implementation | Single |       | Batched  |        |
| -------------- | ------ | ----- | -------- | ------ |
| `fast_stoi`    | 1.5 ms |       | 5.3 ms   |        |
| `torch_stoi`   | 3.2 ms | x 2.1 | 40.0 ms  | x 7.6  |
| `pystoi`       | 9.6 ms | x 6.4 | 144.7 ms | x 27.5 |

Extended STOI:

| Implementation | Single  |        | Batched  |        |
| -------------- | ------- | ------ | -------- | ------ |
| `fast_stoi`    | 1.7 ms  |        | 5.6 ms   |        |
| `torch_stoi`   | 3.5 ms  | x 2.0  | 41.9 ms  | x 7.5  |
| `pystoi`       | 20.7 ms | x 12.2 | 257.1 ms | x 46.2 |

## Develop

```bash
cd fast-stoi-python
uv sync --all-groups
```

Type checking and linting:

```bash
pyrefly check
ruff check
```

Run tests:

```bash
pytest --benchmark-skip
```

Run benchmarks:

```bash
pytest tests/bench/test_speed_standard.py
pytest tests/bench/test_speed_extended.py
```
