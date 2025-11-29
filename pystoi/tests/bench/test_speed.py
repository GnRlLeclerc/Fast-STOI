import numpy as np
from pystoi import stoi


def test_pystoi_reference(benchmark):
    sr = 8_000
    seconds = 3
    batch_siwe = 16
    size = batch_siwe * seconds * sr
    x = np.random.random(size).astype(np.float32)
    y = np.random.random(size).astype(np.float32)
    benchmark(stoi, x, y, sr, False)
