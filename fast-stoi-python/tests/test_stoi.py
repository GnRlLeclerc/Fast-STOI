import numpy as np
from pystoi import stoi as theirs

from fast_stoi import stoi as ours

# TODO: test extended STOI


def test_standard():
    np.random.seed(42)
    values = []
    srs = [8_000, 16_000, 32_000]
    seconds = 3
    for sr in srs:
        for _ in range(100):
            x = np.random.randn(sr * seconds)
            y = np.random.randn(sr * seconds)
            values.append(abs(theirs(x, y, fs_sig=sr) - ours(x, y, fs_sig=sr)))

    assert np.array(values).max() < 1e-7
