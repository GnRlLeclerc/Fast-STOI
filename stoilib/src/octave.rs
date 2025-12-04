//! Third octave bands

use ndarray::{Zip, prelude::*};

use crate::constants::NUM_BANDS;

/// Octave band indices in FFT spectrums of length 512
/// (precomputed from the original STOI implementation)
const OCTAVE_BANDS: [(usize, usize); NUM_BANDS] = [
    (7, 9),
    (9, 11),
    (11, 14),
    (14, 17),
    (17, 22),
    (22, 27),
    (27, 34),
    (34, 43),
    (43, 55),
    (55, 69),
    (69, 87),
    (87, 109),
    (109, 138),
    (138, 174),
    (174, 219),
];

/// Merge FFT spectrogram into octave bands specified by the index ranges in `OCTAVE_BANDS`.
pub fn compute_octave_bands(spectrogram: ArrayView2<'_, f64>) -> Array2<f64> {
    let num_frames = spectrogram.shape()[0];
    let mut band_spectrogram = Array2::<f64>::zeros((num_frames, NUM_BANDS));

    // Iterate over each frame
    Zip::from(spectrogram.rows())
        .and(band_spectrogram.rows_mut())
        .for_each(|rfft, mut bands| {
            // Iterate over each band and compute its energy
            for (band, &(start, end)) in bands.iter_mut().zip(OCTAVE_BANDS.iter()) {
                let sum_squares: f64 = rfft.slice(s![start..end]).iter().map(|&v| v * v).sum();
                *band = sum_squares.sqrt();
            }
        });

    band_spectrogram
}
