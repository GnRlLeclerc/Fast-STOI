//! STFT computation on frames

use std::sync::Arc;

use lazy_static::lazy_static;
use ndarray::{Zip, prelude::*};
use realfft::{RealFftPlanner, RealToComplex};

use crate::constants::{FFT_BINS, FFT_LENGTH};

lazy_static! {
    static ref R2C: Arc<dyn RealToComplex<f64>> =
        RealFftPlanner::<f64>::new().plan_fft_forward(FFT_LENGTH);
}

/// Compute the RFFT of each valid frame as indicated by the mask.
/// Returns a real valued spectrogram of shape (frames, FFT_BINS).
pub fn compute_frame_rffts(
    frames: ArrayView2<'_, f64>,
    mask: ArrayView1<'_, bool>,
    count: usize,
) -> Array2<f64> {
    // Create buffers
    let mut scratch_buffer = R2C.make_scratch_vec();
    let mut input_buffer = R2C.make_input_vec();
    let mut output_buffer = R2C.make_output_vec();

    // Create output array as row-major for faster writes
    let mut spectrogram = Array2::<f64>::zeros((count, FFT_BINS));
    let mut index = 0; // destination row index

    // Iterate over valid frames and compute their RFFT
    Zip::from(frames.rows())
        .and(mask)
        .for_each(|frame, &valid| {
            if !valid {
                return;
            }

            // Copy frame into input buffer with zero padding
            input_buffer[..frame.len()].copy_from_slice(frame.as_slice().unwrap());

            // Perform RFFT
            R2C.process_with_scratch(&mut input_buffer, &mut output_buffer, &mut scratch_buffer)
                .unwrap();

            // Copy magnitude spectrum to output spectrogram
            Zip::from(spectrogram.row_mut(index))
                .and(&output_buffer)
                .for_each(|real, &complex| {
                    *real = complex.norm();
                });

            index += 1;
        });

    spectrogram
}
