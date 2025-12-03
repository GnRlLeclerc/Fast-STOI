//! Slice, filter and preprocess audio frames.

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use windowfunctions::{Symmetry, WindowFunction, window};

/// Compute the L2 norm of a frame.
fn norm_l2(frame: ArrayView1<'_, f64>) -> f64 {
    frame.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Slice 2 input signals into overlapping frames and
/// applies a hann window to each frame.
/// The frames are then filtered based on their energy.
///
/// Returns 2D arrays containing the frames along with
/// a boolean mask and the total amount of valid frames.
///
/// Performance notes:
/// Energy-based filtering is performed once all energies have been computed.
/// For this reason, we cannot know beforehand which frames are to be discarded,
/// hence why we store all frames in an intermediate 2D array.
/// In order to avoid reallocations, we return the unfiltered 2D array along
/// with a boolean mask indicating which frames to keep.
pub fn process_frames(
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    dynamic_range: f64,
    frame_length: usize,
    hop_length: usize,
) -> (Array2<f64>, Array2<f64>, Array1<bool>, usize) {
    // 1. Prepare Hann window
    let hann = window(frame_length + 2, WindowFunction::Hann, Symmetry::Symmetric)
        .collect::<Array1<f64>>();
    let trimmed = hann.slice(s![1..frame_length + 1]);

    // 2. Compute frames and energies
    let n = 1 + (x.len() - frame_length - 1) / hop_length;
    let mut x_frames = Array2::<f64>::zeros((n, frame_length));
    let mut y_frames = Array2::<f64>::zeros((n, frame_length));
    let mut energies = Array1::<f64>::zeros(n);

    for (i, start) in (0..x.len() - frame_length).step_by(hop_length).enumerate() {
        // Compute the energy for the current x frame
        let end = start + frame_length;

        let mut x_frame = x_frames.row_mut(i);
        let mut y_frame = y_frames.row_mut(i);

        // Copy frames
        x_frame.assign(&x.slice(s![start..end]));
        y_frame.assign(&y.slice(s![start..end]));

        // Apply hann window
        x_frame *= &trimmed;
        y_frame *= &trimmed;

        // Compute frame energy
        energies[i] = 20.0 * (norm_l2(x_frame.view()) + f64::EPSILON).log10();
    }

    // 3. Compute frame mask based on energies
    let threshold = energies.max_skipnan() - dynamic_range;
    let mut count = 0;
    let mask = energies.mapv(|e| {
        let valid = e >= threshold;
        count += valid as usize;
        valid
    });

    (x_frames, y_frames, mask, count)
}
