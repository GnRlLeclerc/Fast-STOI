//! Sinc poly resampling

use std::f64::consts::PI;

use ndarray::prelude::*;
use num::integer;
use windowfunctions::{Symmetry, WindowFunction, window};

const REJECTION_DB: f64 = 60.0;

/// Generate an ideal sinc low-pass filter with normalized cutoff frequency f.
/// Returns an iterator over the filter coefficients to avoid allocation.
fn ideal_sinc(f: f64, half_length: usize) -> impl Iterator<Item = f64> {
    (-(half_length as isize)..half_length as isize + 1).map(move |n| {
        if n == 0 {
            2.0 * f
        } else {
            (2.0 * PI * f * n as f64).sin() / (PI * n as f64)
        }
    })
}

/// Generates a Kaiser window with given beta and length.
/// Returns an iterator over the window coefficients to avoid allocation.
fn kaiser(beta: f32, half_length: usize) -> impl Iterator<Item = f64> {
    window(
        2 * half_length + 1,
        WindowFunction::Kaiser { beta },
        Symmetry::Symmetric,
    )
}

/// Generates an apodized Kaiser window collected into an Array1.
fn apodized_kaiser_window(f: f64, beta: f64, half_length: usize) -> Array1<f64> {
    let sinc_iter = ideal_sinc(f, half_length);
    let kaiser_iter = kaiser(beta as f32, half_length);

    Array1::from_iter(
        sinc_iter
            .zip(kaiser_iter)
            .map(|(sinc, kaiser)| sinc * kaiser),
    )
}

/// Polyphase resampling.
///
/// Some information for this doc (reformulate and clean this up later):
/// - zero-phase => the window is symmetric (does not introduce any shift)
/// - FIR filter => finite impulse response. Basically, the window is of finite length.
/// - low-pass => when upsampling by inserting zeros, if we upsample *n, we create
///   high frequency signals. The window must smooth this out and remove these high frequencies
pub fn resample(x: ArrayView1<'_, f64>, from: u32, to: u32) -> Array1<f64> {
    // Compute upsampling and dowsampling ratios
    let gcd = integer::gcd(from, to);
    let up = to / gcd;
    let down = from / gcd;

    let stopband_cutoff_freq = 1.0 / (2.0 * up.max(down) as f64);
    let roll_off_width = stopband_cutoff_freq / 10.0;

    // Compute the filter
    let filter_half_length = ((REJECTION_DB - 8.0) / (28.714 * roll_off_width)).ceil() as u32;
    let beta = 0.1102 * (REJECTION_DB - 8.7);
    let mut filter =
        apodized_kaiser_window(stopband_cutoff_freq, beta, filter_half_length as usize);
    filter /= filter.sum();

    // Create target array
    let target_len = x.len() as u32 * up / down;
    let mut target = Array1::<f64>::zeros(target_len as usize);

    // DEBUG: implementing the naive method to see if I'm correct
    let mut upsampled =
        Array1::<f64>::zeros(x.len() * up as usize + 2 * filter_half_length as usize);
    // fill the upsampled array
    for (i, &val) in x.iter().enumerate() {
        upsampled[i * up as usize + filter_half_length as usize] = val;
    }

    for (i, y) in target.iter_mut().enumerate() {
        let upsampled_index = i * down as usize + filter_half_length as usize;
        *y = filter.dot(
            &upsampled.slice(s![upsampled_index as isize - filter_half_length as isize
                ..upsampled_index as isize + filter_half_length as isize + 1]),
        ) * up as f64;
    }

    target
}
