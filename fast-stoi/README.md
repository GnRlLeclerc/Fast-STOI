# Fast STOI

TODO : un truc personnalisé pour rust.
refer to the root readme for benchmark results and optimizations

-> tout copier coller, rajouter un guide pour l'utilisation en rust.

## Installation

```bash
cargo add fast-stoi
```

## Usage

Compute STOI from `f32` slices:

```rust
let x = vec![0.0; 24_000];
let y = vec![0.0; 24_000];

let stoi = fast_stoi::stoi(&x, &y, 8_000, false).unwrap();

```
