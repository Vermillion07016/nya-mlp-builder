//! # deep_learn
//!
//! `nalgebra` tabanlı, GEMM üzerinden batch eğitim yapan basit bir
//! feed-forward sinir ağı kütüphanesi.
//!
//! Hızlı başlangıç için [`model::ModelBuilder`]'a bakın.

pub mod activations;
pub mod error;
pub mod model;

pub use error::{ModelError, ModelResult};