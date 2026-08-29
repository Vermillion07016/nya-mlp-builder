//! Model tanımı, eğitim döngüsü, tahmin ve kaydetme/yükleme.
pub use layer::Layer;
use nalgebra::DMatrix;
use crate::error::{ModelError, ModelResult};

mod layer;
mod model_builder;

pub use model_builder::ModelBuilder;

/// Eğitilmiş bir sinir ağı: sıralı bir katman listesi.
///
/// Doğrudan oluşturulmaz; [`ModelBuilder`] ile inşa edilir.
#[derive(Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    /// Girişi tüm katmanlardan sırayla geçirip modelin çıktısını üretir.
    ///
    /// `input` şekli `(input_size, batch_size)` olmalıdır.
    pub fn forward(&self, input: &DMatrix<f32>) -> DMatrix<f32> {
        let mut curr = input.clone();
        for layer in &self.layers {
            let (_, act) = layer.forward_batch(&curr);
            curr = act;
        }
        curr
    }
    /// Tek bir batch üzerinde ileri + geri yayılım yapıp ağırlıkları günceller.
    ///
    /// GEMM (matris çarpımı) tabanlı, tüm batch'i tek seferde işler.
    /// Dönüş değeri, bu batch için ortalama kare hata (MSE) kaybıdır.
    pub fn train_batch(&mut self,inputs: &DMatrix<f32>,targets: &DMatrix<f32>,lr: f32,beta: f32) -> f32 {
        let batch_size = inputs.ncols();
        if batch_size == 0 { return 0.0; }

        let scale = 1.0 / batch_size as f32;
        let num_layers = self.layers.len();

        let mut layer_inputs = Vec::with_capacity(num_layers);
        let mut layer_raws = Vec::with_capacity(num_layers);
        let mut curr = inputs.clone();

        for layer in &self.layers {
            layer_inputs.push(curr.clone());
            let (raw, act) = layer.forward_batch(&curr);
            layer_raws.push(raw);
            curr = act;
        }

        let output = curr;
        let diff = &output - targets;
        let loss = diff.map(|x| x.powi(2)).sum() * scale;

        let mut deltas = diff;

        for i in (0..num_layers).rev() {
            let raws = &layer_raws[i];
            let ins = &layer_inputs[i];
            let out_size = self.layers[i].out_size();

            let act_deriv = raws.map(|x| self.layers[i].activation.derivative(x));
            let d_z = deltas.component_mul(&act_deriv);

            let grad_w = (&d_z * ins.transpose()) * scale;
            let col_sum = d_z.column_sum();
            let grad_b = DMatrix::from_column_slice(out_size, 1, col_sum.as_slice()) * scale;

            if i > 0 {
                deltas = self.layers[i].weights.transpose() * &d_z;
            }

            self.layers[i].update_parameters(&grad_w, &grad_b, lr, beta);
        }

        loss
    }
    /// Modelin beklediği giriş boyutu (ilk katmanın `in_size`'ı).
    /// Katman yoksa 0 döner.
    pub fn input_size(&self) -> usize {
        self.layers.first().map(|l| l.in_size()).unwrap_or(0)
    }
    /// Modelin ürettiği çıkış boyutu (son katmanın `out_size`'ı).
    /// Katman yoksa 0 döner.
    pub fn output_size(&self) -> usize {
        self.layers.last().map(|l| l.out_size()).unwrap_or(0)
    }
    /// Tek bir örnek için tahmin yapar.
    ///
    /// `input.len()` [`Model::input_size`]'a eşit olmalıdır; dönüşüm
    /// otomatik yapılır, elle `DMatrix` oluşturmaya gerek yoktur.
    ///
    /// # Hatalar
    /// Boyut uyuşmazlığında [`ModelError::InputShapeMismatch`] döner.
    pub fn predict(&self, input: &[f32]) -> ModelResult<Vec<f32>> {
        let expected = self.input_size();
        if input.len() != expected {
            return Err(ModelError::InputShapeMismatch {
                expected,
                actual: input.len(),
            });
        }

        let input_matrix = DMatrix::from_column_slice(expected, 1, input);
        let output = self.forward(&input_matrix);
        Ok(output.column(0).iter().copied().collect())
    }
    pub fn predict_batch(&self, input: &[f32], sample_count: usize) -> ModelResult<Vec<Vec<f32>>> {
        let in_size = self.layers[0].weights.ncols();
        let input_matrix = DMatrix::from_column_slice(in_size, sample_count, input);
        let output = self.forward(&input_matrix);
        Ok((0..sample_count).map(|i| output.column(i).iter().copied().collect()).collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{LeakyReLU, ReLU, Sigmoid};
    use crate::error::ModelError;
use crate::model::model_builder::ModelBuilder;

    pub fn xor_builder() -> ModelBuilder {
        ModelBuilder::default()
            .add_layer(2, 4, Box::new(LeakyReLU { alpha: 0.01 }))
            .unwrap()
            .add_layer(4, 1, Box::new(Sigmoid))
            .unwrap()
    }
    #[test]
    fn add_layer_shape_mismatch_errors() {
        let result = ModelBuilder::default()
            .add_layer(2, 4, Box::new(ReLU))
            .unwrap()
            .add_layer(5, 1, Box::new(Sigmoid)); // 4 bekleniyordu, 5 verildi

        assert!(matches!(result, Err(ModelError::LayerShapeMismatch { expected: 4, actual: 5 })));
    }

    #[test]
    fn build_produces_model_with_correct_layer_count() {
        let model = xor_builder().build();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.input_size(), 2);
        assert_eq!(model.output_size(), 1);
    }

    #[test]
    fn forward_output_shape_matches_batch_size() {
        let model = xor_builder().build();
        let input = DMatrix::from_column_slice(2, 3, &[
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let output = model.forward(&input);
        assert_eq!(output.nrows(), 1);
        assert_eq!(output.ncols(), 3);
    }

    #[test]
    fn predict_returns_correct_length_and_matches_forward() {
        let model = xor_builder().build();
        let result = model.predict(&[0.5, -0.3]).unwrap();
        assert_eq!(result.len(), 1);

        let input_matrix = DMatrix::from_column_slice(2, 1, &[0.5, -0.3]);
        let expected = model.forward(&input_matrix)[(0, 0)];
        assert!((result[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn predict_wrong_input_size_errors() {
        let model = xor_builder().build();
        let result = model.predict(&[1.0, 2.0, 3.0]); // 2 bekleniyordu, 3 verildi
        assert!(matches!(
            result,
            Err(ModelError::InputShapeMismatch { expected: 2, actual: 3 })
        ));
    }
    #[test]
    fn train_batch_gemm_reduces_loss_on_xor() {
        let mut model = xor_builder().build();

        // XOR problemi: girişler ve hedefler
        let inputs = DMatrix::from_column_slice(2, 4, &[
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ]);
        let targets = DMatrix::from_column_slice(1, 4, &[0.0, 1.0, 1.0, 0.0]);

        let initial_loss = model.train_batch(&inputs, &targets, 0.5, 0.9);
        for _ in 0..200 {
            model.train_batch(&inputs, &targets, 0.5, 0.9);
        }
        let final_loss = model.train_batch(&inputs, &targets, 0.5, 0.9);

        assert!(
            final_loss < initial_loss,
            "loss azalmadı: initial={}, final={}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn train_batch_gemm_empty_batch_returns_zero_loss() {
        let mut model = xor_builder().build();
        let inputs = DMatrix::from_column_slice(2, 0, &[]);
        let targets = DMatrix::from_column_slice(1, 0, &[]);
        let loss = model.train_batch(&inputs, &targets, 0.1, 0.9);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn save_and_load_roundtrip_preserves_predictions() {
        let builder = xor_builder();
        let dir = std::env::temp_dir();
        let path = dir.join("deep_learn_test_model_predict.bin");

        builder.save(&path).unwrap();
        let loaded_model = ModelBuilder::load(&path).unwrap().build();

        let input = [0.5, -0.5];
        let result = loaded_model.predict(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_finite());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_invalid_magic_errors() {
        let dir = std::env::temp_dir();
        let path = dir.join("deep_learn_test_invalid_magic.bin");
        std::fs::write(&path, b"XXXXgarbagebytes").unwrap();

        let result = ModelBuilder::load(&path);
        assert!(matches!(result, Err(ModelError::InvalidMagic)));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_nonexistent_file_errors() {
        let result = ModelBuilder::load("/path/that/does/not/exist_deep_learn.bin");
        assert!(matches!(result, Err(ModelError::Io(_))));
    }

    #[test]
    fn loaded_builder_can_add_more_layers() {
        let builder = xor_builder();
        let dir = std::env::temp_dir();
        let path = dir.join("deep_learn_test_extend.bin");
        builder.save(&path).unwrap();

        // Yüklenen builder'a yeni bir çıkış katmanı ekleyelim (4 -> ... son katman 1 çıktı veriyordu,
        // o yüzden mevcut son katmanın çıkışına uygun yeni bir katman ekliyoruz)
        let extended = ModelBuilder::load(&path)
            .unwrap()
            .add_layer(1, 1, Box::new(Sigmoid))
            .unwrap();
        let model = extended.build();

        assert_eq!(model.layers.len(), 3);
        std::fs::remove_file(&path).ok();
    }
}