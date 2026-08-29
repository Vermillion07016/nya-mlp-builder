//! Tek bir sinir ağı katmanı: ağırlıklar, bias'lar ve momentum durumu.

use crate::activations::Activation;
use nalgebra::DMatrix;

/// Tam bağlantılı (dense) bir sinir ağı katmanı.
///
/// Ağırlık matrisi `(out_size, in_size)` şeklindedir, böylece
/// `weights * input` çarpımı `(out_size, batch_size)` şeklinde bir çıktı üretir.
pub struct Layer {
    pub weights: DMatrix<f32>,
    pub biases: DMatrix<f32>,
    pub activation: Box<dyn Activation>,

    vel_weights: DMatrix<f32>,
    vel_biases: DMatrix<f32>,
}

impl Layer {
    /// Verilen boyutlarda, He başlatması (`sqrt(2/in_size)` ölçekli rastgele
    /// ağırlıklar, sıfır bias) ile yeni bir katman oluşturur.
    pub fn new(in_size: usize, out_size: usize, activation: Box<dyn Activation>) -> Self {
        let mut rng = fastrand::Rng::new();
        let scale = (2.0 / in_size as f32).sqrt();

        let weights = DMatrix::from_fn(out_size, in_size, |_, _| (rng.f32() * 2.0 - 1.0) * scale);
        let biases = DMatrix::zeros(out_size, 1);

        Self {
            weights,biases,activation,
            vel_weights: DMatrix::zeros(out_size, in_size),
            vel_biases: DMatrix::zeros(out_size, 1),
        }
    }
    /// Bu katmanın beklediği giriş boyutu (`weights` sütun sayısı).
    pub fn in_size(&self) -> usize {
        self.weights.ncols()
    }
    /// Bu katmanın ürettiği çıkış boyutu (`weights` satır sayısı).
    pub fn out_size(&self) -> usize {
        self.weights.nrows()
    }
    /// Batch halinde ileri geçiş yapar.
    ///
    /// Dönüş değeri `(raw, activated)` tuple'ıdır: `raw` aktivasyon
    /// öncesi (logit) değerler, `activated` ise aktivasyon sonrası çıktıdır.
    /// `raw` geri yayılım (backprop) sırasında türev hesaplamak için gereklidir.
    pub fn forward_batch(&self, inputs: &DMatrix<f32>) -> (DMatrix<f32>, DMatrix<f32>) {
        let mut raw = &self.weights * inputs;
        for mut col in raw.column_iter_mut() {
            col += &self.biases;
        }
        let activated = raw.map(|x| self.activation.forward(x));
        (raw, activated)
    }
    /// Momentum'lu SGD ile ağırlık ve bias'ları günceller.
    pub fn update_parameters(&mut self,grad_w: &DMatrix<f32>,grad_b: &DMatrix<f32>,lr: f32,beta: f32) {
        self.vel_weights = &self.vel_weights * beta + grad_w * lr;
        self.weights -= &self.vel_weights;
        self.vel_biases = &self.vel_biases * beta + grad_b * lr;
        self.biases -= &self.vel_biases;
    }
    /// Kaydedilmiş ağırlık/bias verilerinden bir katman oluşturur.
    /// Momentum (velocity) matrisleri sıfırdan başlatılır — save/load
    /// sırasında momentum durumu saklanmaz.
    pub fn from_parts(weights: DMatrix<f32>,biases: DMatrix<f32>,activation: Box<dyn Activation>) -> Self {
        let out_size = weights.nrows();
        let in_size = weights.ncols();
        Self {
            weights, biases, activation,
            vel_weights: DMatrix::zeros(out_size, in_size),
            vel_biases: DMatrix::zeros(out_size, 1),
        }
    }
}
impl Clone for Layer {
    /// Katmanı klonlar. Aktivasyon, `kind()` → `to_boxed()` üzerinden
    /// yeniden oluşturulur (trait objesi doğrudan klonlanamadığı için).
    fn clone(&self) -> Self {
        Layer {
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            activation: self.activation.kind().to_boxed(),
            vel_weights: self.vel_weights.clone(),
            vel_biases: self.vel_biases.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::ReLU;

    #[test]
    fn new_layer_has_correct_shapes() {
        let layer = Layer::new(3, 5, Box::new(ReLU));
        assert_eq!(layer.in_size(), 3);
        assert_eq!(layer.out_size(), 5);
        assert_eq!(layer.weights.nrows(), 5);
        assert_eq!(layer.weights.ncols(), 3);
        assert_eq!(layer.biases.nrows(), 5);
        assert_eq!(layer.biases.ncols(), 1);
    }

    #[test]
    fn new_layer_biases_start_at_zero() {
        let layer = Layer::new(4, 2, Box::new(ReLU));
        assert!(layer.biases.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn forward_batch_output_shape_matches_out_size_and_batch() {
        let layer = Layer::new(2, 3, Box::new(ReLU));
        let input = DMatrix::from_column_slice(2, 4, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ]);
        let (raw, activated) = layer.forward_batch(&input);
        assert_eq!(raw.nrows(), 3);
        assert_eq!(raw.ncols(), 4);
        assert_eq!(activated.nrows(), 3);
        assert_eq!(activated.ncols(), 4);
    }

    #[test]
    fn from_parts_preserves_weights_and_biases() {
        let weights = DMatrix::from_column_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let biases = DMatrix::from_column_slice(2, 1, &[0.5, 0.5]);
        let layer = Layer::from_parts(weights.clone(), biases.clone(), Box::new(ReLU));

        assert_eq!(layer.weights, weights);
        assert_eq!(layer.biases, biases);
        assert_eq!(layer.in_size(), 2);
        assert_eq!(layer.out_size(), 2);
    }

    #[test]
    fn update_parameters_changes_weights_and_biases() {
        let mut layer = Layer::new(2, 2, Box::new(ReLU));
        let original_weights = layer.weights.clone();

        let grad_w = DMatrix::from_element(2, 2, 0.1);
        let grad_b = DMatrix::from_element(2, 1, 0.1);
        layer.update_parameters(&grad_w, &grad_b, 0.1, 0.9);

        assert_ne!(layer.weights, original_weights);
    }
}