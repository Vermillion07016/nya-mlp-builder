//! Aktivasyon fonksiyonları ve bunların serileştirilebilir temsili.

use crate::error::{ModelError, ModelResult};

/// Bir katmanın aktivasyon fonksiyonunu tanımlayan trait.
///
/// Kendi aktivasyon fonksiyonunu eklemek için bu trait'i implemente et.
/// Not: save/load ile serileştirmek istiyorsan [`ActivationKind`]'a da
/// karşılık gelen bir varyant eklemen gerekir.
pub trait Activation: Send + Sync {
    /// Ham (logit) değeri alıp aktivasyon çıktısını hesaplar.
    fn forward(&self, x: f32) -> f32;

    /// Aktivasyonun türevini hesaplar. `x` ham (logit) değerdir,
    /// aktivasyon çıktısı değil.
    fn derivative(&self, x: f32) -> f32;

    /// Bu aktivasyonun serileştirilebilir türünü döndürür.
    /// save/load sırasında hangi aktivasyonun kullanıldığını saklamak için kullanılır.
    fn kind(&self) -> ActivationKind;
}

/// Bir aktivasyon fonksiyonunun serileştirilebilir (disk'e yazılabilir) temsili.
///
/// [`Activation`] trait objesi (`Box<dyn Activation>`) doğrudan diske
/// yazılamayacağı için, bunun yerine hangi aktivasyonun (ve varsa
/// parametresinin) kullanıldığını bu enum ile saklıyoruz.
#[derive(Clone, Copy, Debug)]
pub enum ActivationKind {
    Sigmoid,
    ReLU,
    /// `LeakyReLU(alpha)` — negatif girişlerin `alpha` ile çarpılacağı eğim.
    LeakyReLU(f32),
    Tanh,
    Linear
}

impl ActivationKind {
    /// Bu türe karşılık gelen çalışan bir `Box<dyn Activation>` üretir.
    pub fn to_boxed(self) -> Box<dyn Activation> {
        match self {
            ActivationKind::Sigmoid => Box::new(Sigmoid),
            ActivationKind::ReLU => Box::new(ReLU),
            ActivationKind::LeakyReLU(alpha) => Box::new(LeakyReLU { alpha }),
            ActivationKind::Tanh => Box::new(Tanh),
            ActivationKind::Linear => Box::new(Linear)
        }
    }

    /// Bu türü diske yazılabilecek tek byte'lık bir kimliğe çevirir.
    pub fn to_id(self) -> u8 {
        match self {
            ActivationKind::Sigmoid => 0,
            ActivationKind::ReLU => 1,
            ActivationKind::LeakyReLU(_) => 2,
            ActivationKind::Tanh => 3,
            ActivationKind::Linear => 4
        }
    }

    /// Diskten okunan bir id ve (varsa) alpha parametresinden `ActivationKind` üretir.
    ///
    /// # Hatalar
    /// `id` bilinen bir aktivasyona karşılık gelmiyorsa
    /// [`ModelError::UnknownActivationId`] döner.
    pub fn from_id(id: u8, alpha: f32) -> ModelResult<Self> {
        match id {
            0 => Ok(ActivationKind::Sigmoid),
            1 => Ok(ActivationKind::ReLU),
            2 => Ok(ActivationKind::LeakyReLU(alpha)),
            3 => Ok(ActivationKind::Tanh),
            4 => Ok(ActivationKind::Linear),
            _ => Err(ModelError::UnknownActivationId(id)),
        }
    }
}

/// Standart lojistik (sigmoid) aktivasyon fonksiyonu: `f(x) = 1 / (1 + e^-x)`.
///
/// Çıktısı `(0, 1)` aralığındadır, genellikle çıkış katmanında
/// (ikili sınıflandırma) kullanılır.
pub struct Sigmoid;
impl Activation for Sigmoid {
    fn forward(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn derivative(&self, x: f32) -> f32 {
        let a = self.forward(x);
        a * (1.0 - a)
    }
    fn kind(&self) -> ActivationKind {
        ActivationKind::Sigmoid
    }
}

pub struct Tanh;
impl Activation for Tanh {
    fn forward(&self, x: f32) -> f32 {
        x.tanh()
    }
    fn derivative(&self, x: f32) -> f32 {
        let a = self.forward(x);
        1.0 - a.powi(2)
    }
    fn kind(&self) -> ActivationKind {
        ActivationKind::Tanh
    }
}

pub struct Linear;
impl Activation for Linear {
    fn forward(&self, x: f32) -> f32 { x }
    fn derivative(&self, _x: f32) -> f32 { 1.0 }
    fn kind(&self) -> ActivationKind { ActivationKind::Linear }
}

/// Sızıntılı ReLU: pozitif girişleri olduğu gibi bırakır, negatifleri
/// `alpha` ile çarpar. "Ölü nöron" problemini standart ReLU'ya göre azaltır.
pub struct LeakyReLU {
    pub alpha: f32,
}
impl Activation for LeakyReLU {
    fn forward(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { self.alpha * x }
    }
    fn derivative(&self, x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { self.alpha }
    }
    fn kind(&self) -> ActivationKind {
        ActivationKind::LeakyReLU(self.alpha)
    }
}

/// Standart ReLU (Rectified Linear Unit): `f(x) = max(0, x)`.
#[allow(dead_code)]
pub struct ReLU;
impl Activation for ReLU {
    fn forward(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }
    fn derivative(&self, x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
    fn kind(&self) -> ActivationKind {
        ActivationKind::ReLU
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_forward_at_zero_is_half() {
        let s = Sigmoid;
        assert!((s.forward(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_derivative_matches_formula() {
        let s = Sigmoid;
        let x = 1.5;
        let a = s.forward(x);
        let expected = a * (1.0 - a);
        assert!((s.derivative(x) - expected).abs() < 1e-6);
    }

    #[test]
    fn relu_forward_clamps_negative_to_zero() {
        let r = ReLU;
        assert_eq!(r.forward(-3.0), 0.0);
        assert_eq!(r.forward(3.0), 3.0);
    }

    #[test]
    fn relu_derivative_is_step_function() {
        let r = ReLU;
        assert_eq!(r.derivative(1.0), 1.0);
        assert_eq!(r.derivative(-1.0), 0.0);
    }

    #[test]
    fn leaky_relu_forward_scales_negative_by_alpha() {
        let l = LeakyReLU { alpha: 0.1 };
        assert_eq!(l.forward(-2.0), -0.2);
        assert_eq!(l.forward(2.0), 2.0);
    }

    #[test]
    fn activation_kind_roundtrip_sigmoid() {
        let kind = ActivationKind::Sigmoid;
        let id = kind.to_id();
        let restored = ActivationKind::from_id(id, 0.0).unwrap();
        assert!(matches!(restored, ActivationKind::Sigmoid));
    }

    #[test]
    fn activation_kind_roundtrip_leaky_relu_preserves_alpha() {
        let kind = ActivationKind::LeakyReLU(0.05);
        let id = kind.to_id();
        let restored = ActivationKind::from_id(id, 0.05).unwrap();
        match restored {
            ActivationKind::LeakyReLU(alpha) => assert!((alpha - 0.05).abs() < 1e-6),
            _ => panic!("LeakyReLU bekleniyordu"),
        }
    }

    #[test]
    fn activation_kind_from_unknown_id_errors() {
        let result = ActivationKind::from_id(99, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn kind_to_boxed_produces_matching_activation() {
        // to_boxed() ile üretilen Box<dyn Activation>'ın forward davranışı
        // orijinal struct ile aynı olmalı
        let boxed = ActivationKind::ReLU.to_boxed();
        assert_eq!(boxed.forward(-5.0), 0.0);
        assert_eq!(boxed.forward(5.0), 5.0);
    }
}