use thiserror::Error;

/// `deep_learn` crate'indeki tüm hata durumlarını temsil eder.
#[derive(Error, Debug)]
pub enum ModelError {
    /// `ModelBuilder::add_layer` çağrısında, eklenmek istenen katmanın
    /// giriş boyutu bir önceki katmanın çıkış boyutuyla uyuşmadığında döner.
    #[error("Katman boyut uyuşmazlığı: beklenen giriş boyutu {expected}, verilen {actual}")]
    LayerShapeMismatch { expected: usize, actual: usize },

    /// [`crate::model::Model::predict`] çağrısında verilen giriş uzunluğu
    /// modelin beklediği feature sayısıyla uyuşmadığında döner.
    #[error("Giriş boyutu uyuşmuyor: model {expected} feature bekliyor, {actual} verildi")]
    InputShapeMismatch { expected: usize, actual: usize },

    /// [`crate::model::Model::predict_batch`] çağrısında verilen flat slice'ın
    /// uzunluğu `sample_count * feature_size`'a eşit olmadığında döner.
    #[error(
        "Batch giriş boyutu uyuşmuyor: {sample_count} örnek x {feature_size} feature = {expected} eleman bekleniyordu, {actual} verildi"
    )]
    BatchShapeMismatch {
        sample_count: usize,
        feature_size: usize,
        expected: usize,
        actual: usize,
    },

    /// [`crate::model::slice_to_dmatrix`] çağrısında verilen slice'ın uzunluğu
    /// `rows * cols`'a eşit olmadığında döner.
    #[error("Slice boyutu matrise uymuyor: {expected} eleman bekleniyordu ({rows}x{cols}), {actual} bulundu")]
    SliceShapeMismatch {
        expected: usize,
        rows: usize,
        cols: usize,
        actual: usize,
    },

    /// Yüklenen dosyanın başındaki magic byte'lar beklenenle uyuşmadığında
    /// döner — dosyanın bu kütüphaneyle kaydedilmemiş olabileceğini gösterir.
    #[error("Geçersiz dosya formatı: magic byte uyuşmuyor")]
    InvalidMagic,

    /// Dosyadaki format versiyonu, bu kütüphanenin desteklediği versiyonla
    /// uyuşmadığında döner.
    #[error("Desteklenmeyen format versiyonu: {0}")]
    UnsupportedVersion(u32),

    /// Dosyadan okunan aktivasyon id'si bilinen bir [`crate::activations::ActivationKind`]
    /// değerine karşılık gelmediğinde döner.
    #[error("Bilinmeyen aktivasyon id'si: {0}")]
    UnknownActivationId(u8),

    /// Dosya okuma/yazma sırasında oluşan alt seviye I/O hatalarını sarmalar.
    #[error("I/O hatası: {0}")]
    Io(#[from] std::io::Error),
}

/// `deep_learn` crate'indeki tüm fallible işlemler için kısayol tip.
pub type ModelResult<T> = Result<T, ModelError>;