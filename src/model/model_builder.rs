use std::{fs::File, io::{BufReader, BufWriter, Read, Write}, path::Path};
use nalgebra::DMatrix;

use crate::{ModelError, ModelResult, activations::{Activation, ActivationKind}, model::{Layer, Model}};

const MAGIC: &[u8; 4] = b"DLRN";
const FORMAT_VERSION: u32 = 1;

/// [`Model`] inşa etmek için builder. Katmanlar eklendikçe giriş/çıkış
/// boyutlarının uyumu doğrulanır.
#[derive(Default)]
pub struct ModelBuilder {
    pub layers: Vec<Layer>,
    pub last_out_size: Option<usize>,
}

impl ModelBuilder {
    /// Builder'a yeni bir katman ekler.
    ///
    /// # Hatalar
    /// `in_size`, önceki katmanın çıkış boyutuyla eşleşmiyorsa
    /// [`ModelError::LayerShapeMismatch`] döner.
    pub fn add_layer(mut self,in_size: usize,out_size: usize,activation: Box<dyn Activation>) -> ModelResult<Self> {
        if let Some(expected) = self.last_out_size 
        && expected != in_size {
            return Err(ModelError::LayerShapeMismatch {
                expected,
                actual: in_size,
            });
        }
        self.layers.push(Layer::new(in_size, out_size, activation));
        self.last_out_size = Some(out_size);
        Ok(self)
    }
    /// Builder'ı tüketip nihai [`Model`]'i döndürür.
    pub fn build(self) -> Model {
        Model { layers: self.layers }
    }
    /// Builder içindeki katmanları (ağırlıklar, bias'lar, aktivasyon tipleri)
    /// versiyonlu bir binary formatta dosyaya yazar. Momentum durumu saklanmaz.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> ModelResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(MAGIC)?;
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&(self.layers.len() as u32).to_le_bytes())?;

        for layer in &self.layers {
            writer.write_all(&(layer.in_size() as u32).to_le_bytes())?;
            writer.write_all(&(layer.out_size() as u32).to_le_bytes())?;

            let kind = layer.activation.kind();
            writer.write_all(&[kind.to_id()])?;
            let alpha = if let ActivationKind::LeakyReLU(a) = kind { a } else { 0.0 };
            writer.write_all(&alpha.to_le_bytes())?;

            for &w in layer.weights.iter() {
                writer.write_all(&w.to_le_bytes())?;
            }
            for &b in layer.biases.iter() {
                writer.write_all(&b.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }
    /// Daha önce [`ModelBuilder::save`] ile kaydedilmiş bir dosyadan
    /// `ModelBuilder` yükler. Yüklendikten sonra `.add_layer(...)` ile
    /// ek katman eklenebilir ya da doğrudan `.build()` çağrılabilir.
    ///
    /// # Hatalar
    /// - Dosya formatı geçersizse [`ModelError::InvalidMagic`]
    /// - Versiyon desteklenmiyorsa [`ModelError::UnsupportedVersion`]
    /// - Bilinmeyen aktivasyon id'si varsa [`ModelError::UnknownActivationId`]
    /// - I/O hatası varsa [`ModelError::Io`]
    pub fn load<P: AsRef<Path>>(path: P) -> ModelResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(ModelError::InvalidMagic);
        }

        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?;
        let version = u32::from_le_bytes(u32_buf);
        if version != FORMAT_VERSION {
            return Err(ModelError::UnsupportedVersion(version));
        }

        reader.read_exact(&mut u32_buf)?;
        let num_layers = u32::from_le_bytes(u32_buf) as usize;

        let mut layers = Vec::with_capacity(num_layers);
        let mut last_out_size = None;
        let mut f32_buf = [0u8; 4];

        for _ in 0..num_layers {
            reader.read_exact(&mut u32_buf)?;
            let in_size = u32::from_le_bytes(u32_buf) as usize;
            reader.read_exact(&mut u32_buf)?;
            let out_size = u32::from_le_bytes(u32_buf) as usize;

            let mut id_buf = [0u8; 1];
            reader.read_exact(&mut id_buf)?;
            reader.read_exact(&mut f32_buf)?;
            let alpha = f32::from_le_bytes(f32_buf);
            let kind = ActivationKind::from_id(id_buf[0], alpha)?;

            let w_data = read_f32_vec(&mut reader, in_size * out_size)?;
            let weights = slice_to_dmatrix(&w_data, out_size, in_size)?;

            let b_data = read_f32_vec(&mut reader, out_size)?;
            let biases = slice_to_dmatrix(&b_data, out_size, 1)?;

            layers.push(Layer::from_parts(weights, biases, kind.to_boxed()));
            last_out_size = Some(out_size);
        }

        Ok(ModelBuilder { layers, last_out_size })
    }
}

/// Bir `&[f32]` slice'ını verilen (rows, cols) boyutunda bir DMatrix'e çevirir.
/// nalgebra DMatrix column-major sırayla dolduğu için, slice'ın da
/// column-major sırada (yani save fonksiyonunun yazdığı sırayla) olduğu varsayılır.
pub fn slice_to_dmatrix(data: &[f32], rows: usize, cols: usize) -> ModelResult<DMatrix<f32>> {
    let expected = rows * cols;
    if data.len() != expected {
        return Err(
            ModelError::SliceShapeMismatch {expected,rows,cols,actual: data.len()}
        );
    }
    Ok(DMatrix::from_column_slice(rows, cols, data))
}
pub fn read_f32_vec<R: Read>(reader: &mut R, count: usize) -> ModelResult<Vec<f32>> {
    let mut byte_buf = vec![0u8; count * 4];
    reader.read_exact(&mut byte_buf)?;
    Ok(byte_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::model::tests::xor_builder;

use super::*;

    #[test]
    fn slice_to_dmatrix_correct_shape_succeeds() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = slice_to_dmatrix(&data, 2, 3).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn slice_to_dmatrix_wrong_shape_errors() {
        let data = [1.0, 2.0, 3.0];
        let result = slice_to_dmatrix(&data, 2, 2);
        assert!(matches!(result, Err(ModelError::SliceShapeMismatch { .. })));
    }

    #[test]
    fn save_and_load_roundtrip_preserves_weights() {
        let builder = xor_builder();
        let dir = std::env::temp_dir();
        let path = dir.join("deep_learn_test_model.bin");

        builder.save(&path).unwrap();
        let loaded = ModelBuilder::load(&path).unwrap();

        assert_eq!(loaded.layers.len(), builder.layers.len());
        for (original, restored) in builder.layers.iter().zip(loaded.layers.iter()) {
            assert_eq!(original.weights, restored.weights);
            assert_eq!(original.biases, restored.biases);
            assert_eq!(original.in_size(), restored.in_size());
            assert_eq!(original.out_size(), restored.out_size());
        }

        std::fs::remove_file(&path).ok();
    }
}