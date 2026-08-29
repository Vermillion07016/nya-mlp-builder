# deep_learn

Rust ile yazılmış, `nalgebra` tabanlı, GEMM (matris çarpımı) üzerinden batch eğitim yapan basit bir feed-forward sinir ağı kütüphanesi.

## Özellikler

- **Builder pattern** ile katman katman model kurulumu (`ModelBuilder`)
- **Aktivasyon fonksiyonları**: Sigmoid, ReLU, LeakyReLU (trait tabanlı, kolayca genişletilebilir)
- **Batch eğitim**: GEMM tabanlı forward/backward pass, momentum'lu SGD
- **Kaydet/Yükle**: Eğitilmiş modeli versiyonlu bir binary formatta diske yazma ve okuma
- **Tahmin (predict)**: Tek örnek ya da batch halinde, otomatik shape dönüşümü ile
- **Özel hata tipi**: `ModelError` ile anlamlı, panic'siz hata yönetimi
- **Loglama**: `log` crate'i üzerinden eğitim ilerlemesini raporlama (backend'i kullanıcı seçer)
- **Benchmark**: Epoch başına GFLOPS hesaplayan yerleşik eğitim ölçümü

## Kurulum

`Cargo.toml`'a ekle:

```toml
[dependencies]
deep_learn = { path = "../deep_learn" }  # ya da crates.io'daysa: deep_learn = "0.1"
log = "0.4"
env_logger = "0.11"  # loglamayı görmek için (opsiyonel)
```

## Hızlı Başlangıç: XOR Problemi

```rust
use deep_learn::activations::{LeakyReLU, Sigmoid};
use deep_learn::model::ModelBuilder;
use deep_learn::ModelResult;
use nalgebra::DMatrix;

fn main() -> ModelResult<()> {
    env_logger::init();

    // 1. Modeli kur: 2 giriş -> 4 gizli nöron -> 1 çıkış
    let mut model = ModelBuilder::new()
        .add_layer(2, 4, Box::new(LeakyReLU { alpha: 0.01 }))?
        .add_layer(4, 1, Box::new(Sigmoid))?
        .build();

    // 2. XOR veri seti
    let inputs = DMatrix::from_column_slice(2, 4, &[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ]);
    let targets = DMatrix::from_column_slice(1, 4, &[0.0, 1.0, 1.0, 0.0]);

    // 3. Eğit (benchmark ile: loss ve GFLOPS raporlar)
    model.train_with_benchmark(&inputs, &targets, 2000, 0.5, 0.9);

    // 4. Tahmin et
    let result = model.predict(&[1.0, 0.0])?;
    println!("XOR(1, 0) = {:.4}", result[0]);

    Ok(())
}
```

Loglamayı görmek için:

```bash
RUST_LOG=info cargo run
```

## Model Kaydetme ve Yükleme

```rust
use deep_learn::model::ModelBuilder;
use deep_learn::activations::{LeakyReLU, Sigmoid};

# fn example() -> deep_learn::ModelResult<()> {
let builder = ModelBuilder::new()
    .add_layer(2, 4, Box::new(LeakyReLU { alpha: 0.01 }))?
    .add_layer(4, 1, Box::new(Sigmoid))?;

// ... builder.build() ile eğit ...

// Eğitim bitince builder üzerinden kaydet
builder.save("model.bin")?;

// Başka bir çalıştırmada geri yükle
let loaded_model = ModelBuilder::load("model.bin")?.build();
let prediction = loaded_model.predict(&[0.5, 0.5])?;
# Ok(())
# }
```

Yüklenen `ModelBuilder`'a `.add_layer(...)` ile ek katman da eklenebilir (örn. transfer learning veya fine-tuning senaryoları için).

## Tahmin (Predict) API'si

```rust
# fn example(model: deep_learn::model::Model) -> deep_learn::ModelResult<()> {
// Tek bir örnek
let output: Vec<f32> = model.predict(&[0.5, -0.3])?;

// Birden fazla örnek (batch) — flat slice + kaç örnek olduğu
let batch_input = [
    0.5, -0.3,  // örnek 1
    1.0, 1.0,   // örnek 2
];
let outputs: Vec<Vec<f32>> = model.predict_batch(&batch_input, 2)?;
# Ok(())
# }
```

Boyut uyuşmazlığında panic yerine `ModelError::InputShapeMismatch` / `BatchShapeMismatch` döner.

## Hata Yönetimi

Tüm public API'ler `ModelResult<T>` (= `Result<T, ModelError>`) döndürür:

```rust
use deep_learn::ModelError;

match model.predict(&[1.0]) {
    Ok(output) => println!("{:?}", output),
    Err(ModelError::InputShapeMismatch { expected, actual }) => {
        eprintln!("Beklenen {} feature, {} verildi", expected, actual);
    }
    Err(e) => eprintln!("Beklenmeyen hata: {e}"),
}
```

## Kendi Aktivasyon Fonksiyonunu Ekleme

```rust
use deep_learn::activations::Activation;

struct Tanh;
impl Activation for Tanh {
    fn forward(&self, x: f32) -> f32 {
        x.tanh()
    }
    fn derivative(&self, x: f32) -> f32 {
        1.0 - x.tanh().powi(2)
    }
    fn kind(&self) -> deep_learn::activations::ActivationKind {
        // Not: save/load ile serileştirmek istiyorsan ActivationKind enum'una
        // yeni bir varyant eklemen gerekir.
        unimplemented!("Tanh için ActivationKind varyantı ekleyin")
    }
}
```

## Testleri Çalıştırma

```bash
cargo test
```

## Lisans

MIT