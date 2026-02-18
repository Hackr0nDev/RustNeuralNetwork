//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)
//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)
//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)
//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)
//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)

use rand::seq::SliceRandom;
use std::error::Error;
use std::path::Path;

pub type Sample = (Vec<f32>, u8);

/// Загрузить MNIST CSV формата: label,1x1..28x28
/// - пиксели -> f32 и нормализуются в [0;1]
/// - битые строки пропускаются
/// - shuffle опционально
pub fn load_mnist_csv<P: AsRef<Path>>(
    path: P,
    shuffle: bool,
) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)?;

    let mut data: Vec<Sample> = Vec::new();
    let mut skipped = 0usize;

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        // 1 label + 784 pixels
        if record.len() != 785 {
            skipped += 1;
            continue;
        }

        // label
        let y: u8 = match record[0].trim().parse::<u8>() {
            Ok(v) if v <= 9 => v,
            _ => {
                skipped += 1;
                continue;
            }
        };

        // pixels
        let mut x: Vec<f32> = Vec::with_capacity(784);
        let mut ok = true;

        for i in 1..785 {
            let p: u16 = match record[i].trim().parse::<u16>() {
                Ok(v) if v <= 255 => v,
                _ => {
                    ok = false;
                    break;
                }
            };
            x.push((p as f32) / 255.0);
        }

        if ok && x.len() == 784 {
            data.push((x, y));
        } else {
            skipped += 1;
        }
    }

    if shuffle {
        let mut rng = rand::rng(); // rand latest
        data.shuffle(&mut rng);
    }

    eprintln!("[data] loaded={}, skipped={}", data.len(), skipped);
    Ok(data)
}

/// Утилита: перетасовать уже загруженный датасет
pub fn shuffle_in_place(data: &mut [Sample]) {
    let mut rng = rand::rng();
    data.shuffle(&mut rng);
}

/// Утилита: split по доле train (например 0.9)
pub fn split<'a>(data: &'a [Sample], train_frac: f32) -> (&'a [Sample], &'a [Sample]) {
    let frac = train_frac.clamp(0.0, 1.0);
    let split_idx = (data.len() as f32 * frac) as usize;
    data.split_at(split_idx)
}
