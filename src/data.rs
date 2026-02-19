//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)
//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)
//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)
//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)
//ВНИМАНИЕ ЭТО ВСЕ ПИСАЛ CHAT GPT!!! Я НЕ ХОЧУ СЕЙЧАС ВОЗИТСЯ С ПАРСИНГОМ CSV :)

use rand::seq::SliceRandom;
use std::error::Error;
use std::path::Path;

pub type Sample = (Vec<f32>, u8);

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

        let label: u8 = match record[0].trim().parse() {
            Ok(v) if v <= 9 => v,
            _ => {
                skipped += 1;
                continue;
            }
        };

        let mut pixels = Vec::with_capacity(784);
        let mut ok = true;

        for i in 1..785 {
            let p: u16 = match record[i].trim().parse() {
                Ok(v) if v <= 255 => v,
                _ => {
                    ok = false;
                    break;
                }
            };
            pixels.push(p as f32 / 255.0);
        }

        if ok && pixels.len() == 784 {
            data.push((pixels, label));
        } else {
            skipped += 1;
        }
    }

    if shuffle {
        let mut rng = rand::rng(); // 🔥 НОВЫЙ API
        data.shuffle(&mut rng);
    }

    eprintln!(
        "[data] loaded samples: {}, skipped rows: {}",
        data.len(),
        skipped
    );

    Ok(data)
}
