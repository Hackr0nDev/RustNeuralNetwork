use rand::RngExt;
use std::f32::consts::E;
mod data;
use data::{load_mnist_csv, Sample};

#[derive(Debug)]
struct NeuralNetwork {
    //stohastic_gradient_weights: Vec<Vec<Vec<f32>>>,
    //stohastic_gradient_biases: Vec<Vec<f32>>,
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,

    input_neurons: usize,
    output_neurons: usize,

    hidden_layers: usize,
    hidden_neurons: usize, //пока что количество нейронов в скрытых слоях будет одинаковое.

    learning_rate: f32,   // пока что будет фиксированным.
    mini_batch_size: u16, // допустим изначально будет 100
}

impl NeuralNetwork {
    fn create(&mut self) {
        let mut rng = rand::rng();

        //ТОЛЬКО ВЕСА

        //Входные веса
        self.weights.push(Vec::new());
        for i in 0..self.input_neurons {
            self.weights[0].push(Vec::new());

            for _j in 0..self.hidden_neurons {
                self.weights[0][i].push(0.1); //rng
            }
        }

        //Скрытые веса.
        for i in 1..=self.hidden_layers {
            self.weights.push(Vec::new());

            for j in 0..self.hidden_neurons {
                self.weights[i].push(Vec::new());

                for _k in 0..self.hidden_neurons {
                    self.weights[i][j].push(0.1); //rng
                }
            }
        }

        //Выходные веса.
        self.weights.push(Vec::new());
        for i in 0..self.output_neurons {
            self.weights[self.hidden_layers + 1].push(Vec::new());

            for _j in 0..self.hidden_neurons {
                self.weights[self.hidden_layers + 1][i].push(0.1); //rng.random_range(-0.5..=0.5)
            }
        }

        //ТОЛЬКО СДВИГИ.

        // входные не нужны.

        //Скрытые сдвиги.
        for i in 0..self.hidden_layers {
            self.biases.push(Vec::new());

            for _j in 0..self.hidden_neurons {
                self.biases[i].push(0.0);
            }
        }

        //Выходной слой сдвиги.
        self.biases.push(Vec::new());
        for _i in 0..self.output_neurons {
            self.biases[self.hidden_layers].push(0.0);
        }
    }
    fn predict(&self, data: (Vec<f32>, u8)) {
        //TODO Push forward algorythm!

        let mut all_neurons: Vec<Vec<f32>> = vec![data.0];
    }
}

fn sigm(z: f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

fn main() {
    let mut net1 = NeuralNetwork {
        input_neurons: 3,
        output_neurons: 3,

        hidden_layers: 2,
        hidden_neurons: 2,

        learning_rate: 0.01,
        mini_batch_size: 100,

        weights: Vec::new(),
        biases: Vec::new(),
    };

    net1.create();
    println!("{:?}", net1.weights);
    println!();
    //net1.predict((vec![1.0, 0.234, 0.943], 4));
}
