use rand::{rng, RngExt};
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
                self.weights[0][i].push(rng.random_range(-0.5..=0.5));
            }
        }

        //Скрытые веса.
        for i in 1..self.hidden_layers {
            self.weights.push(Vec::new());

            for j in 0..self.hidden_neurons {
                self.weights[i].push(Vec::new());

                for _k in 0..self.hidden_neurons {
                    self.weights[i][j].push(rng.random_range(-0.5..=0.5));
                }
            }
        }

        //Выходные веса.
        self.weights.push(Vec::new());
        for i in 0..self.hidden_neurons {
            self.weights[self.hidden_layers].push(Vec::new());

            for _j in 0..self.output_neurons {
                self.weights[self.hidden_layers][i].push(rng.random_range(-0.5..=0.5));
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

    //Супер неоптимизированная залупа.
    fn predict(&self, data: (Vec<f32>, u8)) -> Vec<Vec<f32>> {
        //TODO Push forward algorythm!

        let mut all_neurons: Vec<Vec<f32>> = vec![data.0];

        for i in 0..=self.hidden_layers {
            all_neurons.push(Vec::new());
            for j in 0..self.biases[i].iter().len() {
                // нейрон в слое l
                let mut neuron: f32 = self.biases[i][j];
                // мы должны взять все веса предыдущего слоя с индексом j(это индекс текущего нейрона)
                for k in 0..self.weights[i].iter().len() {
                    neuron += self.weights[i][k][j] * all_neurons[i][k];
                }
                all_neurons[i + 1].push(sigm(neuron));
            }
        }
        let v_output = all_neurons.iter().last().unwrap();
        println!("Выходной слой нейронов: {:?}", v_output);
        println!();
        println!("Cost function: {}", f_cost(v_output.to_vec(), data.1));

        all_neurons
    }
}

fn sigm(z: f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

fn f_cost(output_vec: Vec<f32>, answ: u8) -> f32 {
    let mut c: f32 = 0.0;

    for i in 0..10 {
        if i != answ {
            c += output_vec[i as usize] * output_vec[i as usize];
        } else {
            c += (1.0 - output_vec[i as usize]) * (1.0 - output_vec[i as usize]);
        }
    }
    c / 2.0
}

fn main() {
    let mut train_full: Vec<Sample> = data::load_mnist_csv("MNIST/mnist_train.csv", true)
        .expect("Наебнулось что то в train_full");
    let test_full: Vec<Sample> = data::load_mnist_csv("MNIST/mnist_test.csv", false)
        .expect("Наебнулось что то в train_full");

    let mut net1 = NeuralNetwork {
        input_neurons: 784,
        output_neurons: 10,

        hidden_layers: 2,
        hidden_neurons: 16,

        learning_rate: 0.01,
        mini_batch_size: 100,

        weights: Vec::new(),
        biases: Vec::new(),
    };

    net1.create();

    //println!(
    //    "кол-во входных нейронов: {},
    //    кол-во выходных нейронов: {},
    //    кол-во скрытых слоев: {},
    //    кол-во скрытых нейронов в каждом слое: {},
    //    функция активации - sigmoida,
    //    ====================================",
    //    net1.input_neurons, net1.output_neurons, net1.hidden_layers, net1.hidden_neurons
    //);
    //
    ////println!("веса: {:?}", net1.weights);
    ////println!();
    ////println!("смещения: {:?}", net1.biases);
    ////println!();
    //let mut inp_vec: Vec<f32> = vec![];
    //for _i in 0..net1.input_neurons {
    //    inp_vec.push(rng().random_range(0.0..=1.0));
    //}

    net1.predict(train_full[0].clone());
}
