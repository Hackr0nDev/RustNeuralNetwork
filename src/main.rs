use rand::{rng, RngExt};
use std::{f32::consts::E, process::exit, usize};
mod data;
use data::{load_mnist_csv, Sample};

#[derive(Debug)]
struct NeuralNetwork {
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

        self.weights.push(Vec::new());
        for i in 0..self.hidden_neurons {
            self.weights[0].push(Vec::new());

            for _j in 0..self.input_neurons {
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
        for i in 0..self.output_neurons {
            self.weights[self.hidden_layers].push(Vec::new());

            for _j in 0..self.hidden_neurons {
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

    fn predict(
        &self,
        data: &(Vec<f32>, u8),
        grad_w: &mut Vec<Vec<Vec<f32>>>,
        grad_b: &mut Vec<Vec<f32>>,
    ) {
        let mut all_neurons: Vec<Vec<f32>> = vec![data.0.clone()]; // мб клон

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
        let c = f_cost(v_output.to_vec(), data.1); // локальная функция ошибки;

        let mut guilt_vec: Vec<f32> = vec![];

        for i in 0..10 {
            if i != data.1 {
                guilt_vec.push(
                    v_output[i as usize] * v_output[i as usize] * (1.0 - v_output[i as usize]),
                );
            } else {
                guilt_vec.push(
                    (v_output[i as usize] - 1.0)
                        * v_output[i as usize]
                        * (1.0 - v_output[i as usize]),
                )
            }
        }

        //TODO у нас есть δ^L, с помощью него мы должны заполнить grad_w, grad_b этого слоя.

        let n = all_neurons.len();
        for i in 0..guilt_vec.len() {
            for j in 0..all_neurons[n - 2].len() {
                grad_w[self.weights.len() - 1][i][j] += guilt_vec[i] * all_neurons[n - 2][j];
            }
            grad_b[self.biases.len() - 1][i] += guilt_vec[i];
        }

        //TODO разработать все остальное (δ^l)
        for i in (1..n - 1).rev() {
            // от L до 1
            let mut new_guilt: Vec<f32> = Vec::with_capacity(self.biases[i - 1].len());
            for j in 0..self.biases[i - 1].len() {
                let z = all_neurons[i][j] * (1.0 - all_neurons[i][j]);
                let mut guilt_l_j = 0.0;
                for k in 0..guilt_vec.len() {
                    guilt_l_j += self.weights[i][k][j] * guilt_vec[k];
                }
                new_guilt.push(guilt_l_j * z);

                grad_b[i - 1][j] += new_guilt[j];
                for k in 0..self.weights[i - 1][j].len() {
                    grad_w[i - 1][j][k] += new_guilt[j] * all_neurons[i - 1][k];
                }
            }
            guilt_vec = new_guilt;
        }
    }

    fn train(&mut self) {
        let mut train_full: Vec<Sample> = data::load_mnist_csv("MNIST/mnist_train.csv", false)
            .expect("Наебнулось что то в train_full");
        //let _test_full: Vec<Sample> = data::load_mnist_csv("MNIST/mnist_test.csv", false)
        //    .expect("Наебнулось что то в train_full");
        let q = -self.learning_rate / self.mini_batch_size as f32;

        for i in 0..train_full.len() / self.mini_batch_size as usize {
            //Создаем градиенты W, B
            let mut grad_w: Vec<Vec<Vec<f32>>> = self.weights.clone();
            for j in 0..grad_w.len() {
                for k in 0..grad_w[j].len() {
                    for h in 0..grad_w[j][k].len() {
                        grad_w[j][k][h] = 0.0;
                    }
                }
            }
            let mut grad_b: Vec<Vec<f32>> = self.biases.clone();
            for j in 0..grad_b.len() {
                for k in 0..grad_b[j].len() {
                    grad_b[j][k] = 0.0;
                }
            }
            train_full = data::load_mnist_csv("MNIST/mnist_train.csv", true)
                .expect("Наебнулось что то в train_full"); // Чтоб оно там шафлилось.

            //Начинаем итерироваться по минибатчу и ему туда передаем эти градиенты.
            for j in 0..10 {
                //self.mini_batch_size {
                self.predict(
                    &train_full[(j as usize * (i as usize + 1) as usize) as usize],
                    &mut grad_w,
                    &mut grad_b,
                );
            }
            //TODO в этом месте нужно слить grad_w*-(lr/mini_batch_size) в weights и grad_b*(-lr/mini_batch_size)
            for k in 0..self.weights.len() {
                //слои
                for n in 0..self.weights[i].len() {
                    //нейроны
                    for l in 0..self.weights[i][n].len() {
                        // веса
                        self.weights[k][n][l] += grad_w[k][n][l] * q;
                    }
                    self.biases[k][n] += grad_b[k][n] * q;
                }
            }
        }
    }
}

fn sigm(z: f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

fn f_cost(output_vec: Vec<f32>, answ: u8) -> f32 {
    let mut temp_c: f32 = 0.0;

    for i in 0..10 {
        if i != answ {
            temp_c += output_vec[i as usize] * output_vec[i as usize];
        } else {
            temp_c += (1.0 - output_vec[i as usize]) * (1.0 - output_vec[i as usize]);
        }
    }
    temp_c / 2.0
}

fn main() {
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
    net1.train();

    //Ручной минибатч.
    //push to git-hub
}
