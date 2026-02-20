use rand::{rng, RngExt};
use std::{f32::consts::E, usize};
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

    fn predict(
        &self,
        data: &(Vec<f32>, u8),
        grad_w: &mut Vec<Vec<Vec<f32>>>,
        grad_b: &mut Vec<Vec<f32>>,
    ) {
        let mut all_neurons: Vec<Vec<f32>> = vec![data.0]; // мб клон

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

        //TODO не забыть потом высчитывать С для сравнения точности. (Глобальное С)
        //TODO На основе локального с мы должны вычислить текущий градиент

        //                         δ^L
        let mut gradient: Vec<f32> = vec![];

        for i in 0..10 {
            if i != data.1 {
                gradient.push(
                    v_output[i as usize] * v_output[i as usize] * (1.0 - v_output[i as usize]),
                );
            } else {
                gradient.push(
                    (v_output[i as usize] - 1.0)
                        * v_output[i as usize]
                        * (1.0 - v_output[i as usize]),
                )
            }
        }

        //      ВЕЛИКАЯ ПРОБЛЕММА: НАХУЯ Я КАЖДЫЙ РАЗ ПЕРЕСОЗДАЮ ГРАДИЕНТ????
        //      ГРАДИЕНТ W,B ДОЛЖНЫ ПЕРЕДАВАТЬСЯ(СОЗДАВАТЬСЯ) В НАЧАЛЕ МИНИБАТЧА.

        //TODO работаем с δ^L

        grad_w.push(Vec::new()); // Добавили СЛОЙ
        grad_b.push(Vec::new());
        for i in 0..gradient.len() {
            grad_w[i].push(Vec::new());
            for j in 0..all_neurons[all_neurons.len() - 2].len() {
                grad_w[i][j].push(gradient[i] * all_neurons[all_neurons.len() - 2][j]);
            }
            grad_b[i].push(gradient[i]);
        }
    }

    fn train(&self) {
        let mut train_full: Vec<Sample> = data::load_mnist_csv("MNIST/mnist_train.csv", true)
            .expect("Наебнулось что то в train_full");
        //let _test_full: Vec<Sample> = data::load_mnist_csv("MNIST/mnist_test.csv", false)
        //    .expect("Наебнулось что то в train_full");

        for i in 0..train_full.len() / self.mini_batch_size as usize {
            //Создаем градиенты W, B
            let mut grad_w: Vec<Vec<Vec<f32>>> = self.weights.clone();
            for j in 0..grad_w.len() {
                for k in 0..grad_w[i].len() {
                    for h in 0..grad_w[i][j].len() {
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
            for j in 0..self.mini_batch_size {
                self.predict(
                    &train_full[(j as usize * (i + 1) as usize) as usize],
                    &mut grad_w,
                    &mut grad_b,
                );
            }
            //TODO в этом месте нужно слить grad_w*-(lr/mini_batch_size) в weights и grad_b*(-lr/mini_batch_size)
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

    //Ручной минибатч.
}
