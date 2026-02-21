use rand::RngExt;
use std::f32::consts::E;
mod data;
use data::Sample;

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
                self.weights[0][i].push(rng.random_range(-0.0866..=0.0866));
            }
        }

        //Скрытые веса.
        for i in 1..self.hidden_layers {
            self.weights.push(Vec::new());

            for j in 0..self.hidden_neurons {
                self.weights[i].push(Vec::new());

                for _k in 0..self.hidden_neurons {
                    self.weights[i][j].push(rng.random_range(-0.4330..=0.4330));
                }
            }
        }

        //Выходные веса.
        self.weights.push(Vec::new());
        for i in 0..self.output_neurons {
            self.weights[self.hidden_layers].push(Vec::new());

            for _j in 0..self.hidden_neurons {
                self.weights[self.hidden_layers][i].push(rng.random_range(-0.480..=0.480));
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

    fn predict(&self, data: &Vec<f32>) -> u8 {
        let mut all_neurons: Vec<Vec<f32>> = vec![data.clone()]; // мб клон

        for i in 0..=self.hidden_layers {
            all_neurons.push(Vec::new());
            for j in 0..self.biases[i].len() {
                // нейрон в слое l
                let mut neuron: f32 = self.biases[i][j];
                // мы должны взять все веса предыдущего слоя с индексом j(это индекс текущего нейрона)
                for k in 0..all_neurons[i].len() {
                    neuron += self.weights[i][j][k] * all_neurons[i][k];
                }
                all_neurons[i + 1].push(sigm(neuron));
            }
        }
        let v_output = all_neurons.iter().last().unwrap();
        let mut answ = 0;
        let mut max = 0.0;
        //println!("{:?}", v_output);
        for i in 0..v_output.len() {
            if max < v_output[i] {
                max = v_output[i];
                answ = i;
            }
        }
        answ as u8
    }

    fn gradient(
        &self,
        data: &(Vec<f32>, u8),
        grad_w: &mut Vec<Vec<Vec<f32>>>,
        grad_b: &mut Vec<Vec<f32>>,
    ) -> f32 {
        let mut all_neurons: Vec<Vec<f32>> = vec![data.0.clone()]; // мб клон

        for i in 0..=self.hidden_layers {
            all_neurons.push(Vec::new());
            for j in 0..self.biases[i].len() {
                // нейрон в слое l
                let mut neuron: f32 = self.biases[i][j];
                // мы должны взять все веса предыдущего слоя с индексом j(это индекс текущего нейрона)
                for k in 0..all_neurons[i].len() {
                    neuron += self.weights[i][j][k] * all_neurons[i][k];
                }
                all_neurons[i + 1].push(sigm(neuron));
            }
        }
        let v_output = all_neurons.iter().last().unwrap();
        let c = f_cost(v_output.to_vec(), data.1); // локальная функция ошибки;

        let mut guilt_vec: Vec<f32> = vec![];

        for i in 0..self.output_neurons {
            if i != data.1.into() {
                guilt_vec.push(v_output[i] * v_output[i] * (1.0 - v_output[i]));
            } else {
                guilt_vec.push((v_output[i] - 1.0) * v_output[i] * (1.0 - v_output[i]))
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
        c
    }

    fn train(&mut self) {
        let train_full: Vec<Sample> = data::load_mnist_csv("MNIST/mnist_train.csv", true)
            .expect("Наебнулось что то в train_full");
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

            //Начинаем итерироваться по минибатчу и ему туда передаем эти градиенты.
            let mut _c: f32 = 0.0;
            for j in 0..self.mini_batch_size {
                _c += self.gradient(
                    &train_full[i * self.mini_batch_size as usize + j as usize],
                    &mut grad_w,
                    &mut grad_b,
                )
            }
            //println!(
            //    "Итерация: {} Ошибка: {}",
            //    i,
            //    c / (self.mini_batch_size as f32)
            //);
            //TODO в этом месте нужно слить grad_w*-(lr/mini_batch_size) в weights и grad_b*(-lr/mini_batch_size)
            for k in 0..self.weights.len() {
                //слои
                for n in 0..self.weights[k].len() {
                    //нейроны
                    for l in 0..self.weights[k][n].len() {
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
    let test_full: Vec<Sample> = data::load_mnist_csv("MNIST/mnist_test.csv", false)
        .expect("Наебнулось что то в train_full");

    let mut net1 = NeuralNetwork {
        input_neurons: 784,
        output_neurons: 10,

        hidden_layers: 2,
        hidden_neurons: 16,

        learning_rate: 12.0,
        mini_batch_size: 100,

        weights: Vec::new(),
        biases: Vec::new(),
    };

    net1.create();

    println!();
    println!();

    println!("Тест на проверочной выборке до обучения:");
    let mut counter = 0;
    for i in 0..test_full.len() {
        if net1.predict(&test_full[i].0) == test_full[i].1 {
            counter += 1;
        }
    }
    println!("Проверочный результат: {}/{}", counter, test_full.len());
    println!(
        "Accuracy: {:.2}%",
        100.0 * counter as f32 / test_full.len() as f32
    );

    for i in 0..10 {
        net1.train();
    }
    println!();
    println!("Обучение завершено!");
    println!();

    println!("Тест на проверочной выборке:");
    let mut counter = 0;
    for i in 0..test_full.len() {
        if net1.predict(&test_full[i].0) == test_full[i].1 {
            counter += 1;
        }
    }
    println!("Финальный результат: {}/{}", counter, test_full.len());
    println!(
        "Accuracy: {:.2}%",
        100.0 * counter as f32 / test_full.len() as f32
    );
}
