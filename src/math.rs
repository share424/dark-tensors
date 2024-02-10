use crate::tensor::Tensor;


pub fn abs(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i].abs());
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn add(tensor1: &Tensor, tensor2: &Tensor) -> Tensor {
    assert_eq!(tensor1.shape, tensor2.shape);
    let mut data = Vec::new();
    for i in 0..tensor1.numel() {
        data.push(tensor1.data[i] + tensor2.data[i]);
    }
    Tensor::new(data, tensor1.shape.clone())
}

pub fn add_scalar(tensor: &Tensor, scalar: f32) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i] + scalar);
    }
    Tensor::new(data, tensor.shape.clone())
}


pub fn mul(tensor1: &Tensor, tensor2: &Tensor) -> Tensor {
    assert_eq!(tensor1.shape, tensor2.shape);
    let mut data = Vec::new();
    for i in 0..tensor1.numel() {
        data.push(tensor1.data[i] * tensor2.data[i]);
    }
    Tensor::new(data, tensor1.shape.clone())
}

pub fn mul_scalar(tensor: &Tensor, scalar: f32) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i] * scalar);
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn pow(tensor: &Tensor, exponent: f32) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i].powf(exponent));
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn variance(tensor: &Tensor) -> f32 {
    let mean = tensor.data.iter().sum::<f32>() / tensor.numel() as f32;
    let mut variance = 0.0;
    for i in 0..tensor.numel() {
        variance += (tensor.data[i] - mean).powf(2.0);
    }
    variance / tensor.numel() as f32
}

pub fn std(tensor: &Tensor) -> f32 {
    variance(tensor).sqrt()
}

pub fn rsqrt(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(1.0 / tensor.data[i].sqrt());
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn ln(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i].ln());
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn exp(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i].exp());
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn softmax(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    // to prevent overflow
    // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    let max = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for i in 0..tensor.numel() {
        let value = (tensor.data[i] - max).exp();
        data.push(value);
    }
    let sum: f32 = data.iter().sum();
    for i in 0..tensor.numel() {
        data[i] /= sum;
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn matmul(input: &Tensor, other: &Tensor) -> Tensor {
    // input must be matrix
    assert_eq!(input.ndim(), 2);
    assert_eq!(other.ndim(), 2);

    // matrix multiplication
    assert_eq!(input.shape[1], other.shape[0]);

    let m = input.shape[0];
    let n = input.shape[1];
    let p = other.shape[1];
    let mut data = Tensor::zeros(vec![m, p]);
    
    for i in 0..m {
        for j in 0..p {
            for k in 0..n {
                data.data[i * p + j] += input.data[i * n + k] * other.data[k * p + j];
            }
        }
    }
    data
}

pub fn argmax(tensor: &Tensor, axis: usize) -> Tensor {
    let mut data = Vec::new();
    if axis == 0 {
        for i in 0..tensor.shape[1] {
            let mut max = f32::NEG_INFINITY;
            let mut index = 0;
            for j in 0..tensor.shape[0] {
                if tensor.data[j * tensor.shape[1] + i] > max {
                    max = tensor.data[j * tensor.shape[1] + i];
                    index = j;
                }
            }
            data.push(index as f32);
        }
    } else {
        for i in 0..tensor.shape[0] {
            let mut max = f32::NEG_INFINITY;
            let mut index = 0;
            for j in 0..tensor.shape[1] {
                if tensor.data[i * tensor.shape[1] + j] > max {
                    max = tensor.data[i * tensor.shape[1] + j];
                    index = j;
                }
            }
            data.push(index as f32);
        }
    }
    let len = data.len();
    Tensor::new(data, vec![1, len])
}

pub fn cos(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i].cos());
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn sin(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i].sin());
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn transpose(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.shape[1] {
        for j in 0..tensor.shape[0] {
            data.push(tensor.data[j * tensor.shape[1] + i]);
        }
    }
    Tensor::new(data, vec![tensor.shape[1], tensor.shape[0]])
}
