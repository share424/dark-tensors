use crate::tensor::Tensor;


pub fn relu(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i].max(0.0));
    }
    Tensor::new(data, tensor.shape.clone())
}

pub fn silu(tensor: &Tensor) -> Tensor {
    let mut data = Vec::new();
    for i in 0..tensor.numel() {
        data.push(tensor.data[i] / (1.0 + (-tensor.data[i]).exp()));
    }
    Tensor::new(data, tensor.shape.clone())
}