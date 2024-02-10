use crate::nn;
use crate::tensor;
use crate::activation;
use crate::math;

pub struct MNISTClassifier {
    pub hidden_layer: nn::Linear,
    pub output_layer: nn::Linear,
}

impl MNISTClassifier {
    pub fn new(hidden_size: usize) -> Self {
        let hidden_layer = nn::Linear::new(784, hidden_size);
        let output_layer = nn::Linear::new(hidden_size, 10);

        MNISTClassifier { hidden_layer, output_layer }
    }

    pub fn forward(&self, input: &tensor::Tensor) -> tensor::Tensor {
        let output = self.hidden_layer.forward(input);
        let output = activation::relu(&output);
        let output = self.output_layer.forward(&output);
        let output = math::softmax(&output);

        output
    }

    pub fn from_pretrained(model_path: &str) -> Self {
        let tensors = tensor::load_weights(model_path);

        let hidden_layer = nn::Linear::from_weights(
            tensors.get("0.linear.weights").unwrap().clone(),
            tensors.get("0.linear.bias").unwrap().clone(),
        );
        let output_layer = nn::Linear::from_weights(
            tensors.get("1.linear.weights").unwrap().clone(),
            tensors.get("1.linear.bias").unwrap().clone(),
        );

        MNISTClassifier { hidden_layer, output_layer }
    }
}