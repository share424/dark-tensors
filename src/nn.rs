use crate::tensor::Tensor;
use crate::math;

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let weight = Tensor::zeros(vec![input_dim, output_dim]);
        let bias = Tensor::zeros(vec![1, output_dim]);

        Linear { weight, bias }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // input must be 2D in 1xN shape
        assert_eq!(input.ndim(), 2);
        assert_eq!(input.shape[0], 1);
        
        // matrix multiplication
        let output = math::matmul(&input, &self.weight);

        // add bias
        let output = math::add(&output, &self.bias);
        
        output
    }

    pub fn from_weights(weight: Tensor, bias: Tensor) -> Self {
        // weight must be 2D
        assert_eq!(weight.ndim(), 2);
        
        if bias.ndim() == 1 {
            let mut bias = bias.clone();
            bias.reshape(vec![1, bias.shape[0]]);
            return Linear { weight, bias };
        }

        // bias must be 2D
        assert_eq!(bias.ndim(), 2);
        assert_eq!(bias.shape[0], 1);

        Linear { weight, bias }
    }
}