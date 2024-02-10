// adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

use crate::{math, tensor, nn, activation};

pub struct LlamaRMSNorm {
    pub weights: tensor::Tensor,
    pub variance_epsilon: f32,
}

impl LlamaRMSNorm {
    pub fn new(hidden_size: usize, variance_epsilon: Option<f32>) -> Self {
        let weights = tensor::Tensor::ones(vec![hidden_size]);

        LlamaRMSNorm { weights, variance_epsilon: variance_epsilon.unwrap_or(1e-6) }
    }

    pub fn forward(&self, hidden_states: &tensor::Tensor) -> tensor::Tensor {
        let mut ss: f32 = 0.0;
        for i in 0..hidden_states.numel() {
            ss += hidden_states.data[i] * hidden_states.data[i];
        }
        ss /= hidden_states.numel() as f32;
        ss += self.variance_epsilon;
        ss = 1.0 / ss.sqrt();

        // normalize and scale
        let mut output = Vec::new();
        for i in 0..hidden_states.numel() {
            output.push(self.weights.data[i] * (ss * hidden_states.data[i]));
        }
        tensor::Tensor { data: output, shape: hidden_states.shape.clone() }
    }
}

pub struct LlamaMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl LlamaMLP {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        LlamaMLP {
            gate_proj: nn::Linear::new(hidden_size, intermediate_size),
            up_proj: nn::Linear::new(hidden_size, intermediate_size),
            down_proj: nn::Linear::new(hidden_size, intermediate_size),
        }
    }

    pub fn forward(&self, hidden_states: &tensor::Tensor) -> tensor::Tensor {
        let gate = activation::silu(&self.gate_proj.forward(hidden_states));
        let up = &self.up_proj.forward(hidden_states);
        let temp = math::mul(&gate, &up);
        let output = self.down_proj.forward(&temp);

        output
    }

    pub fn from_weights(gate_proj: nn::Linear, up_proj: nn::Linear, down_proj: nn::Linear) -> Self {
        LlamaMLP { gate_proj, up_proj, down_proj }
    }
}

pub struct LlamaRotaryEmbedding {
    pub inv_freq: Vec<f32>,
}

impl LlamaRotaryEmbedding {
    pub fn new(dim: usize, base: Option<usize>) -> Self {
        let b = base.unwrap_or(10000) as f32;
        let mut inv_freq = Vec::new();
        for i in (0..dim).step_by(2) {
            let freq = 1.0 / b.powf(i as f32 / dim as f32);
            inv_freq.push(freq);
        }

        LlamaRotaryEmbedding { inv_freq }
    }

    pub fn forward(&self, position_ids: tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
        let n = position_ids.shape[0];

        let mut inv_freq = tensor::Tensor::new(self.inv_freq.clone(), vec![1, self.inv_freq.len()]);
        for _ in 1..n {
            inv_freq = tensor::concat(vec![&inv_freq, &inv_freq.clone()], 0);
        }

        inv_freq.reshape(vec![self.inv_freq.len(), n]);
        let mut emb = math::matmul(&inv_freq, &position_ids);
        emb = math::transpose(&emb);
        emb = tensor::concat(vec![&emb.clone(), &emb.clone()], 1);
        let cos_emb = math::cos(&emb);
        let sin_emb = math::sin(&emb);

        (cos_emb, sin_emb)
    }
}

pub fn rotate_half(x: tensor::Tensor) -> tensor::Tensor {
    // take the first half of the last dimension
    let x1 = x.slice(vec![0], vec![x.shape[0] / 2]);
    // take the second half of the last dimension
    let mut x2 = x.slice(vec![x.shape[0] / 2], vec![x.shape[0]]);
    x2 = math::mul_scalar(&x2, -1.0);
    // swap the two halves
    tensor::concat(vec![&x2, &x1], 0)
}

pub fn apply_rotary_pos_embed(q: tensor::Tensor, k: tensor::Tensor, cos: tensor::Tensor, sin: tensor::Tensor) -> (tensor::Tensor, tensor::Tensor) {
    let ql = math::mul(&q, &cos);
    let qr = math::mul(&rotate_half(q), &sin);
    let q_embed = math::add(&ql, &qr);

    let kl = math::mul(&k, &cos);
    let kr = math::mul(&rotate_half(k), &sin);
    let k_embed = math::add(&kl, &kr);

    (q_embed, k_embed)
}

pub struct LlamaConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32
}

pub struct LlamaAttention {
    pub config: LlamaConfig,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub rotary_emb: LlamaRotaryEmbedding,
}

impl LlamaAttention {
    pub fn new(
        config: LlamaConfig
    ) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let q_proj = nn::Linear::new(config.hidden_size, config.num_attention_heads * head_dim);
        let k_proj = nn::Linear::new(config.hidden_size, config.num_attention_heads * head_dim);
        let v_proj = nn::Linear::new(config.hidden_size, config.num_attention_heads * head_dim);
        let o_proj = nn::Linear::new(config.num_attention_heads * head_dim, config.hidden_size);

        let rotary_emb = LlamaRotaryEmbedding::new(head_dim, None);

        LlamaAttention { config, q_proj, k_proj, v_proj, o_proj, rotary_emb }
    }
}