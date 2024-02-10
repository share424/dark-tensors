use std::env;

pub mod tensor;
pub mod math;
pub mod nn;
pub mod activation;
pub mod model;

fn run_mnist(model_path: &String) {
    let args: Vec<String> = env::args().collect();
    let model = model::mnist_classifier::MNISTClassifier::from_pretrained(model_path);

    let input_image = &args[3];
    let mut input_tensor = tensor::Tensor::from_image_file(input_image);
    // normalize
    input_tensor = math::mul_scalar(&input_tensor, 1.0 / 255.0);
    input_tensor.reshape(vec![1, 784]);
    let output = model.forward(&input_tensor);

    println!("Prediction: {:?}", math::argmax(&output, 1));
    println!("Probs: {:?}", output);
}



fn main() {
    let args: Vec<String> = env::args().collect();
    
    let model_name = &args[1];
    let model_path = &args[2];

    match model_name.as_str() {
        "mnist-classifier" => run_mnist(model_path),
        _ => println!("Model not found"),
    }    

    // let model = model::llama2::LlamaRotaryEmbedding::new(100, None);
    // let position_ids = tensor::Tensor::ones(vec![1, 100]);
    // let output = model.forward(position_ids);
    // println!("{:?}", output.0);

}
