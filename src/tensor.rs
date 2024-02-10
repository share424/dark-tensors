use safetensors::{SafeTensors, tensor::Dtype};
use memmap2::MmapOptions;
use std::{collections::HashMap, fs::File};
use image;
use image::GenericImageView;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(&mut self, shape: Vec<usize>)  {
        assert_eq!(self.shape.iter().product::<usize>(), shape.iter().product::<usize>());
        self.shape = shape.clone();
    }

    pub fn get(&self, indices: Vec<usize>) -> f32 {
        let mut index = 0;
        for i in 0..indices.len() {
            index *= self.shape[i];
            index += indices[i];
        }
        self.data[index]
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0; size];
        Tensor { data, shape }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        let data = vec![1.0; size as usize];
        Tensor { data, shape }
    }

    pub fn from_image_file(filename: &str) -> Self {
        let img = image::open(filename).unwrap();
        let (width, height) = img.dimensions();

        // load image
        let mut data = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                for c in 0..img.color().channel_count() {
                    data.push(pixel[c as usize] as f32);
                }
            }
        }

        Tensor::new(data, vec![height as usize, width as usize, img.color().channel_count() as usize])
    }

    pub fn slice(&self, start: Vec<usize>, end: Vec<usize>) -> Self {
        // example usage
        // let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // let slice = tensor.slice(vec![0, 0], vec![1, 2]);
        // assert_eq!(slice.data, vec![1.0, 2.0]);
        let mut shape = self.shape.clone();
        let mut data = Vec::new();
        for i in 0..shape.len() {
            shape[i] = end[i] - start[i];
        }

        for i in 0..shape.iter().product::<usize>() {
            let mut indices = Vec::new();
            let mut index = i;
            for j in 0..shape.len() {
                indices.push(index % shape[j]);
                index /= shape[j];
            }

            let mut index = 0;
            for j in 0..shape.len() {
                index *= self.shape[j];
                index += start[j] + indices[j];
            }
            data.push(self.data[index]);
        }

        Tensor::new(data, shape)
    }
}

fn deserialize_weights(value: &[u8], dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F32 => f32::from_le_bytes(value.try_into().unwrap()),
        Dtype::I32 => i32::from_le_bytes(value.try_into().unwrap()) as f32,
        Dtype::F64 => f64::from_le_bytes(value.try_into().unwrap()) as f32,
        Dtype::I64 => i64::from_le_bytes(value.try_into().unwrap()) as f32,
        _ => panic!("Unsupported data type"),
    }
}

pub fn load_weights(filename: &str) -> HashMap<String, Tensor> {
    let file = File::open(filename).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();
    
    let mut weights = HashMap::new();
    let names = tensors.names();

    for name in names {
        let tensor = tensors.tensor(&name).unwrap();
        let buffer_size = match tensor.dtype() {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::I64 => 8,
            _ => panic!("Unsupported data type"),
        };

        let mut data: Vec<f32> = Vec::new();
        let raw_data = tensor.data();
        for i in 0..raw_data.len() / buffer_size {
            let mut bytes = Vec::new();
            for j in 0..buffer_size {
                bytes.push(raw_data[i * buffer_size + j]);
            }
            let value = deserialize_weights(&bytes, tensor.dtype());
            data.push(value);
        }

        weights.insert(name.to_string(), Tensor::new(data, tensor.shape().to_vec()));
    }

    weights

}


pub fn concat(tensors: Vec<&Tensor>, axis: usize) -> Tensor {
    let mut shape = tensors[0].shape.clone();
    let mut data = Vec::new();
    for i in 1..tensors.len() {
        assert_eq!(tensors[i].shape.len(), shape.len());
        for j in 0..shape.len() {
            if j == axis {
                shape[j] += tensors[i].shape[j];
            } else {
                assert_eq!(tensors[i].shape[j], shape[j]);
            }
        }
    }

    for i in 0..shape.iter().product::<usize>() {
        let mut indices = Vec::new();
        let mut index = i;
        for j in 0..shape.len() {
            indices.push(index % shape[j]);
            index /= shape[j];
        }

        let mut tensor_index = 0;
        let mut tensor_offset = 0;
        for j in 0..tensors.len() {
            if indices[axis] < tensors[j].shape[axis] {
                tensor_index = j;
                tensor_offset = indices[axis];
                break;
            } else {
                indices[axis] -= tensors[j].shape[axis];
            }
        }

        indices[axis] = tensor_offset;
        let value = tensors[tensor_index].get(indices);
        data.push(value);
    }

    Tensor::new(data, shape)
}