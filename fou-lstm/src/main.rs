use candle_core::{DType, Device, Module, Result, Tensor};
use candle_datasets::Batcher;
use candle_nn::{
    linear, AdamW, LSTMConfig, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap, LSTM, RNN,
};
use stochastic_rs::diffusions::ou::fou;

struct Model {
    // num_layers: usize,
    lstm: Vec<LSTM>,
    out_layer: Linear,
}

impl Model {
    #[must_use]
    fn new(
        vs: VarBuilder,
        in_dim: usize,
        hidden_dim: usize,
        num_lstm_layers: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let vs = &vs.pp("lstm");
        let mut lstm_layers = Vec::with_capacity(num_lstm_layers);
        for layer_idx in 0..num_lstm_layers {
            let config = LSTMConfig {
                layer_idx,
                ..Default::default()
            };
            let lstm = candle_nn::lstm(in_dim, hidden_dim, config, vs.clone())?;
            lstm_layers.push(lstm);
        }
        let out_layer = linear(hidden_dim, out_dim, vs.clone())?;
        Ok(Self {
            lstm: lstm_layers,
            out_layer,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.lstm {
            let states = layer.seq(&x)?;
            x = layer.states_to_tensor(&states)?.unsqueeze(0)?;
        }
        let out = self.out_layer.forward(&x)?;
        Ok(out)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);

    let epochs = 25;
    let epoch_size = 128;
    let in_dim = 1_600_usize;
    let hidden_dim = 64_usize;
    let num_lstm_layers = 2_usize;
    let out_dim = 1_usize;
    let batch_size = 64_usize;
    let net = Model::new(vs, in_dim, hidden_dim, num_lstm_layers, out_dim).unwrap();
    let adamw_params = ParamsAdamW::default();
    let mut opt = AdamW::new(varmap.all_vars(), adamw_params)?;

    for _ in 0..epochs {
        let mut paths = Vec::with_capacity(epoch_size);
        for _ in 0..epoch_size {
            let path_i = fou(0.7, 2.8, 1.0, 5.0, in_dim, Some(0.0), Some(1.0));
            paths.push(Ok(Tensor::from_vec(path_i, &[in_dim], &device)?));
        }
        let batcher = Batcher::new_r1(paths.into_iter())
            .batch_size(batch_size)
            .return_last_incomplete_batch(true);

        for batch in batcher {
            let input = batch?;
            let y = net.forward(&input.unsqueeze(0)?)?;
            let loss = y.mean(0)?;
            opt.backward_step(&loss)?;
            println!("loss: {:?}", loss);
        }
    }

    Ok(())
}
