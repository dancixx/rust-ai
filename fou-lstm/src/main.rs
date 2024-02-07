use candle_core::{DType, Device, Module, Result, Tensor};
use candle_datasets::Batcher;
use candle_nn::{
    linear, lstm, AdamW, LSTMConfig, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap, LSTM, RNN,
};
use stochastic_rs::diffusions::ou::fou;

struct Model {
    // num_layers: usize,
    lstm1: LSTM,
    lstm2: LSTM,
    linear: Linear,
}

impl Model {
    #[must_use]
    fn new(vs: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        let lstm1 = lstm(
            in_dim,
            hidden_dim,
            LSTMConfig {
                layer_idx: 0,
                ..Default::default()
            },
            vs.pp("lstm1"),
        )?;
        println!("lstm1: {:?}", lstm1);
        let lstm2 = lstm(
            hidden_dim,
            hidden_dim,
            LSTMConfig {
                layer_idx: 1,
                ..Default::default()
            },
            vs.pp("lstm2"),
        )?;
        println!("lstm2: {:?}", lstm2);
        let linear = linear(hidden_dim, out_dim, vs.clone())?;
        println!("linear: {:?}", linear);
        Ok(Self {
            lstm1,
            lstm2,
            linear,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone().unsqueeze(1)?;
        let states1 = self.lstm1.seq(&x)?;
        x = self.lstm1.states_to_tensor(&states1)?;
        let states2 = self.lstm2.seq(&x.unsqueeze(1)?)?;
        x = self.lstm2.states_to_tensor(&states2)?;
        let out = self.linear.forward(&x)?;
        Ok(out)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);

    let epochs = 25;
    let epoch_size = 10_000;
    let in_dim = 1_600_usize;
    let hidden_dim = 64_usize;
    let out_dim = 1_usize;
    let batch_size = 64_usize;
    let net = Model::new(vs, in_dim, hidden_dim, out_dim).unwrap();
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
            let y = net.forward(&input)?;
            let loss = y.mean(0)?;
            opt.backward_step(&loss)?;
            println!("loss: {:?}", loss);
        }
    }

    Ok(())
}
