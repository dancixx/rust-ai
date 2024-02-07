use candle_core::{DType, Device, Module, Result, Tensor};
use candle_datasets::Batcher;
use candle_nn::{
    linear, loss::mse, lstm, AdamW, LSTMConfig, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap,
    LSTM, RNN,
};

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use stochastic_rs::diffusions::ou::fou;

struct Model {
    lstm1: LSTM,
    lstm2: LSTM,
    linear: Linear,
}

impl Model {
    #[must_use]
    fn new(vs: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(Self {
            lstm1: lstm(
                in_dim,
                hidden_dim,
                LSTMConfig {
                    layer_idx: 0,
                    ..Default::default()
                },
                vs.pp("lstm1"),
            )?,
            lstm2: lstm(
                hidden_dim,
                hidden_dim,
                LSTMConfig {
                    layer_idx: 1,
                    ..Default::default()
                },
                vs.pp("lstm2"),
            )?,
            linear: linear(hidden_dim, out_dim, vs.clone())?,
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

    let epochs = 200;
    let epoch_size = 10_000_usize;
    let in_dim = 1_600_usize;
    let hidden_dim = 64_usize;
    let out_dim = 1_usize;
    let batch_size = 128_usize;
    let net = Model::new(vs, in_dim, hidden_dim, out_dim).unwrap();
    let adamw_params = ParamsAdamW::default();
    let mut opt = AdamW::new(varmap.all_vars(), adamw_params)?;

    let n = 1_600_usize;
    let hurst = 0.7;
    let mu = 2.8;
    let sigma = 1.0;

    for epoch in 0..epochs {
        let mut paths = Vec::with_capacity(epoch_size);
        let thetas = Array1::random(epoch_size, Uniform::new(0.0, 5.0)).to_vec();
        let progress_bar = ProgressBar::new(epoch_size as u64);
        progress_bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})",
            )?
            .progress_chars("#>-"),
        );
        for idx in 0..epoch_size {
            let path = fou(hurst, mu, sigma, thetas[idx], n, Some(0.0), Some(1.0));
            // let mean = path.mean().unwrap();
            // let std = path.std(0.0);
            // path = (path - mean) / std;
            //let path = path.to_vec();

            //path.extend(vec![mu, sigma, hurst]);
            paths.push(Ok((
                Tensor::from_vec(path, &[in_dim], &device)?,
                Tensor::new(&[thetas[idx]], &device)?,
            )));
            progress_bar.inc(1);
        }
        progress_bar.finish();

        let batcher = Batcher::new_r2(paths.into_iter())
            .batch_size(batch_size)
            .return_last_incomplete_batch(false);

        'inner: for (batch_idx, batch) in batcher.enumerate() {
            match batch {
                Ok((x, target)) => {
                    let inp = net.forward(&x)?;
                    let loss = mse(&inp, &target)?;
                    opt.backward_step(&loss)?;
                    println!(
                        "Epoch: {}, Batch: {}, Loss: {:?}",
                        epoch + 1,
                        batch_idx + 1,
                        loss
                    );
                }
                Err(_) => break 'inner,
            }
        }
    }

    Ok(())
}
