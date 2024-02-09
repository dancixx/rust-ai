use candle_core::{DType, Device, Module, Result, Tensor};
use candle_datasets::Batcher;
use candle_nn::{
    linear, loss::mse, lstm, seq, AdamW, LSTMConfig, Optimizer, ParamsAdamW, Sequential,
    VarBuilder, VarMap, LSTM, RNN,
};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::time::Instant;
use stochastic_rs::diffusions::ou::fou;

mod datasets;

struct Model {
    lstm1: LSTM,
    lstm2: LSTM,
    mlp: Sequential,
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
            vs.pp("lstm-1"),
        )?;
        let lstm2 = lstm(
            hidden_dim,
            hidden_dim,
            LSTMConfig {
                layer_idx: 1,
                ..Default::default()
            },
            vs.pp("lstm-2"),
        )?;
        let mlp = seq()
            .add(linear(hidden_dim, hidden_dim, vs.pp("mpl-linear-1"))?)
            .add_fn(|x| x.gelu())
            .add(linear(hidden_dim, hidden_dim, vs.pp("mpl-linear-2"))?)
            .add_fn(|x| x.gelu())
            .add(linear(hidden_dim, out_dim, vs.pp("mpl-linear-3"))?);

        Ok(Self { lstm1, lstm2, mlp })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone().unsqueeze(1)?;
        let states1 = self.lstm1.seq(&x)?;
        x = self.lstm1.states_to_tensor(&states1)?;
        let states2 = self.lstm2.seq(&x.unsqueeze(1)?)?;
        x = self.lstm2.states_to_tensor(&states2)?;
        let out = self.mlp.forward(&x)?;
        Ok(out)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);

    let epochs = 20_usize;
    let epoch_size = 12_800_usize;
    let in_dim = 4_096_usize;
    let hidden_dim = 64_usize;
    let out_dim = 1_usize;
    let batch_size = 64_usize;
    let net = Model::new(vs, in_dim, hidden_dim, out_dim).unwrap();
    let adamw_params = ParamsAdamW::default();
    let mut opt = AdamW::new(varmap.all_vars(), adamw_params)?;

    let n = 4_096_usize;
    let _hurst = 0.7;
    let mu = 2.8;
    let sigma = 1.0;

    let start = Instant::now();

    for epoch in 0..epochs {
        let mut paths = Vec::with_capacity(epoch_size);
        let thetas = Array1::random(epoch_size, Uniform::new(0.0, 3.0)).to_vec();
        let hursts = Array1::random(epoch_size, Uniform::new(0.01, 0.99)).to_vec();
        let progress_bar = ProgressBar::new(epoch_size as u64);
        progress_bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] ({eta})",
            )?
            .progress_chars("#>-"),
        );
        for idx in 0..epoch_size {
            let hurst = hursts[idx];
            let theta = thetas[idx];
            let mut path = Array1::from_vec(fou(hurst, mu, sigma, theta, n, Some(0.0), Some(16.0)));
            let mean = path.mean().unwrap();
            let std = path.std(0.0);
            path = (path - mean) / std;

            let path = path.to_vec();
            // path.extend(vec![mu, sigma, hurst, theta]);
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
                        loss.to_scalar::<f64>()?
                    );
                }
                Err(_) => break 'inner,
            }
        }

        // let test_dataset = datasets::test_vasicek(1000, in_dim, batch_size, n, &device)?;
        // 'test: for batch in test_dataset {
        //     match batch {
        //         Ok((x, target)) => {
        //             let inp = net.forward(&x)?;
        //             let inp_vec = inp
        //                 .to_vec2::<f64>()?
        //                 .into_iter()
        //                 .flatten()
        //                 .collect::<Vec<_>>();
        //             let target_vec = target
        //                 .to_vec2::<f64>()?
        //                 .into_iter()
        //                 .flatten()
        //                 .collect::<Vec<_>>();
        //             let zip = inp_vec.iter().zip(target_vec.iter()).collect::<Vec<_>>();
        //             println!("result: {:?}", zip);
        //         }
        //         Err(_) => break 'test,
        //     }
        // }

        println!("Epoch {} took {:?}", epoch + 1, start.elapsed());
    }

    Ok(())
}
