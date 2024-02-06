use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    linear, lstm, AdamW, LSTMConfig, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap, LSTM, RNN,
};

struct Model {
    num_layers: usize,
    lstm1: LSTM,
    linear1: Linear,
}

impl Model {
    #[must_use]
    fn new(
        vs: VarBuilder,
        input_dim: usize,
        hidden_size: usize,
        num_layers: usize,
        out_dim: usize,
    ) -> Result<Self> {
        let lstm1 = lstm(
            input_dim,
            hidden_size,
            LSTMConfig::default(),
            vs.push_prefix("lstm1"),
        )?;
        let linear1 = linear(hidden_size, out_dim, vs.push_prefix("linear1"))?;

        Ok(Self {
            lstm1,
            linear1,
            num_layers,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let state = self
            .lstm1
            .step(&x, &self.lstm1.zero_state(self.num_layers)?)?;

        let out = self.lstm1.states_to_tensor(&[state])?;
        let out = self.linear1.forward(&out)?;

        Ok(out)
    }
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let input_dim = 1_604_usize;
    let hidden_size = 64_usize;
    let num_layers = 2_usize;
    let out_dim = 1_usize;
    let net = Model::new(vs, input_dim, hidden_size, num_layers, out_dim).unwrap();
    let adamw_params = ParamsAdamW::default();
    let mut opt = AdamW::new(varmap.all_vars(), adamw_params)?;

    let epochs = 25;

    for _ in 0..epochs {
        let x = Tensor::randn(0_f32, 1.0_f32, &[num_layers, input_dim], &device)?;
        println!("x: {:?}", x);
        let y = net.forward(&x)?;
        let loss = y.mean(0)?;
        opt.backward_step(&loss)?;

        println!("loss: {:?}", loss);
    }

    Ok(())
}
