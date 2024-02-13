use anyhow::Ok;
use clap::Parser;

mod datasets;

mod lstm_model_1_d;
mod lstm_model_2_d;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.model.as_str() {
        "lstm_model_1_d" => Ok(lstm_model_1_d::test()?),
        "lstm_model_2_d" => Ok(lstm_model_2_d::test()?),
        _ => todo!("Model not found"),
    }
}
