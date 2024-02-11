use anyhow::Ok;

mod datasets;
#[allow(non_snake_case)]
mod lstm_model_1D;

fn main() -> anyhow::Result<()> {
    lstm_model_1D::test()?;

    Ok(())
}
