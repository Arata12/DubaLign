use clap::Parser;
use dubalign::spectro;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Espectrogramas apilados de todas las pistas de audio de un archivo.")]
struct Args {
    file: PathBuf,
    #[arg(long, help = "inicio del zoom (s)")]
    start: Option<f64>,
    #[arg(long, default_value_t = 2.5, help = "duración del zoom (s)")]
    dur: f64,
    #[arg(long, help = "episodio completo en vez de zoom")]
    full: bool,
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let out = args.output.unwrap_or_else(|| {
        let stem = args
            .file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "out".into());
        let kind = if args.full { "full" } else { "zoom" };
        PathBuf::from("output").join(format!("{stem}.{kind}.png"))
    });
    spectro::render(
        &args.file,
        &out,
        args.start,
        args.dur,
        args.full,
        &HashMap::new(),
        &[],
        None,
    )?;
    println!("{}", out.display());
    Ok(())
}
