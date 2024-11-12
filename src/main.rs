#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ai_cli::AiCli;
use anyhow::Result;
use clap::Parser;
use tracing::info;
use tracing_log::AsTrace;
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;
    
    let start = std::time::Instant::now();
    info!("Beginning script");
    let ai_cli = AiCli::parse();
    // a builder for `FmtSubscriber`.
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(ai_cli.verbose.log_level_filter().as_trace())
        // .with_line_number(false)
        // .pretty()
        // .with_target(true)
        // completes the builder.
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");
    let _guard = if ai_cli.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    
    info!("Parsed arguments in {:?}", start.elapsed());
    info!("Args: {ai_cli:?}");
   
    ai_cli.exec(&start)?;
    
    Ok(())
}

// fn mmlu<P: AsRef<std::path::Path>>(
//     mut model: Model,
//     tokenizer: Tokenizer,
//     device: &Device,
//     mmlu_dir: P,
// ) -> anyhow::Result<()> {
//     for dir_entry in mmlu_dir.as_ref().read_dir()?.flatten() {
//         let dir_entry = dir_entry.path();
//         let theme = match dir_entry.file_stem().and_then(|v| v.to_str()) {
//             None => "".to_string(),
//             Some(v) => match v.strip_suffix("_test") {
//                 None => v.replace('_', " "),
//                 Some(v) => v.replace('_', " "),
//             },
//         };
//         if dir_entry.extension().as_ref().and_then(|v| v.to_str()) != Some("csv") {
//             continue;
//         }
//         println!("reading {dir_entry:?}");
//         let dir_entry = std::fs::File::open(dir_entry)?;
//         let mut reader = csv::ReaderBuilder::new()
//             .has_headers(false)
//             .from_reader(dir_entry);
//         let token_a = tokenizer.token_to_id("A").unwrap();
//         let token_b = tokenizer.token_to_id("B").unwrap();
//         let token_c = tokenizer.token_to_id("C").unwrap();
//         let token_d = tokenizer.token_to_id("D").unwrap();
//         for row in reader.records() {
//             let row = match row {
//                 Err(_) => continue,
//                 Ok(row) => row,
//             };
//             if row.len() < 5 {
//                 continue;
//             }
//             let question = row.get(0).unwrap();
//             let answer_a = row.get(1).unwrap();
//             let answer_b = row.get(2).unwrap();
//             let answer_c = row.get(3).unwrap();
//             let answer_d = row.get(4).unwrap();
//             let answer = row.get(5).unwrap();
//             let prompt = format!(
//                     "{} {theme}.\n{question}\nA. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}\nAnswer:\n",
//                     "The following are multiple choice questions (with answers) about"
//                 );
//             let tokens = tokenizer.encode(prompt.as_str(), true).map_err(E::msg)?;
//             let tokens = tokens.get_ids().to_vec();
//             let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
//             let logits = match &mut model {
//                 Model::MixFormer(m) => {
//                     m.clear_kv_cache();
//                     m.forward(&input)?
//                 }
//                 Model::Phi(m) => {
//                     m.clear_kv_cache();
//                     m.forward(&input)?
//                 }
//                 Model::Phi3(m) => {
//                     m.clear_kv_cache();
//                     m.forward(&input, 0)?
//                 }
//                 Model::Quantized(m) => {
//                     m.clear_kv_cache();
//                     m.forward(&input)?
//                 }
//             };
//             let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
//             let logits_v: Vec<f32> = logits.to_vec1()?;
//             let pr_a = logits_v[token_a as usize];
//             let pr_b = logits_v[token_b as usize];
//             let pr_c = logits_v[token_c as usize];
//             let pr_d = logits_v[token_d as usize];
//             let model_answer = if pr_a > pr_b && pr_a > pr_c && pr_a > pr_d {
//                 "A"
//             } else if pr_b > pr_c && pr_b > pr_d {
//                 "B"
//             } else if pr_c > pr_d {
//                 "C"
//             } else {
//                 "D"
//             };

//             println!("{prompt}\n -> {model_answer} vs {answer}");
//         }
//     }
//     Ok(())
// }
