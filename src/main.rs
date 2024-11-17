#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ai::{AiCli, AiCliArgs, Settings};
use anyhow::Result;
use clap::Parser;
use tracing::{error, info};
use tracing_log::AsTrace;
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let start = std::time::Instant::now();

    let ai_cli_args = AiCliArgs::parse();

    let settings = Settings::new()?;
    //convert settings.verbosity String into Levelfilter
    // set filter to ai_cli if present, else, from settings
    let log_level_filter = ai_cli_args.verbose.log_level_filter();

    // a builder for `FmtSubscriber`.
    let subscriber = FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(log_level_filter.as_trace())
        // .with_line_number(false)
        // .pretty()
        // .with_target(true)
        // completes the builder.
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    info!(
        "Initialized args, settings, and logging in {:?}",
        start.elapsed()
    );
    let _guard = if ai_cli_args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let concatenated_args = ai_cli_args.other_args.join(" ");

    info!("Prompt is {}", concatenated_args);
    let ai_cli = AiCli::new(
        settings,
        ai_cli_args,
        Some(start),
        log_level_filter
            .to_level()
            .expect("Unable to load log level configuration."),
        concatenated_args,
    );

    match ai_cli.exec() {
        Ok(_) => {}
        Err(e) => {
            error!("{:?}", e);
            error!("Exiting due to error");
            return Ok(());
        }
    }
    Ok(())
}
