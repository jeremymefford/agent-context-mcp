use crate::commands::support::{self, DEFAULT_LAUNCHD_LABEL};
use anyhow::{Context, Result};
use std::process::Command;

pub async fn run(label: Option<&str>) -> Result<()> {
    let label = label.unwrap_or(DEFAULT_LAUNCHD_LABEL);
    let plist_path = support::default_plist_path(label)?;
    let domain_target = format!("gui/{}/{}", unsafe { libc::geteuid() }, label);
    let _ = Command::new("launchctl")
        .args(["bootout", &domain_target])
        .status();

    if plist_path.exists() {
        std::fs::remove_file(&plist_path)
            .with_context(|| format!("removing {}", plist_path.display()))?;
    }

    println!("Removed launchd service `{label}`");
    Ok(())
}
