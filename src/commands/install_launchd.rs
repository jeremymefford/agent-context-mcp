use crate::commands::support::{self, DEFAULT_LAUNCHD_LABEL};
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

pub async fn run(
    label: Option<&str>,
    listen: &str,
    workdir: Option<&Path>,
    config_path: Option<&Path>,
) -> Result<()> {
    let label = label.unwrap_or(DEFAULT_LAUNCHD_LABEL);
    let plist_path = support::default_plist_path(label)?;
    let executable = std::env::current_exe().context("resolving current executable")?;
    let workdir = match workdir {
        Some(path) => path.to_path_buf(),
        None => std::env::current_dir().context("reading current directory")?,
    };

    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::create_dir_all(support::logs_dir()?).context("creating launchd log directory")?;

    let plist = support::plist_contents(label, &executable, &workdir, listen, config_path)?;
    std::fs::write(&plist_path, plist)
        .with_context(|| format!("writing {}", plist_path.display()))?;

    let domain_target = format!("gui/{}/{}", unsafe { libc::geteuid() }, label);
    let _ = Command::new("launchctl")
        .args(["bootout", &domain_target])
        .status();
    let bootstrap_target = format!("gui/{}", unsafe { libc::geteuid() });
    let status = Command::new("launchctl")
        .args([
            "bootstrap",
            &bootstrap_target,
            &plist_path.display().to_string(),
        ])
        .status()
        .context("running launchctl bootstrap")?;
    if !status.success() {
        anyhow::bail!("launchctl bootstrap failed for {}", plist_path.display());
    }

    let status = Command::new("launchctl")
        .args(["kickstart", "-k", &domain_target])
        .status()
        .context("running launchctl kickstart")?;
    if !status.success() {
        anyhow::bail!("launchctl kickstart failed for {domain_target}");
    }

    println!(
        "Installed launchd service `{label}` at {}",
        plist_path.display()
    );
    Ok(())
}
