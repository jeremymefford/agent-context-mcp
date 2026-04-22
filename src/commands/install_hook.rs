use crate::commands::support;
use anyhow::Result;
use std::path::Path;

pub async fn run(repo: &Path) -> Result<()> {
    let hook_path = support::install_managed_hook(repo)?;
    println!(
        "Installed managed post-commit hook at {}",
        hook_path.display()
    );
    Ok(())
}
