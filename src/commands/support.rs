use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_LISTEN: &str = "127.0.0.1:8765";
const HOOK_START: &str = "# >>> agent-context managed block >>>";
const HOOK_END: &str = "# <<< agent-context managed block <<<";

pub fn library_dir() -> Result<PathBuf> {
    let home = dirs_home()?;
    Ok(home.join("Library"))
}

pub fn logs_dir() -> Result<PathBuf> {
    Ok(library_dir()?.join("Logs"))
}

pub fn hooks_log_path() -> Result<PathBuf> {
    Ok(logs_dir()?.join("agent-context-hooks.log"))
}

pub fn resolve_repo_root(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("reading current directory")?
            .join(path)
    };
    let absolute = absolute.canonicalize().unwrap_or(absolute);

    for candidate in absolute.ancestors() {
        let dot_git = candidate.join(".git");
        if dot_git.exists() {
            return Ok(candidate.to_path_buf());
        }
    }

    bail!("{} is not inside a git repository", absolute.display())
}

pub fn render_hook_block(repo: &Path) -> Result<String> {
    let hooks_log = hooks_log_path()?;
    let repo_shell = shell_single_quote(&repo.display().to_string());
    let hooks_log_shell = shell_single_quote(&hooks_log.display().to_string());
    Ok(format!(
        "{HOOK_START}\nAGENT_CONTEXT_BIN=\"${{AGENT_CONTEXT_BIN:-$(command -v agent-context 2>/dev/null || printf '%s' \"$HOME/.cargo/bin/agent-context\")}}\"\nif [ -x \"$AGENT_CONTEXT_BIN\" ]; then\n  nohup \"$AGENT_CONTEXT_BIN\" refresh-one {repo_shell} >> {hooks_log_shell} 2>&1 &\nfi\n{HOOK_END}"
    ))
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

pub fn install_managed_hook(repo: &Path) -> Result<PathBuf> {
    let repo_root = resolve_repo_root(repo)?;
    let hook_path = repo_root.join(".git/hooks/post-commit");
    let mut content = if hook_path.exists() {
        fs::read_to_string(&hook_path)
            .with_context(|| format!("reading {}", hook_path.display()))?
    } else {
        "#!/bin/sh\n".to_string()
    };

    if !content.starts_with("#!") {
        content = format!("#!/bin/sh\n{content}");
    }

    let block = render_hook_block(&repo_root)?;
    content = replace_managed_block(&content, &block);
    if !content.ends_with('\n') {
        content.push('\n');
    }

    if let Some(parent) = hook_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    fs::write(&hook_path, content).with_context(|| format!("writing {}", hook_path.display()))?;
    make_executable(&hook_path)?;
    Ok(hook_path)
}

fn replace_managed_block(existing: &str, block: &str) -> String {
    if let Some(start) = existing.find(HOOK_START)
        && let Some(end) = existing.find(HOOK_END)
    {
        let end = end + HOOK_END.len();
        let mut rendered = String::new();
        rendered.push_str(existing[..start].trim_end());
        rendered.push('\n');
        rendered.push_str(block);
        rendered.push('\n');
        rendered.push_str(existing[end..].trim_start());
        return rendered;
    }

    let mut rendered = existing.trim_end().to_string();
    if !rendered.ends_with('\n') {
        rendered.push('\n');
    }
    rendered.push('\n');
    rendered.push_str(block);
    rendered.push('\n');
    rendered
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path)
            .with_context(|| format!("reading {}", path.display()))?
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
            .with_context(|| format!("updating permissions for {}", path.display()))?;
    }
    Ok(())
}

fn dirs_home() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home))
}

#[cfg(test)]
mod tests {
    use super::{render_hook_block, replace_managed_block};
    use std::path::Path;

    #[test]
    fn hook_block_targets_repo_refresh() {
        let block = render_hook_block(Path::new("/tmp/repo")).unwrap();
        assert!(block.contains("refresh-one"));
        assert!(block.contains("refresh-one '/tmp/repo'"));
    }

    #[test]
    fn managed_block_replacement_is_deterministic() {
        let existing = "#!/bin/sh\n\n# >>> agent-context managed block >>>\nold\n# <<< agent-context managed block <<<\n";
        let updated = replace_managed_block(
            existing,
            "# >>> agent-context managed block >>>\nnew\n# <<< agent-context managed block <<<",
        );
        assert!(updated.contains("new"));
        assert!(!updated.contains("\nold\n"));
    }

    #[test]
    fn hook_block_shell_escapes_repo_path() {
        let block = render_hook_block(Path::new("/tmp/repo'$(whoami)")).unwrap();
        assert!(block.contains("refresh-one '/tmp/repo'\"'\"'$(whoami)'"));
    }
}
