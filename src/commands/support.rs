use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_LAUNCHD_LABEL: &str = "dev.agent-context.mcp";
pub const DEFAULT_LISTEN: &str = "127.0.0.1:8765";
const HOOK_START: &str = "# >>> agent-context managed block >>>";
const HOOK_END: &str = "# <<< agent-context managed block <<<";

pub fn library_dir() -> Result<PathBuf> {
    let home = dirs_home()?;
    Ok(home.join("Library"))
}

pub fn launch_agents_dir() -> Result<PathBuf> {
    Ok(library_dir()?.join("LaunchAgents"))
}

pub fn logs_dir() -> Result<PathBuf> {
    Ok(library_dir()?.join("Logs"))
}

pub fn default_plist_path(label: &str) -> Result<PathBuf> {
    Ok(launch_agents_dir()?.join(format!("{label}.plist")))
}

pub fn stdout_log_path() -> Result<PathBuf> {
    Ok(logs_dir()?.join("agent-context-mcp.log"))
}

pub fn stderr_log_path() -> Result<PathBuf> {
    Ok(logs_dir()?.join("agent-context-mcp.err.log"))
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
    Ok(format!(
        "{HOOK_START}\nAGENT_CONTEXT_BIN=\"${{AGENT_CONTEXT_BIN:-$(command -v agent-context 2>/dev/null || printf '%s' \"$HOME/.cargo/bin/agent-context\")}}\"\nif [ -x \"$AGENT_CONTEXT_BIN\" ]; then\n  nohup \"$AGENT_CONTEXT_BIN\" refresh-one \"{}\" >> \"{}\" 2>&1 &\nfi\n{HOOK_END}",
        repo.display(),
        hooks_log.display()
    ))
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

pub fn plist_contents(
    label: &str,
    executable: &Path,
    workdir: &Path,
    listen: &str,
    config_path: Option<&Path>,
) -> Result<String> {
    let stdout = stdout_log_path()?;
    let stderr = stderr_log_path()?;
    let mut args = vec![format!(
        "<string>{}</string>",
        xml_escape(&executable.display().to_string())
    )];
    if let Some(path) = config_path {
        args.push("<string>--config</string>".to_string());
        args.push(format!(
            "<string>{}</string>",
            xml_escape(&path.display().to_string())
        ));
    }
    args.extend([
        "<string>serve</string>".to_string(),
        "<string>--listen</string>".to_string(),
        format!("<string>{}</string>", xml_escape(listen)),
    ]);

    Ok(format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    {}
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>{}</string>
  <key>StandardOutPath</key>
  <string>{}</string>
  <key>StandardErrorPath</key>
  <string>{}</string>
</dict>
</plist>
"#,
        args.join("\n    "),
        xml_escape(&workdir.display().to_string()),
        xml_escape(&stdout.display().to_string()),
        xml_escape(&stderr.display().to_string()),
    ))
}

pub fn existing_launchd_labels() -> Vec<&'static str> {
    vec![DEFAULT_LAUNCHD_LABEL, "com.jeremy.agent-context-mcp"]
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

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn dirs_home() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home))
}

#[cfg(test)]
mod tests {
    use super::{plist_contents, render_hook_block, replace_managed_block};
    use std::path::Path;

    #[test]
    fn hook_block_targets_repo_refresh() {
        let block = render_hook_block(Path::new("/tmp/repo")).unwrap();
        assert!(block.contains("refresh-one"));
        assert!(block.contains("/tmp/repo"));
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
    fn plist_generation_includes_serve_arguments() {
        let plist = plist_contents(
            "dev.agent-context.mcp",
            Path::new("/tmp/agent-context"),
            Path::new("/tmp"),
            "127.0.0.1:8765",
            Some(Path::new("/tmp/config.toml")),
        )
        .unwrap();
        assert!(plist.contains("serve"));
        assert!(plist.contains("--listen"));
        assert!(plist.contains("/tmp/config.toml"));
    }
}
