use crate::snapshot::SnapshotEntry;
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AuditFingerprint {
    pub head: Option<String>,
    pub index_mtime: Option<u64>,
    pub root_mtime: Option<u64>,
}

pub fn fingerprint_repo(repo: &Path) -> Result<AuditFingerprint> {
    let root_mtime = modified_epoch_millis(repo)?;

    if let Some(git_dir) = resolve_git_dir(repo)? {
        return Ok(AuditFingerprint {
            head: read_git_head(&git_dir)?,
            index_mtime: modified_epoch_millis(&git_dir.join("index"))?,
            root_mtime,
        });
    }

    if let Some(fingerprint) = fingerprint_jj(repo, root_mtime)? {
        return Ok(fingerprint);
    }

    if let Some(fingerprint) = fingerprint_hg(repo, root_mtime)? {
        return Ok(fingerprint);
    }

    if let Some(fingerprint) = fingerprint_svn(repo, root_mtime)? {
        return Ok(fingerprint);
    }

    if let Some(fingerprint) = fingerprint_fossil(repo, root_mtime)? {
        return Ok(fingerprint);
    }

    Ok(AuditFingerprint {
        head: None,
        index_mtime: None,
        root_mtime,
    })
}

pub fn fingerprint_changed(entry: Option<&SnapshotEntry>, fingerprint: &AuditFingerprint) -> bool {
    match entry {
        None => false,
        Some(entry) => {
            entry.last_head.as_deref() != fingerprint.head.as_deref()
                || entry.last_index_mtime != fingerprint.index_mtime
                || entry.last_root_mtime != fingerprint.root_mtime
        }
    }
}

pub fn apply_fingerprint(entry: &mut SnapshotEntry, fingerprint: &AuditFingerprint) {
    entry.last_head = fingerprint.head.clone();
    entry.last_index_mtime = fingerprint.index_mtime;
    entry.last_root_mtime = fingerprint.root_mtime;
    entry.last_audit = Some(crate::snapshot::timestamp());
}

fn resolve_git_dir(repo: &Path) -> Result<Option<PathBuf>> {
    let git_path = repo.join(".git");
    if git_path.is_dir() {
        return Ok(Some(git_path));
    }
    if !git_path.is_file() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(&git_path)
        .with_context(|| format!("reading {}", git_path.display()))?;
    let Some(gitdir) = text.trim().strip_prefix("gitdir:") else {
        return Ok(None);
    };
    let gitdir = gitdir.trim();
    let path = Path::new(gitdir);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo.join(path)
    };
    Ok(Some(resolved))
}

fn read_git_head(git_dir: &Path) -> Result<Option<String>> {
    let head_path = git_dir.join("HEAD");
    if !head_path.exists() {
        return Ok(None);
    }

    let head_text = std::fs::read_to_string(&head_path)
        .with_context(|| format!("reading {}", head_path.display()))?;
    let head_text = head_text.trim().to_string();
    if let Some(reference) = head_text.strip_prefix("ref: ") {
        let ref_path = git_dir.join(reference);
        if ref_path.exists() {
            let commit = std::fs::read_to_string(&ref_path)
                .with_context(|| format!("reading {}", ref_path.display()))?;
            return Ok(Some(commit.trim().to_string()));
        }
    }
    Ok(Some(head_text))
}

fn fingerprint_jj(repo: &Path, root_mtime: Option<u64>) -> Result<Option<AuditFingerprint>> {
    let jj_dir = repo.join(".jj");
    if !jj_dir.exists() {
        return Ok(None);
    }

    let op_id = read_optional_trimmed(&jj_dir.join("working_copy/operation_id"))?;
    let tree_state = read_optional_trimmed(&jj_dir.join("working_copy/tree_state"))?;
    let op_heads = read_dir_names(&jj_dir.join("repo/op_heads/heads"))?;

    let mut parts = Vec::new();
    if let Some(op_id) = op_id {
        parts.push(format!("op={op_id}"));
    }
    if let Some(tree_state) = tree_state {
        parts.push(format!("tree={tree_state}"));
    }
    if !op_heads.is_empty() {
        parts.push(format!("heads={}", op_heads.join(",")));
    }

    Ok(Some(AuditFingerprint {
        head: (!parts.is_empty()).then(|| format!("jj:{}", parts.join(";"))),
        index_mtime: max_modified_epoch_millis(&[
            jj_dir.join("working_copy"),
            jj_dir.join("working_copy/operation_id"),
            jj_dir.join("working_copy/tree_state"),
            jj_dir.join("repo/op_heads/heads"),
        ])?,
        root_mtime,
    }))
}

fn fingerprint_hg(repo: &Path, root_mtime: Option<u64>) -> Result<Option<AuditFingerprint>> {
    let hg_dir = repo.join(".hg");
    if !hg_dir.exists() {
        return Ok(None);
    }

    let branch = read_optional_trimmed(&hg_dir.join("branch"))?;
    let bookmark = read_optional_trimmed(&hg_dir.join("bookmarks.current"))?;
    let mut parts = Vec::new();
    if let Some(branch) = branch {
        parts.push(format!("branch={branch}"));
    }
    if let Some(bookmark) = bookmark {
        parts.push(format!("bookmark={bookmark}"));
    }

    Ok(Some(AuditFingerprint {
        head: (!parts.is_empty()).then(|| format!("hg:{}", parts.join(";"))),
        index_mtime: max_modified_epoch_millis(&[
            hg_dir.join("dirstate"),
            hg_dir.join("branch"),
            hg_dir.join("bookmarks.current"),
            hg_dir.join("store"),
        ])?,
        root_mtime,
    }))
}

fn fingerprint_svn(repo: &Path, root_mtime: Option<u64>) -> Result<Option<AuditFingerprint>> {
    let svn_dir = repo.join(".svn");
    if !svn_dir.exists() {
        return Ok(None);
    }

    Ok(Some(AuditFingerprint {
        head: Some("svn".to_string()),
        index_mtime: max_modified_epoch_millis(&[svn_dir.join("wc.db"), svn_dir.join("entries")])?,
        root_mtime,
    }))
}

fn fingerprint_fossil(repo: &Path, root_mtime: Option<u64>) -> Result<Option<AuditFingerprint>> {
    let checkout = repo.join(".fslckout");
    let alt_checkout = repo.join("_FOSSIL_");
    if !checkout.exists() && !alt_checkout.exists() {
        return Ok(None);
    }

    Ok(Some(AuditFingerprint {
        head: Some("fossil".to_string()),
        index_mtime: max_modified_epoch_millis(&[checkout, alt_checkout])?,
        root_mtime,
    }))
}

fn read_optional_trimmed(path: &Path) -> Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}

fn read_dir_names(path: &Path) -> Result<Vec<String>> {
    if !path.is_dir() {
        return Ok(Vec::new());
    }

    let mut names = std::fs::read_dir(path)
        .with_context(|| format!("reading {}", path.display()))?
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect::<Vec<_>>();
    names.sort();
    Ok(names)
}

fn modified_epoch_millis(path: &Path) -> Result<Option<u64>> {
    if !path.exists() {
        return Ok(None);
    }
    let modified = std::fs::metadata(path)
        .with_context(|| format!("reading metadata for {}", path.display()))?
        .modified()
        .with_context(|| format!("reading modified time for {}", path.display()))?;
    let millis = modified
        .duration_since(std::time::UNIX_EPOCH)
        .context("converting modified time")?
        .as_millis();
    Ok(Some(millis as u64))
}

fn max_modified_epoch_millis(paths: &[PathBuf]) -> Result<Option<u64>> {
    let mut max: Option<u64> = None;
    for path in paths {
        if let Some(value) = modified_epoch_millis(path)? {
            max = Some(max.map_or(value, |current| current.max(value)));
        }
    }
    Ok(max)
}

pub fn merkle_snapshot_path(merkle_dir: &Path, repo: &Path) -> PathBuf {
    let digest = md5::compute(repo.display().to_string());
    merkle_dir.join(format!("{digest:x}.json"))
}

#[cfg(test)]
mod tests {
    use super::{fingerprint_repo, resolve_git_dir};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("agent-context-{prefix}-{nanos}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn resolves_git_worktree_file() {
        let repo = temp_dir("gitdir");
        let actual_git_dir = repo.join(".actual-git");
        fs::create_dir_all(&actual_git_dir).unwrap();
        fs::write(repo.join(".git"), "gitdir: .actual-git\n").unwrap();

        let resolved = resolve_git_dir(&repo).unwrap();
        assert_eq!(resolved.as_deref(), Some(actual_git_dir.as_path()));

        let _ = fs::remove_dir_all(repo);
    }

    #[test]
    fn fingerprints_hg_repo_without_git() {
        let repo = temp_dir("hg");
        let hg_dir = repo.join(".hg");
        fs::create_dir_all(&hg_dir).unwrap();
        fs::write(hg_dir.join("branch"), "default\n").unwrap();
        fs::write(hg_dir.join("dirstate"), "state").unwrap();

        let fingerprint = fingerprint_repo(&repo).unwrap();
        assert_eq!(fingerprint.head.as_deref(), Some("hg:branch=default"));
        assert!(fingerprint.index_mtime.is_some());

        let _ = fs::remove_dir_all(repo);
    }

    #[test]
    fn fingerprints_svn_repo_without_git() {
        let repo = temp_dir("svn");
        let svn_dir = repo.join(".svn");
        fs::create_dir_all(&svn_dir).unwrap();
        fs::write(svn_dir.join("wc.db"), "state").unwrap();

        let fingerprint = fingerprint_repo(&repo).unwrap();
        assert_eq!(fingerprint.head.as_deref(), Some("svn"));
        assert!(fingerprint.index_mtime.is_some());

        let _ = fs::remove_dir_all(repo);
    }
}
