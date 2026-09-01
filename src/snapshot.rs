use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;

static ATOMIC_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct SnapshotStore {
    path: PathBuf,
    lock: Mutex<()>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    #[serde(default = "default_format_version", rename = "formatVersion")]
    pub format_version: String,
    #[serde(
        default = "default_index_format_version",
        rename = "indexFormatVersion"
    )]
    pub index_format_version: String,
    #[serde(default = "default_search_root_version", rename = "searchRootVersion")]
    pub search_root_version: String,
    #[serde(
        default,
        rename = "embeddingFingerprint",
        skip_serializing_if = "Option::is_none"
    )]
    pub embedding_fingerprint: Option<String>,
    #[serde(default)]
    pub codebases: BTreeMap<String, SnapshotEntry>,
    #[serde(default)]
    pub worktrees: BTreeMap<String, WorktreeSnapshotEntry>,
    #[serde(
        default,
        rename = "lastUpdated",
        skip_serializing_if = "Option::is_none"
    )]
    pub last_updated: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotEntry {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexed_files: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_chunks: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexing_percentage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_attempted_percentage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(
        default,
        rename = "embeddingProfile",
        skip_serializing_if = "Option::is_none"
    )]
    pub embedding_profile: Option<String>,
    #[serde(
        default,
        rename = "embeddingFingerprint",
        skip_serializing_if = "Option::is_none"
    )]
    pub embedding_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_audit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_head: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_index_mtime: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_root_mtime: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WorktreeSnapshotEntry {
    pub status: String,
    pub canonical_root: String,
    pub repo_identity: String,
    pub overlay_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub changed_files: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_files: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlay_mismatch_reason: Option<String>,
    #[serde(
        default,
        rename = "embeddingProfile",
        skip_serializing_if = "Option::is_none"
    )]
    pub embedding_profile: Option<String>,
    #[serde(
        default,
        rename = "embeddingFingerprint",
        skip_serializing_if = "Option::is_none"
    )]
    pub embedding_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
}

impl SnapshotStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            lock: Mutex::new(()),
        }
    }

    pub async fn read(&self) -> Result<Snapshot> {
        let _guard = self.lock.lock().await;
        self.read_unlocked().await
    }

    pub async fn update<F>(&self, mutate: F) -> Result<Snapshot>
    where
        F: FnOnce(&mut Snapshot),
    {
        let _guard = self.lock.lock().await;
        let mut snapshot = self.read_unlocked().await?;
        mutate(&mut snapshot);
        snapshot.last_updated = Some(timestamp());
        self.write_unlocked(&snapshot).await?;
        Ok(snapshot)
    }

    async fn read_unlocked(&self) -> Result<Snapshot> {
        if !self.path.exists() {
            return Ok(Snapshot {
                format_version: default_format_version(),
                index_format_version: default_index_format_version(),
                search_root_version: default_search_root_version(),
                ..Snapshot::default()
            });
        }

        let text = tokio::fs::read_to_string(&self.path)
            .await
            .with_context(|| format!("reading snapshot at {}", self.path.display()))?;
        let mut snapshot: Snapshot =
            serde_json::from_str(&text).context("parsing snapshot json")?;
        if snapshot.format_version.is_empty() {
            snapshot.format_version = default_format_version();
        }
        if snapshot.index_format_version.is_empty() {
            snapshot.index_format_version = default_index_format_version();
        }
        if snapshot.search_root_version.is_empty() {
            snapshot.search_root_version = default_search_root_version();
        }
        Ok(snapshot)
    }

    async fn write_unlocked(&self, snapshot: &Snapshot) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating snapshot dir {}", parent.display()))?;
        }

        let text = serde_json::to_string_pretty(snapshot).context("serializing snapshot json")?;
        atomic_replace(&self.path, format!("{text}\n").as_bytes())
            .await
            .with_context(|| format!("writing snapshot {}", self.path.display()))?;
        Ok(())
    }

    pub async fn remove(&self, key: &str) -> Result<()> {
        let _guard = self.lock.lock().await;
        let mut snapshot = self.read_unlocked().await?;
        snapshot.codebases.remove(key);
        snapshot.last_updated = Some(timestamp());
        self.write_unlocked(&snapshot).await
    }

    pub async fn mark_interrupted_indexing_failed_except(
        &self,
        reason: &str,
        resumable: &BTreeSet<String>,
    ) -> Result<usize> {
        let _guard = self.lock.lock().await;
        let mut snapshot = self.read_unlocked().await?;
        let mut healed = 0usize;

        for (repo, entry) in &mut snapshot.codebases {
            if entry.status == "indexing" && !resumable.contains(repo) {
                *entry = SnapshotEntry::failed(
                    reason.to_string(),
                    entry
                        .indexing_percentage
                        .or(entry.last_attempted_percentage),
                    entry.embedding_profile.clone(),
                    entry.embedding_fingerprint.clone(),
                );
                healed += 1;
            }
        }
        for (repo, entry) in &mut snapshot.worktrees {
            if entry.status == "indexing" && !resumable.contains(repo) {
                entry.status = "indexfailed".to_string();
                entry.overlay_status = Some("failed".to_string());
                entry.overlay_mismatch_reason = Some(reason.to_string());
                entry.last_updated = Some(timestamp());
                healed += 1;
            }
        }

        if healed > 0 {
            snapshot.last_updated = Some(timestamp());
            self.write_unlocked(&snapshot).await?;
        }

        Ok(healed)
    }
}

pub(crate) async fn atomic_replace(path: &Path, contents: &[u8]) -> Result<()> {
    let sequence = ATOMIC_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp_path = path.with_extension(format!("tmp-{}-{sequence}", std::process::id()));

    if let Err(error) = tokio::fs::write(&temp_path, contents).await {
        let _ = tokio::fs::remove_file(&temp_path).await;
        return Err(error)
            .with_context(|| format!("writing atomic temp file {}", temp_path.display()));
    }
    if let Err(error) = tokio::fs::rename(&temp_path, path).await {
        let _ = tokio::fs::remove_file(&temp_path).await;
        return Err(error).with_context(|| {
            format!(
                "replacing {} from atomic temp file {}",
                path.display(),
                temp_path.display()
            )
        });
    }
    Ok(())
}

impl WorktreeSnapshotEntry {
    pub fn indexing(
        canonical_root: impl Into<String>,
        repo_identity: impl Into<String>,
        overlay_id: impl Into<String>,
        embedding_profile: Option<String>,
        embedding_fingerprint: Option<String>,
    ) -> Self {
        Self {
            status: "indexing".to_string(),
            canonical_root: canonical_root.into(),
            repo_identity: repo_identity.into(),
            overlay_id: overlay_id.into(),
            overlay_status: Some("queued".to_string()),
            changed_files: Some(0),
            deleted_files: Some(0),
            overlay_bytes: Some(0),
            overlay_mismatch_reason: None,
            embedding_profile,
            embedding_fingerprint,
            last_updated: Some(timestamp()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn indexed(
        canonical_root: impl Into<String>,
        repo_identity: impl Into<String>,
        overlay_id: impl Into<String>,
        overlay_status: impl Into<String>,
        changed_files: u64,
        deleted_files: u64,
        overlay_bytes: u64,
        embedding_profile: Option<String>,
        embedding_fingerprint: Option<String>,
    ) -> Self {
        Self {
            status: "indexed".to_string(),
            canonical_root: canonical_root.into(),
            repo_identity: repo_identity.into(),
            overlay_id: overlay_id.into(),
            overlay_status: Some(overlay_status.into()),
            changed_files: Some(changed_files),
            deleted_files: Some(deleted_files),
            overlay_bytes: Some(overlay_bytes),
            overlay_mismatch_reason: None,
            embedding_profile,
            embedding_fingerprint,
            last_updated: Some(timestamp()),
        }
    }

    pub fn failed(
        canonical_root: impl Into<String>,
        repo_identity: impl Into<String>,
        overlay_id: impl Into<String>,
        message: impl Into<String>,
        embedding_profile: Option<String>,
        embedding_fingerprint: Option<String>,
    ) -> Self {
        Self {
            status: "indexfailed".to_string(),
            canonical_root: canonical_root.into(),
            repo_identity: repo_identity.into(),
            overlay_id: overlay_id.into(),
            overlay_status: Some("failed".to_string()),
            changed_files: None,
            deleted_files: None,
            overlay_bytes: None,
            overlay_mismatch_reason: Some(message.into()),
            embedding_profile,
            embedding_fingerprint,
            last_updated: Some(timestamp()),
        }
    }
}

impl SnapshotEntry {
    pub fn indexed_with_status(
        indexed_files: Option<u64>,
        total_chunks: Option<u64>,
        index_status: impl Into<String>,
        embedding_profile: Option<String>,
        embedding_fingerprint: Option<String>,
    ) -> Self {
        Self {
            status: "indexed".to_string(),
            indexed_files,
            total_chunks,
            indexing_percentage: None,
            last_attempted_percentage: None,
            index_status: Some(index_status.into()),
            error_message: None,
            embedding_profile,
            embedding_fingerprint,
            last_updated: Some(timestamp()),
            last_audit: None,
            last_head: None,
            last_index_mtime: None,
            last_root_mtime: None,
        }
    }

    pub fn indexing(
        progress: f64,
        index_status: impl Into<String>,
        embedding_profile: Option<String>,
        embedding_fingerprint: Option<String>,
    ) -> Self {
        Self {
            status: "indexing".to_string(),
            indexed_files: None,
            total_chunks: None,
            indexing_percentage: Some(progress),
            last_attempted_percentage: Some(progress),
            index_status: Some(index_status.into()),
            error_message: None,
            embedding_profile,
            embedding_fingerprint,
            last_updated: Some(timestamp()),
            last_audit: None,
            last_head: None,
            last_index_mtime: None,
            last_root_mtime: None,
        }
    }

    pub fn failed(
        message: impl Into<String>,
        last_attempted_percentage: Option<f64>,
        embedding_profile: Option<String>,
        embedding_fingerprint: Option<String>,
    ) -> Self {
        Self {
            status: "indexfailed".to_string(),
            indexed_files: None,
            total_chunks: None,
            indexing_percentage: None,
            last_attempted_percentage,
            index_status: Some("failed".to_string()),
            error_message: Some(message.into()),
            embedding_profile,
            embedding_fingerprint,
            last_updated: Some(timestamp()),
            last_audit: None,
            last_head: None,
            last_index_mtime: None,
            last_root_mtime: None,
        }
    }

    pub fn set_indexing_progress(&mut self, progress: f64, index_status: impl Into<String>) {
        self.status = "indexing".to_string();
        self.indexed_files = None;
        self.total_chunks = None;
        self.indexing_percentage = Some(progress);
        self.last_attempted_percentage = Some(progress);
        self.index_status = Some(index_status.into());
        self.error_message = None;
        self.last_updated = Some(timestamp());
    }
}

pub fn timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn default_format_version() -> String {
    "v3".to_string()
}

fn default_index_format_version() -> String {
    "v1".to_string()
}

fn default_search_root_version() -> String {
    "v1".to_string()
}

#[cfg(test)]
mod tests {
    use super::atomic_replace;
    use serde_json::Value;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[tokio::test]
    async fn concurrent_atomic_replacements_do_not_share_temp_files() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agent-context-atomic-write-{nanos}"));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let path = root.join("state.json");

        let mut writes = Vec::new();
        for value in 0..32 {
            let path = path.clone();
            writes.push(tokio::spawn(async move {
                let body = format!(r#"{{"value":{value}}}"#);
                atomic_replace(&path, body.as_bytes()).await
            }));
        }
        for write in writes {
            write.await.unwrap().unwrap();
        }

        let body = tokio::fs::read_to_string(&path).await.unwrap();
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert!(parsed["value"].as_u64().is_some());
        let mut entries = tokio::fs::read_dir(&root).await.unwrap();
        let mut names = Vec::new();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            names.push(entry.file_name());
        }
        assert_eq!(names, vec!["state.json"]);

        tokio::fs::remove_dir_all(root).await.unwrap();
    }
}
