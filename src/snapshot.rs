use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use tokio::sync::Mutex;

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

        let temp_path = self.path.with_extension("tmp");
        let text = serde_json::to_string_pretty(snapshot).context("serializing snapshot json")?;
        tokio::fs::write(&temp_path, format!("{text}\n"))
            .await
            .with_context(|| format!("writing snapshot temp file {}", temp_path.display()))?;
        tokio::fs::rename(&temp_path, &self.path)
            .await
            .with_context(|| format!("replacing snapshot {}", self.path.display()))?;
        Ok(())
    }

    pub async fn remove(&self, key: &str) -> Result<()> {
        let _guard = self.lock.lock().await;
        let mut snapshot = self.read_unlocked().await?;
        snapshot.codebases.remove(key);
        snapshot.last_updated = Some(timestamp());
        self.write_unlocked(&snapshot).await
    }

    pub async fn mark_interrupted_indexing_failed(&self, reason: &str) -> Result<usize> {
        let _guard = self.lock.lock().await;
        let mut snapshot = self.read_unlocked().await?;
        let mut healed = 0usize;

        for entry in snapshot.codebases.values_mut() {
            if entry.status == "indexing" {
                *entry = SnapshotEntry::failed(
                    reason.to_string(),
                    entry
                        .indexing_percentage
                        .or(entry.last_attempted_percentage),
                );
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

impl SnapshotEntry {
    pub fn indexed_with_status(
        indexed_files: Option<u64>,
        total_chunks: Option<u64>,
        index_status: impl Into<String>,
    ) -> Self {
        Self {
            status: "indexed".to_string(),
            indexed_files,
            total_chunks,
            indexing_percentage: None,
            last_attempted_percentage: None,
            index_status: Some(index_status.into()),
            error_message: None,
            last_updated: Some(timestamp()),
            last_audit: None,
            last_head: None,
            last_index_mtime: None,
            last_root_mtime: None,
        }
    }

    pub fn indexing(progress: f64) -> Self {
        Self {
            status: "indexing".to_string(),
            indexed_files: None,
            total_chunks: None,
            indexing_percentage: Some(progress),
            last_attempted_percentage: Some(progress),
            index_status: Some("running".to_string()),
            error_message: None,
            last_updated: Some(timestamp()),
            last_audit: None,
            last_head: None,
            last_index_mtime: None,
            last_root_mtime: None,
        }
    }

    pub fn failed(message: impl Into<String>, last_attempted_percentage: Option<f64>) -> Self {
        Self {
            status: "indexfailed".to_string(),
            indexed_files: None,
            total_chunks: None,
            indexing_percentage: None,
            last_attempted_percentage,
            index_status: Some("failed".to_string()),
            error_message: Some(message.into()),
            last_updated: Some(timestamp()),
            last_audit: None,
            last_head: None,
            last_index_mtime: None,
            last_root_mtime: None,
        }
    }

    pub fn set_indexing_progress(&mut self, progress: f64) {
        self.status = "indexing".to_string();
        self.indexed_files = None;
        self.total_chunks = None;
        self.indexing_percentage = Some(progress);
        self.last_attempted_percentage = Some(progress);
        self.index_status = Some("running".to_string());
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
