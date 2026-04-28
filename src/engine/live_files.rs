use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use xxhash_rust::xxh3::xxh3_128;

const TEXT_SAMPLE_BYTES: usize = 8 * 1024;

#[derive(Clone)]
pub struct LiveFileStore {
    max_entries: usize,
    cache: Arc<Mutex<LiveFileCacheState>>,
}

#[derive(Default)]
struct LiveFileCacheState {
    access_tick: u64,
    files: HashMap<PathBuf, CachedLiveFile>,
}

struct CachedLiveFile {
    snapshot: Arc<LiveFileSnapshot>,
    byte_len: u64,
    modified_at: SystemTime,
    last_access_tick: u64,
}

#[derive(Debug, Clone)]
pub struct LiveFileSnapshot {
    pub repo: PathBuf,
    pub relative_path: String,
    #[allow(dead_code)]
    pub absolute_path: PathBuf,
    pub text: Arc<String>,
    pub line_starts: Arc<Vec<usize>>,
    pub file_hash: String,
    pub byte_len: u64,
    pub modified_at: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextMatch {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: u64,
    pub end_line: u64,
}

impl LiveFileStore {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            cache: Arc::new(Mutex::new(LiveFileCacheState::default())),
        }
    }

    pub fn load_snapshot(&self, repo: &Path, relative_path: &str) -> Result<Arc<LiveFileSnapshot>> {
        let canonical_repo = repo
            .canonicalize()
            .with_context(|| format!("resolving repo root {}", repo.display()))?;
        let normalized_relative_path = normalize_relative_path(relative_path);
        if normalized_relative_path.is_empty() {
            bail!("file path is empty after normalization");
        }

        let requested_path = canonical_repo.join(&normalized_relative_path);
        let canonical_file = requested_path
            .canonicalize()
            .with_context(|| format!("resolving file {}", requested_path.display()))?;
        if !canonical_file.starts_with(&canonical_repo) {
            bail!(
                "file `{normalized_relative_path}` escapes repo root {}",
                canonical_repo.display()
            );
        }

        let metadata = fs::metadata(&canonical_file)
            .with_context(|| format!("reading metadata for {}", canonical_file.display()))?;
        if !metadata.is_file() {
            bail!("{} is not a regular file", canonical_file.display());
        }
        let byte_len = metadata.len();
        let modified_at = metadata.modified().unwrap_or(UNIX_EPOCH);

        if let Some(snapshot) = self.cached_snapshot(&canonical_file, byte_len, modified_at)? {
            return Ok(snapshot);
        }

        let bytes = fs::read(&canonical_file)
            .with_context(|| format!("reading {}", canonical_file.display()))?;
        let sample_len = bytes.len().min(TEXT_SAMPLE_BYTES);
        let sample = &bytes[..sample_len];
        if looks_like_lfs_pointer(sample) || looks_binary(sample) {
            bail!("{} is not a text file", canonical_file.display());
        }

        let text = String::from_utf8(bytes)
            .map_err(|_| anyhow::anyhow!("{} is not valid UTF-8 text", canonical_file.display()))?;
        let relative_path = canonical_file
            .strip_prefix(&canonical_repo)
            .unwrap_or(Path::new(&normalized_relative_path))
            .display()
            .to_string()
            .replace('\\', "/");

        let snapshot = Arc::new(LiveFileSnapshot {
            repo: canonical_repo,
            relative_path,
            absolute_path: canonical_file.clone(),
            line_starts: Arc::new(line_start_offsets(&text)),
            file_hash: format!("{:032x}", xxh3_128(text.as_bytes())),
            byte_len,
            modified_at,
            text: Arc::new(text),
        });
        self.insert_snapshot(canonical_file, snapshot.clone())?;
        Ok(snapshot)
    }

    fn cached_snapshot(
        &self,
        canonical_path: &Path,
        byte_len: u64,
        modified_at: SystemTime,
    ) -> Result<Option<Arc<LiveFileSnapshot>>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("live file cache poisoned"))?;
        let tick = next_access_tick(&mut cache);
        let Some(entry) = cache.files.get_mut(canonical_path) else {
            return Ok(None);
        };
        entry.last_access_tick = tick;
        if entry.byte_len == byte_len && entry.modified_at == modified_at {
            return Ok(Some(entry.snapshot.clone()));
        }
        cache.files.remove(canonical_path);
        Ok(None)
    }

    fn insert_snapshot(
        &self,
        canonical_path: PathBuf,
        snapshot: Arc<LiveFileSnapshot>,
    ) -> Result<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| anyhow::anyhow!("live file cache poisoned"))?;
        let tick = next_access_tick(&mut cache);
        cache.files.insert(
            canonical_path.clone(),
            CachedLiveFile {
                byte_len: snapshot.byte_len,
                modified_at: snapshot.modified_at,
                snapshot,
                last_access_tick: tick,
            },
        );
        evict_lru_files(&mut cache, self.max_entries, Some(&canonical_path));
        Ok(())
    }
}

impl LiveFileSnapshot {
    pub fn total_lines(&self) -> u64 {
        self.line_starts.len() as u64
    }

    pub fn slice_lines(&self, start_line: u64, end_line: u64) -> Option<&str> {
        let (start, end) = self.line_bounds(start_line, end_line)?;
        self.text.get(start..end)
    }

    pub fn line_text(&self, line: u64) -> Option<&str> {
        self.slice_lines(line, line)
            .map(|text| text.trim_end_matches(['\r', '\n']))
    }

    pub fn find_literal_matches(
        &self,
        query: &str,
        case_sensitive: bool,
        whole_word: bool,
        max_matches: usize,
    ) -> Result<Vec<TextMatch>> {
        if query.is_empty() {
            bail!("query is empty");
        }
        if max_matches == 0 {
            return Ok(Vec::new());
        }

        let mut matches = Vec::new();
        if case_sensitive {
            for (start_byte, matched) in self.text.match_indices(query) {
                let end_byte = start_byte + matched.len();
                if whole_word && !is_word_boundary(&self.text, start_byte, end_byte) {
                    continue;
                }
                matches.push(self.text_match(start_byte, end_byte));
                if matches.len() >= max_matches {
                    break;
                }
            }
            return Ok(matches);
        }

        let lowered_text = self.text.to_lowercase();
        let lowered_query = query.to_lowercase();
        for (start_byte, matched) in lowered_text.match_indices(&lowered_query) {
            let end_byte = start_byte + matched.len();
            if whole_word && !is_word_boundary(&self.text, start_byte, end_byte) {
                continue;
            }
            matches.push(self.text_match(start_byte, end_byte));
            if matches.len() >= max_matches {
                break;
            }
        }
        Ok(matches)
    }

    fn line_bounds(&self, start_line: u64, end_line: u64) -> Option<(usize, usize)> {
        if start_line == 0 || end_line < start_line {
            return None;
        }
        let start_index = usize::try_from(start_line.saturating_sub(1)).ok()?;
        let end_index = usize::try_from(end_line.saturating_sub(1)).ok()?;
        let start = *self.line_starts.get(start_index)?;
        let end = self
            .line_starts
            .get(end_index + 1)
            .copied()
            .unwrap_or_else(|| self.text.len());
        Some((start, end))
    }

    fn text_match(&self, start_byte: usize, end_byte: usize) -> TextMatch {
        TextMatch {
            start_byte,
            end_byte,
            start_line: line_number_for_offset(&self.line_starts, start_byte),
            end_line: line_number_for_offset(&self.line_starts, end_byte.saturating_sub(1)),
        }
    }
}

fn normalize_relative_path(path: &str) -> String {
    path.replace('\\', "/").trim_matches('/').to_string()
}

fn line_start_offsets(text: &str) -> Vec<usize> {
    let mut offsets = vec![0usize];
    for (index, byte) in text.bytes().enumerate() {
        if byte == b'\n' && index + 1 < text.len() {
            offsets.push(index + 1);
        }
    }
    offsets
}

fn line_number_for_offset(line_starts: &[usize], byte_offset: usize) -> u64 {
    match line_starts.binary_search(&byte_offset) {
        Ok(index) => index as u64 + 1,
        Err(index) => index as u64,
    }
}

fn is_word_boundary(text: &str, start_byte: usize, end_byte: usize) -> bool {
    let prev = text[..start_byte].chars().next_back();
    let next = text[end_byte..].chars().next();
    !prev.is_some_and(is_word_char) && !next.is_some_and(is_word_char)
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

fn looks_like_lfs_pointer(sample: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(sample) else {
        return false;
    };
    let text = text.trim_start_matches('\u{feff}');
    text.starts_with("version https://git-lfs.github.com/spec/v1")
        && text.contains("\noid sha256:")
        && text.contains("\nsize ")
}

fn looks_binary(sample: &[u8]) -> bool {
    if sample.is_empty() {
        return false;
    }
    if sample.contains(&0) {
        return true;
    }
    std::str::from_utf8(sample).is_err()
}

fn next_access_tick(state: &mut LiveFileCacheState) -> u64 {
    state.access_tick = state.access_tick.saturating_add(1);
    state.access_tick
}

fn evict_lru_files(state: &mut LiveFileCacheState, max_entries: usize, preserve: Option<&Path>) {
    if max_entries == 0 {
        state.files.clear();
        return;
    }
    while state.files.len() > max_entries {
        let Some((path, _)) = state
            .files
            .iter()
            .filter(|(path, _)| preserve.is_none_or(|keep| keep != path.as_path()))
            .min_by_key(|(_, entry)| entry.last_access_tick)
            .map(|(path, entry)| (path.clone(), entry.last_access_tick))
        else {
            break;
        };
        state.files.remove(&path);
    }
}

#[cfg(test)]
mod tests {
    use super::LiveFileStore;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos();
        let path = std::env::temp_dir().join(format!("agent-context-live-file-{name}-{nanos}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn load_snapshot_rejects_repo_escape() {
        let root = temp_dir("escape");
        let repo = root.join("repo");
        let other = root.join("other.txt");
        fs::create_dir_all(&repo).unwrap();
        fs::write(&other, "hello").unwrap();

        let store = LiveFileStore::new(4);
        let error = store.load_snapshot(&repo, "../other.txt").unwrap_err();

        assert!(error.to_string().contains("escapes repo root"));
    }

    #[test]
    fn load_snapshot_returns_exact_lines_and_reuses_cache() {
        let repo = temp_dir("cache");
        let file = repo.join("src.txt");
        fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();

        let store = LiveFileStore::new(4);
        let first = store.load_snapshot(&repo, "src.txt").unwrap();
        let second = store.load_snapshot(&repo, "src.txt").unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.slice_lines(2, 3), Some("beta\ngamma\n"));
    }

    #[test]
    fn literal_matches_support_case_insensitive_and_whole_word() {
        let repo = temp_dir("literal");
        let file = repo.join("src.txt");
        fs::write(&file, "alpha foo\nfoobar\nFoo\n").unwrap();

        let store = LiveFileStore::new(4);
        let snapshot = store.load_snapshot(&repo, "src.txt").unwrap();

        let whole = snapshot
            .find_literal_matches("foo", false, true, 10)
            .unwrap();
        let all = snapshot
            .find_literal_matches("foo", false, false, 10)
            .unwrap();

        assert_eq!(whole.len(), 2);
        assert_eq!(all.len(), 3);
        assert_eq!(whole[0].start_line, 1);
        assert_eq!(whole[1].start_line, 3);
    }

    #[test]
    fn load_snapshot_rejects_binary_files() {
        let repo = temp_dir("binary");
        let file = repo.join("data.bin");
        fs::write(&file, [0_u8, 159, 146, 150]).unwrap();

        let store = LiveFileStore::new(4);
        let error = store.load_snapshot(&repo, "data.bin").unwrap_err();

        assert!(error.to_string().contains("not a text file"));
    }
}
