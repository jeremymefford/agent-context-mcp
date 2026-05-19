use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Component, Path, PathBuf};

const DEFAULT_CONFIG_PATH: &str = "~/Library/Application Support/agent-context/config.toml";
const LEGACY_CONFIG_PATH: &str = "~/.config/agent-context/config.toml";
const DEFAULT_SNAPSHOT_PATH: &str =
    "~/Library/Application Support/agent-context/state/snapshot.json";
const LEGACY_SNAPSHOT_PATH: &str = "~/.context/agent-context-snapshot.json";
const DEFAULT_INDEX_ROOT: &str = "~/Library/Application Support/agent-context/index-v1";
const LEGACY_INDEX_ROOT: &str = "~/.context/agent-context-search-v1";
const DEFAULT_DEFAULT_GROUP: &str = "workspace";
const DEFAULT_VOYAGE_MODEL: &str = "voyage-code-3";
const DEFAULT_OPENAI_MODEL: &str = "text-embedding-3-small";
const DEFAULT_OLLAMA_MODEL: &str = "embeddinggemma";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MILVUS_DATABASE: &str = "";
const DEFAULT_EMBEDDING_PROFILE: &str = "default";

#[derive(Debug, Clone)]
pub struct Config {
    pub config_path: PathBuf,
    pub embedding: EmbeddingConfig,
    pub milvus: MilvusConfig,
    pub snapshot_path: PathBuf,
    pub index_root: PathBuf,
    pub merkle_dir: PathBuf,
    pub default_group: String,
    pub groups: Vec<GroupConfig>,
    pub freshness: FreshnessConfig,
    pub search: SearchConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingProvider {
    Voyage,
    OpenAi,
    Ollama,
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub default_profile: String,
    pub profiles: BTreeMap<String, EmbeddingProfileConfig>,
    pub assignments: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingProfileConfig {
    pub provider: EmbeddingProvider,
    pub model: String,
    pub voyage: VoyageProviderConfig,
    pub openai: OpenAiProviderConfig,
    pub ollama: OllamaProviderConfig,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingAssignment {
    pub repo: String,
    pub profile: String,
}

#[derive(Debug, Clone, Default)]
pub struct VoyageProviderConfig {
    pub api_key_env: String,
    pub key_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct OpenAiProviderConfig {
    pub api_key_env: String,
    pub key_file: Option<PathBuf>,
    pub base_url: String,
}

#[derive(Debug, Clone, Default)]
pub struct OllamaProviderConfig {
    pub base_url: String,
}

#[derive(Debug, Clone)]
pub struct MilvusConfig {
    pub address: String,
    pub token_env: Option<String>,
    pub token: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub database: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct FreshnessConfig {
    #[serde(default)]
    pub audit_interval_secs: Option<u64>,
    #[serde(default)]
    pub max_parallel_searches: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_max_concurrent_repo_searches")]
    pub max_concurrent_repo_searches: usize,
    #[serde(default = "default_max_concurrent_lexical_tasks")]
    pub max_concurrent_lexical_tasks: usize,
    #[serde(default = "default_max_concurrent_dense_tasks")]
    pub max_concurrent_dense_tasks: usize,
    #[serde(default = "default_max_warm_repos")]
    pub max_warm_repos: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: default_max_concurrent_requests(),
            max_concurrent_repo_searches: default_max_concurrent_repo_searches(),
            max_concurrent_lexical_tasks: default_max_concurrent_lexical_tasks(),
            max_concurrent_dense_tasks: default_max_concurrent_dense_tasks(),
            max_warm_repos: default_max_warm_repos(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GroupConfig {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    pub repos: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ScopeKind {
    Repo,
    Group,
}

#[derive(Debug, Clone)]
pub struct ResolvedScope {
    pub kind: ScopeKind,
    pub id: String,
    pub label: String,
    pub repos: Vec<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    embedding: Option<RawEmbeddingConfig>,
    #[serde(default)]
    voyage: Option<RawVoyageConfig>,
    #[serde(default)]
    milvus: Option<RawMilvusConfig>,
    #[serde(default)]
    freshness: Option<FreshnessConfig>,
    #[serde(default)]
    search: Option<SearchConfig>,
    #[serde(default)]
    snapshot_path: Option<String>,
    #[serde(default)]
    index_root: Option<String>,
    #[serde(default)]
    default_group: Option<String>,
    #[serde(default)]
    groups: Option<Vec<GroupConfig>>,
    #[serde(default)]
    voyage_key_file: Option<String>,
    #[serde(default)]
    embedding_model: Option<String>,
    #[serde(default)]
    milvus_address: Option<String>,
    #[serde(default)]
    milvus_token: Option<String>,
    #[serde(default)]
    milvus_username: Option<String>,
    #[serde(default)]
    milvus_password: Option<String>,
    #[serde(default)]
    milvus_database: Option<String>,
    #[serde(default)]
    projects: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Default)]
struct RawEmbeddingConfig {
    #[serde(default)]
    default_profile: Option<String>,
    #[serde(default)]
    profiles: Option<BTreeMap<String, RawEmbeddingProfileConfig>>,
    #[serde(default)]
    assignments: Option<Vec<RawEmbeddingAssignment>>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    voyage: Option<RawVoyageProviderConfig>,
    #[serde(default)]
    openai: Option<RawOpenAiProviderConfig>,
    #[serde(default)]
    ollama: Option<RawOllamaProviderConfig>,
}

#[derive(Debug, Deserialize, Default)]
struct RawEmbeddingProfileConfig {
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    voyage: Option<RawVoyageProviderConfig>,
    #[serde(default)]
    openai: Option<RawOpenAiProviderConfig>,
    #[serde(default)]
    ollama: Option<RawOllamaProviderConfig>,
}

#[derive(Debug, Deserialize, Clone)]
struct RawEmbeddingAssignment {
    repo: String,
    profile: String,
}

#[derive(Debug, Deserialize, Default)]
struct RawVoyageConfig {
    #[serde(default)]
    key_file: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RawVoyageProviderConfig {
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    key_file: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RawOpenAiProviderConfig {
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    key_file: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RawOllamaProviderConfig {
    #[serde(default)]
    base_url: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct RawMilvusConfig {
    #[serde(default)]
    address: Option<String>,
    #[serde(default)]
    token_env: Option<String>,
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    username: Option<String>,
    #[serde(default)]
    password: Option<String>,
    #[serde(default)]
    database: Option<String>,
}

impl Config {
    pub fn load() -> Result<Self> {
        for candidate in default_config_path_candidates()? {
            if candidate.exists() {
                return Self::load_from_path(&candidate);
            }
        }

        let default_path = expand_path(DEFAULT_CONFIG_PATH)?;
        bail!(
            "no agent-context config found at {}; run `agent-context init` to create one",
            default_path.display()
        );
    }

    pub fn load_from_path(config_path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading config at {}", config_path.display()))?;
        let raw: RawConfig = toml::from_str(&text).context("parsing config.toml")?;
        let config_dir = config_path.parent().unwrap_or_else(|| Path::new("/"));
        let legacy_config_path = expand_path(LEGACY_CONFIG_PATH)?;
        let legacy_snapshot_path = expand_path(LEGACY_SNAPSHOT_PATH)?;
        let default_snapshot_path = expand_path(DEFAULT_SNAPSHOT_PATH)?;
        let default_index_root = expand_path(DEFAULT_INDEX_ROOT)?;
        let legacy_index_root = expand_path(LEGACY_INDEX_ROOT)?;
        let using_legacy_layout = config_path == legacy_config_path;

        let milvus = MilvusConfig {
            address: raw
                .milvus
                .as_ref()
                .and_then(|value| value.address.clone())
                .or(raw.milvus_address.clone())
                .context("missing `milvus.address` or legacy `milvus_address` in config")?,
            token_env: raw
                .milvus
                .as_ref()
                .and_then(|value| value.token_env.clone()),
            token: raw
                .milvus
                .as_ref()
                .and_then(|value| value.token.clone())
                .or(raw.milvus_token.clone()),
            username: raw
                .milvus
                .as_ref()
                .and_then(|value| value.username.clone())
                .or(raw.milvus_username.clone()),
            password: raw
                .milvus
                .as_ref()
                .and_then(|value| value.password.clone())
                .or(raw.milvus_password.clone()),
            database: raw
                .milvus
                .as_ref()
                .and_then(|value| value.database.clone())
                .or(raw.milvus_database.clone())
                .unwrap_or_else(|| DEFAULT_MILVUS_DATABASE.to_string()),
        };

        let snapshot_path = if let Some(path) = raw.snapshot_path.as_deref() {
            expand_path_from(path, config_dir)?
        } else if using_legacy_layout {
            legacy_snapshot_path.clone()
        } else {
            default_snapshot_path
        };

        let explicit_index_root = raw
            .index_root
            .as_deref()
            .map(|path| expand_path_from(path, config_dir))
            .transpose()?;

        let index_root = if let Some(path) = explicit_index_root.clone() {
            path
        } else if using_legacy_layout || snapshot_path == legacy_snapshot_path {
            legacy_index_root
        } else {
            default_index_root
        };

        let merkle_dir = if explicit_index_root.is_some()
            || (!using_legacy_layout && snapshot_path != legacy_snapshot_path)
        {
            index_root.join("merkle")
        } else {
            snapshot_path
                .parent()
                .map(|parent| parent.join("merkle"))
                .unwrap_or_else(|| index_root.join("merkle"))
        };

        let freshness = raw.freshness.clone().unwrap_or_default();
        let search = build_search_config(raw.search.as_ref(), &freshness)?;

        let mut groups = if let Some(mut groups) = raw.groups.clone() {
            for group in &mut groups {
                group.repos = group
                    .repos
                    .iter()
                    .map(|repo| normalize_path_from(repo, config_dir))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .map(|repo| repo.display().to_string())
                    .collect();
            }
            groups
        } else {
            let repos = raw.projects.clone().unwrap_or_default();
            if repos.is_empty() {
                bail!("config defines no groups and no legacy `projects` list");
            }
            vec![GroupConfig {
                id: DEFAULT_DEFAULT_GROUP.to_string(),
                label: Some("Workspace".to_string()),
                repos: repos
                    .iter()
                    .map(|repo| normalize_path_from(repo, config_dir))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .map(|repo| repo.display().to_string())
                    .collect(),
            }]
        };

        if groups.is_empty() {
            bail!("config contains an empty `groups` list");
        }

        validate_groups(&groups)?;

        let default_group = raw.default_group.clone().unwrap_or_else(|| {
            groups
                .first()
                .map(|group| group.id.clone())
                .unwrap_or_else(|| DEFAULT_DEFAULT_GROUP.to_string())
        });

        if !groups.iter().any(|group| group.id == default_group) {
            bail!("default_group `{default_group}` does not match any configured group");
        }

        let embedding = build_embedding_config(&raw, config_dir, &groups)?;

        Ok(Self {
            config_path: config_path.to_path_buf(),
            embedding,
            milvus,
            snapshot_path,
            index_root,
            merkle_dir,
            default_group,
            groups: std::mem::take(&mut groups),
            freshness,
            search,
        })
    }

    pub fn default_config_path() -> Result<PathBuf> {
        expand_path(DEFAULT_CONFIG_PATH)
    }

    pub fn merkle_dir(&self) -> PathBuf {
        self.merkle_dir.clone()
    }

    pub fn search_index_dir(&self) -> PathBuf {
        self.index_root.clone()
    }

    pub fn symbol_db_path(&self) -> PathBuf {
        self.index_root.join("symbols.sqlite3")
    }

    pub fn all_repos(&self) -> Result<Vec<PathBuf>> {
        let mut repos = BTreeSet::new();
        for group in &self.groups {
            for repo in &group.repos {
                repos.insert(normalize_path(repo)?);
            }
        }
        Ok(repos.into_iter().collect())
    }

    pub fn resolve_scope(
        &self,
        scope: Option<&str>,
        path_alias: Option<&str>,
    ) -> Result<ResolvedScope> {
        self.resolve_scope_inner(scope, path_alias, false)
    }

    pub fn resolve_mcp_scope(
        &self,
        scope: Option<&str>,
        path_alias: Option<&str>,
    ) -> Result<ResolvedScope> {
        self.resolve_scope_inner(scope, path_alias, true)
    }

    fn resolve_scope_inner(
        &self,
        scope: Option<&str>,
        path_alias: Option<&str>,
        require_configured_repo: bool,
    ) -> Result<ResolvedScope> {
        let raw = scope.or(path_alias).unwrap_or(&self.default_group);
        if raw.starts_with('/') || raw.starts_with('~') {
            let repo = normalize_absolute_path(raw)?;
            if require_configured_repo && !self.is_configured_repo(&repo)? {
                bail!(
                    "repo scope `{raw}` is not configured; MCP only allows configured repo roots or group ids"
                );
            }
            return Ok(ResolvedScope {
                kind: ScopeKind::Repo,
                id: repo.display().to_string(),
                label: repo
                    .file_name()
                    .map(|value| value.to_string_lossy().to_string())
                    .unwrap_or_else(|| repo.display().to_string()),
                repos: vec![repo],
            });
        }

        if looks_like_relative_repo_path(raw) {
            bail!(
                "repo scope `{raw}` must be an absolute path or start with `~`; relative repo paths are not supported"
            );
        }

        let group = self
            .groups
            .iter()
            .find(|group| group.id == raw)
            .with_context(|| format!("unknown scope `{raw}`"))?;

        Ok(ResolvedScope {
            kind: ScopeKind::Group,
            id: group.id.clone(),
            label: group.label.clone().unwrap_or_else(|| group.id.clone()),
            repos: group
                .repos
                .iter()
                .map(|repo| normalize_path(repo))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn is_configured_repo(&self, repo: &Path) -> Result<bool> {
        Ok(self
            .all_repos()?
            .iter()
            .any(|configured| configured == repo))
    }
}

impl EmbeddingConfig {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn default_profile_name(&self) -> &str {
        &self.default_profile
    }

    pub fn profiles(&self) -> impl ExactSizeIterator<Item = (&String, &EmbeddingProfileConfig)> {
        self.profiles.iter()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn profile(&self, name: &str) -> Result<&EmbeddingProfileConfig> {
        self.profiles
            .get(name)
            .with_context(|| format!("unknown embedding profile `{name}`"))
    }

    pub fn profile_name_for_repo<'a>(&'a self, repo: &Path) -> Result<&'a str> {
        let repo_key = clean_path(repo).display().to_string();
        Ok(self
            .assignments
            .get(&repo_key)
            .map(String::as_str)
            .unwrap_or(self.default_profile.as_str()))
    }
}

impl EmbeddingProfileConfig {
    pub fn provider_name(&self) -> &'static str {
        match self.provider {
            EmbeddingProvider::Voyage => "voyage",
            EmbeddingProvider::OpenAi => "openai",
            EmbeddingProvider::Ollama => "ollama",
        }
    }

    pub fn default_model_for(provider: &EmbeddingProvider) -> &'static str {
        match provider {
            EmbeddingProvider::Voyage => DEFAULT_VOYAGE_MODEL,
            EmbeddingProvider::OpenAi => DEFAULT_OPENAI_MODEL,
            EmbeddingProvider::Ollama => DEFAULT_OLLAMA_MODEL,
        }
    }

    pub fn api_key(&self) -> Result<Option<String>> {
        match self.provider {
            EmbeddingProvider::Voyage => self
                .read_env_or_file(
                    &self.voyage.api_key_env,
                    self.voyage.key_file.as_deref(),
                    "Voyage",
                )
                .map(Some),
            EmbeddingProvider::OpenAi => self
                .read_env_or_file(
                    &self.openai.api_key_env,
                    self.openai.key_file.as_deref(),
                    "OpenAI",
                )
                .map(Some),
            EmbeddingProvider::Ollama => Ok(None),
        }
    }

    fn read_env_or_file(
        &self,
        env_var: &str,
        key_file: Option<&Path>,
        provider_name: &str,
    ) -> Result<String> {
        if !env_var.trim().is_empty()
            && let Ok(value) = std::env::var(env_var)
        {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Ok(value);
            }
        }

        if let Some(path) = key_file {
            validate_secret_file(path, provider_name)?;
            let key = std::fs::read_to_string(path)
                .with_context(|| format!("reading {provider_name} key at {}", path.display()))?;
            let key = key.trim().to_string();
            if !key.is_empty() {
                return Ok(key);
            }
            bail!("{provider_name} key file is empty: {}", path.display());
        }

        if env_var.trim().is_empty() {
            bail!("{provider_name} requires an API key, but no API key source is configured");
        }

        bail!("{provider_name} API key is not available; set {env_var} or configure a key file");
    }
}

impl MilvusConfig {
    pub fn token(&self) -> Option<String> {
        if let Some(env_var) = &self.token_env
            && let Ok(value) = std::env::var(env_var)
        {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
        self.token.clone().filter(|value| !value.trim().is_empty())
    }
}

fn build_embedding_config(
    raw: &RawConfig,
    config_dir: &Path,
    groups: &[GroupConfig],
) -> Result<EmbeddingConfig> {
    let configured_repos = groups
        .iter()
        .flat_map(|group| group.repos.iter().cloned())
        .collect::<BTreeSet<_>>();
    let raw_embedding = raw.embedding.as_ref();
    let raw_profiles = raw_embedding.and_then(|value| value.profiles.as_ref());

    if let Some(raw_profiles) = raw_profiles {
        if raw_profiles.is_empty() {
            bail!("embedding.profiles must not be empty");
        }

        let mut profiles = BTreeMap::new();
        for (name, profile) in raw_profiles {
            validate_embedding_profile_name(name)?;
            let rendered = build_embedding_profile_config(profile, config_dir)?;
            profiles.insert(name.clone(), rendered);
        }

        let default_profile = raw_embedding
            .and_then(|value| value.default_profile.clone())
            .or_else(|| {
                (profiles.len() == 1)
                    .then(|| profiles.keys().next().cloned())
                    .flatten()
            })
            .unwrap_or_else(|| DEFAULT_EMBEDDING_PROFILE.to_string());

        if !profiles.contains_key(&default_profile) {
            bail!(
                "embedding.default_profile `{default_profile}` does not match any configured embedding profile"
            );
        }

        let mut assignments = BTreeMap::new();
        if let Some(raw_assignments) = raw_embedding.and_then(|value| value.assignments.as_ref()) {
            for assignment in raw_assignments {
                let profile = assignment.profile.trim();
                if profile.is_empty() {
                    bail!("embedding assignment profile must not be empty");
                }
                if !profiles.contains_key(profile) {
                    bail!(
                        "embedding assignment for `{}` references unknown profile `{profile}`",
                        assignment.repo
                    );
                }
                let repo = normalize_path_from(&assignment.repo, config_dir)?
                    .display()
                    .to_string();
                if !configured_repos.contains(&repo) {
                    bail!(
                        "embedding assignment repo `{repo}` is not present in any configured group"
                    );
                }
                if assignments
                    .insert(repo.clone(), profile.to_string())
                    .is_some()
                {
                    bail!("duplicate embedding assignment for repo `{repo}`");
                }
            }
        }

        return Ok(EmbeddingConfig {
            default_profile,
            profiles,
            assignments,
        });
    }

    let legacy_profile = build_legacy_embedding_profile(raw, config_dir)?;
    let profiles = BTreeMap::from([(DEFAULT_EMBEDDING_PROFILE.to_string(), legacy_profile)]);
    Ok(EmbeddingConfig {
        default_profile: DEFAULT_EMBEDDING_PROFILE.to_string(),
        profiles,
        assignments: BTreeMap::new(),
    })
}

fn build_legacy_embedding_profile(
    raw: &RawConfig,
    config_dir: &Path,
) -> Result<EmbeddingProfileConfig> {
    let provider = parse_provider(
        raw.embedding
            .as_ref()
            .and_then(|value| value.provider.as_deref())
            .unwrap_or("voyage"),
    )?;
    let model = raw
        .embedding
        .as_ref()
        .and_then(|value| value.model.clone())
        .or(raw.voyage.as_ref().and_then(|value| value.model.clone()))
        .or(raw.embedding_model.clone())
        .unwrap_or_else(|| EmbeddingProfileConfig::default_model_for(&provider).to_string());
    let voyage = VoyageProviderConfig {
        api_key_env: raw
            .embedding
            .as_ref()
            .and_then(|value| value.voyage.as_ref())
            .and_then(|value| value.api_key_env.clone())
            .unwrap_or_else(|| "VOYAGE_API_KEY".to_string()),
        key_file: raw
            .embedding
            .as_ref()
            .and_then(|value| value.voyage.as_ref())
            .and_then(|value| value.key_file.as_deref())
            .or(raw
                .voyage
                .as_ref()
                .and_then(|value| value.key_file.as_deref()))
            .or(raw.voyage_key_file.as_deref())
            .map(|path| expand_path_from(path, config_dir))
            .transpose()?,
    };
    let openai = OpenAiProviderConfig {
        api_key_env: raw
            .embedding
            .as_ref()
            .and_then(|value| value.openai.as_ref())
            .and_then(|value| value.api_key_env.clone())
            .unwrap_or_else(|| "OPENAI_API_KEY".to_string()),
        key_file: raw
            .embedding
            .as_ref()
            .and_then(|value| value.openai.as_ref())
            .and_then(|value| value.key_file.as_deref())
            .map(|path| expand_path_from(path, config_dir))
            .transpose()?,
        base_url: raw
            .embedding
            .as_ref()
            .and_then(|value| value.openai.as_ref())
            .and_then(|value| value.base_url.clone())
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_string()),
    };
    let ollama = OllamaProviderConfig {
        base_url: raw
            .embedding
            .as_ref()
            .and_then(|value| value.ollama.as_ref())
            .and_then(|value| value.base_url.clone())
            .unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string()),
    };

    Ok(EmbeddingProfileConfig {
        provider,
        model,
        voyage,
        openai,
        ollama,
    })
}

fn build_embedding_profile_config(
    raw: &RawEmbeddingProfileConfig,
    config_dir: &Path,
) -> Result<EmbeddingProfileConfig> {
    let provider = parse_provider(
        raw.provider
            .as_deref()
            .context("embedding profile is missing `provider`")?,
    )?;
    let model = raw
        .model
        .clone()
        .unwrap_or_else(|| EmbeddingProfileConfig::default_model_for(&provider).to_string());
    let voyage = VoyageProviderConfig {
        api_key_env: raw
            .voyage
            .as_ref()
            .and_then(|value| value.api_key_env.clone())
            .unwrap_or_else(|| "VOYAGE_API_KEY".to_string()),
        key_file: raw
            .voyage
            .as_ref()
            .and_then(|value| value.key_file.as_deref())
            .map(|path| expand_path_from(path, config_dir))
            .transpose()?,
    };
    let openai = OpenAiProviderConfig {
        api_key_env: raw
            .openai
            .as_ref()
            .and_then(|value| value.api_key_env.clone())
            .unwrap_or_else(|| "OPENAI_API_KEY".to_string()),
        key_file: raw
            .openai
            .as_ref()
            .and_then(|value| value.key_file.as_deref())
            .map(|path| expand_path_from(path, config_dir))
            .transpose()?,
        base_url: raw
            .openai
            .as_ref()
            .and_then(|value| value.base_url.clone())
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_string()),
    };
    let ollama = OllamaProviderConfig {
        base_url: raw
            .ollama
            .as_ref()
            .and_then(|value| value.base_url.clone())
            .unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string()),
    };

    Ok(EmbeddingProfileConfig {
        provider,
        model,
        voyage,
        openai,
        ollama,
    })
}

fn validate_embedding_profile_name(name: &str) -> Result<()> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        bail!("embedding profile names must not be empty");
    }
    if trimmed.starts_with('.')
        || trimmed.starts_with('/')
        || trimmed.starts_with('~')
        || trimmed.contains('/')
        || trimmed.contains('\\')
    {
        bail!(
            "embedding profile `{name}` is invalid; use a simple identifier without path-like prefixes or separators"
        );
    }
    Ok(())
}

fn parse_provider(value: &str) -> Result<EmbeddingProvider> {
    match value.trim().to_ascii_lowercase().as_str() {
        "voyage" => Ok(EmbeddingProvider::Voyage),
        "openai" => Ok(EmbeddingProvider::OpenAi),
        "ollama" => Ok(EmbeddingProvider::Ollama),
        other => {
            bail!("unsupported embedding provider `{other}`; expected voyage, openai, or ollama")
        }
    }
}

fn default_config_path_candidates() -> Result<Vec<PathBuf>> {
    Ok(vec![
        expand_path(DEFAULT_CONFIG_PATH)?,
        expand_path(LEGACY_CONFIG_PATH)?,
    ])
}

fn build_search_config(
    raw: Option<&SearchConfig>,
    freshness: &FreshnessConfig,
) -> Result<SearchConfig> {
    let mut search = raw.cloned().unwrap_or_default();
    if raw.is_none()
        && let Some(legacy_parallelism) = freshness.max_parallel_searches
    {
        search.max_concurrent_repo_searches = legacy_parallelism;
    }
    validate_search_config(&search)?;
    Ok(search)
}

fn validate_search_config(search: &SearchConfig) -> Result<()> {
    for (name, value) in [
        (
            "search.max_concurrent_requests",
            search.max_concurrent_requests,
        ),
        (
            "search.max_concurrent_repo_searches",
            search.max_concurrent_repo_searches,
        ),
        (
            "search.max_concurrent_lexical_tasks",
            search.max_concurrent_lexical_tasks,
        ),
        (
            "search.max_concurrent_dense_tasks",
            search.max_concurrent_dense_tasks,
        ),
        ("search.max_warm_repos", search.max_warm_repos),
    ] {
        if value == 0 {
            bail!("{name} must be greater than 0");
        }
    }
    Ok(())
}

fn default_max_concurrent_requests() -> usize {
    2
}

fn default_max_concurrent_repo_searches() -> usize {
    4
}

fn default_max_concurrent_lexical_tasks() -> usize {
    2
}

fn default_max_concurrent_dense_tasks() -> usize {
    2
}

fn default_max_warm_repos() -> usize {
    4
}

fn expand_path(path: &str) -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("reading current directory")?;
    expand_path_from(path, &cwd)
}

pub fn normalize_path(path: &str) -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("reading current directory")?;
    normalize_path_from(path, &cwd)
}

fn clean_path(path: &Path) -> PathBuf {
    let mut cleaned = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                cleaned.pop();
            }
            _ => cleaned.push(component),
        }
    }
    cleaned
}

fn expand_path_from(path: &str, base_dir: &Path) -> Result<PathBuf> {
    normalize_path_from(path, base_dir)
}

fn normalize_path_from(path: &str, base_dir: &Path) -> Result<PathBuf> {
    let expanded = shellexpand::tilde(path).into_owned();
    let input = Path::new(&expanded);
    let absolute = if input.is_absolute() {
        input.to_path_buf()
    } else {
        base_dir.join(input)
    };
    Ok(clean_path(&absolute))
}

fn normalize_absolute_path(path: &str) -> Result<PathBuf> {
    let expanded = shellexpand::tilde(path).into_owned();
    let input = Path::new(&expanded);
    if !input.is_absolute() {
        bail!(
            "repo path `{path}` must be absolute or start with `~`; relative repo paths are not supported"
        );
    }
    Ok(clean_path(input))
}

fn validate_secret_file(path: &Path, provider_name: &str) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("reading {provider_name} key metadata at {}", path.display()))?;
    if !metadata.is_file() {
        bail!(
            "{provider_name} key file is not a regular file: {}",
            path.display()
        );
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let symlink_metadata = std::fs::symlink_metadata(path).with_context(|| {
            format!(
                "reading {provider_name} key file metadata at {}",
                path.display()
            )
        })?;
        if symlink_metadata.file_type().is_symlink() {
            bail!(
                "{provider_name} key file must not be a symlink: {}",
                path.display()
            );
        }

        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            bail!(
                "{provider_name} key file permissions are too open at {}; expected 600 or stricter",
                path.display()
            );
        }

        if metadata.uid() != unsafe { libc::geteuid() } {
            bail!(
                "{provider_name} key file must be owned by the current user: {}",
                path.display()
            );
        }
    }

    Ok(())
}

fn looks_like_relative_repo_path(raw: &str) -> bool {
    raw.starts_with('.')
        || raw.contains('/')
        || raw.contains('\\')
        || raw == "~"
        || !raw.starts_with("~/") && raw.starts_with('~')
}

fn validate_groups(groups: &[GroupConfig]) -> Result<()> {
    let mut seen_ids = HashSet::new();
    for group in groups {
        let id = group.id.trim();
        if id.is_empty() {
            bail!("group ids must not be empty");
        }
        if id.starts_with('.')
            || id.starts_with('/')
            || id.starts_with('~')
            || id.contains('/')
            || id.contains('\\')
        {
            bail!(
                "group id `{}` is invalid; use a simple identifier without path-like prefixes or separators",
                group.id
            );
        }
        if !seen_ids.insert(group.id.clone()) {
            bail!("duplicate group id `{}`", group.id);
        }
        if group.repos.is_empty() {
            bail!("group `{}` must contain at least one repo", group.id);
        }

        let mut seen_repos = HashSet::new();
        for repo in &group.repos {
            if !seen_repos.insert(repo.clone()) {
                bail!("group `{}` contains duplicate repo `{repo}`", group.id);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Config, EmbeddingProvider, normalize_absolute_path};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("agent-context-config-{prefix}-{nanos}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_config(dir: &Path, text: &str) -> PathBuf {
        let path = dir.join("config.toml");
        fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn config_paths_resolve_relative_to_config_file() {
        let root = temp_dir("relative");
        let config_dir = root.join("nested");
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = write_config(
            &config_dir,
            r#"
                snapshot_path = "../state/snapshot.json"
                index_root = "../state/index-v1"
                default_group = "workspace"

                [embedding]
                provider = "voyage"

                [embedding.voyage]
                key_file = "../keys/voyage_key"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["../repos/app"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        let profile = config
            .embedding
            .profile(config.embedding.default_profile_name())
            .unwrap();

        assert_eq!(profile.voyage.key_file, Some(root.join("keys/voyage_key")));
        assert_eq!(config.snapshot_path, root.join("state/snapshot.json"));
        assert_eq!(config.index_root, root.join("state/index-v1"));
        assert_eq!(config.merkle_dir, root.join("state/index-v1/merkle"));
        assert_eq!(
            config.groups[0].repos,
            vec![root.join("repos/app").display().to_string()]
        );
    }

    #[test]
    fn parses_openai_provider_block() {
        let dir = temp_dir("openai");
        let keys_dir = dir.join("keys");
        fs::create_dir_all(&keys_dir).unwrap();
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "openai"
                model = "text-embedding-3-large"

                [embedding.openai]
                api_key_env = "OPENAI_SECRET"
                key_file = "./keys/openai_key"
                base_url = "https://example.com/v1"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        let profile = config
            .embedding
            .profile(config.embedding.default_profile_name())
            .unwrap();
        assert_eq!(profile.provider, EmbeddingProvider::OpenAi);
        assert_eq!(profile.model, "text-embedding-3-large");
        assert_eq!(profile.openai.api_key_env, "OPENAI_SECRET");
        assert_eq!(profile.openai.key_file, Some(dir.join("keys/openai_key")));
        assert_eq!(profile.openai.base_url, "https://example.com/v1");
    }

    #[test]
    fn parses_ollama_provider_block() {
        let dir = temp_dir("ollama");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "ollama"
                model = "qwen3-embedding"

                [embedding.ollama]
                base_url = "http://localhost:11434"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        let profile = config
            .embedding
            .profile(config.embedding.default_profile_name())
            .unwrap();
        assert_eq!(profile.provider, EmbeddingProvider::Ollama);
        assert_eq!(profile.ollama.base_url, "http://localhost:11434");
    }

    #[test]
    fn parses_named_embedding_profiles_and_assignments() {
        let dir = temp_dir("named-profiles");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                default_profile = "hosted"

                [embedding.profiles.hosted]
                provider = "voyage"
                model = "voyage-code-3"

                [embedding.profiles.local]
                provider = "ollama"
                model = "nomic-embed"

                [embedding.profiles.local.ollama]
                base_url = "http://127.0.0.1:11435"

                [[embedding.assignments]]
                repo = "./repos/local-app"
                profile = "local"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["./repos/hosted-app", "./repos/local-app"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        let hosted_repo = dir.join("repos/hosted-app");
        let local_repo = dir.join("repos/local-app");

        assert_eq!(config.embedding.default_profile_name(), "hosted");
        assert_eq!(
            config
                .embedding
                .profile_name_for_repo(&hosted_repo)
                .unwrap(),
            "hosted"
        );
        assert_eq!(
            config.embedding.profile_name_for_repo(&local_repo).unwrap(),
            "local"
        );
        assert_eq!(
            config.embedding.profile("local").unwrap().ollama.base_url,
            "http://127.0.0.1:11435"
        );
    }

    #[test]
    fn legacy_embedding_config_normalizes_to_default_profile() {
        let dir = temp_dir("legacy-profile");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "ollama"
                model = "embeddinggemma"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        assert_eq!(config.embedding.default_profile_name(), "default");
        assert_eq!(config.embedding.profiles().len(), 1);
        let profile = config.embedding.profile("default").unwrap();
        assert_eq!(profile.provider, EmbeddingProvider::Ollama);
        assert_eq!(profile.model, "embeddinggemma");
    }

    #[test]
    fn rejects_assignment_to_unknown_profile() {
        let dir = temp_dir("unknown-assignment-profile");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                default_profile = "hosted"

                [embedding.profiles.hosted]
                provider = "voyage"

                [[embedding.assignments]]
                repo = "/tmp/a"
                profile = "missing"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let error = Config::load_from_path(&config_path).unwrap_err();
        assert!(error.to_string().contains("unknown profile"));
    }

    #[test]
    fn rejects_assignment_to_unconfigured_repo() {
        let dir = temp_dir("unknown-assignment-repo");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                default_profile = "hosted"

                [embedding.profiles.hosted]
                provider = "voyage"

                [[embedding.assignments]]
                repo = "/tmp/not-configured"
                profile = "hosted"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let error = Config::load_from_path(&config_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("is not present in any configured group")
        );
    }

    #[test]
    fn rejects_duplicate_embedding_assignments() {
        let dir = temp_dir("duplicate-assignments");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                default_profile = "hosted"

                [embedding.profiles.hosted]
                provider = "voyage"

                [[embedding.assignments]]
                repo = "/tmp/a"
                profile = "hosted"

                [[embedding.assignments]]
                repo = "/tmp/a"
                profile = "hosted"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let error = Config::load_from_path(&config_path).unwrap_err();
        assert!(error.to_string().contains("duplicate embedding assignment"));
    }

    #[test]
    fn search_config_defaults_are_applied_when_block_is_absent() {
        let dir = temp_dir("search-defaults");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "voyage"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        assert_eq!(config.search.max_concurrent_requests, 2);
        assert_eq!(config.search.max_concurrent_repo_searches, 4);
        assert_eq!(config.search.max_concurrent_lexical_tasks, 2);
        assert_eq!(config.search.max_concurrent_dense_tasks, 2);
        assert_eq!(config.search.max_warm_repos, 4);
    }

    #[test]
    fn legacy_freshness_parallelism_maps_to_search_repo_budget() {
        let dir = temp_dir("legacy-search-alias");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "voyage"

                [milvus]
                address = "localhost:19530"

                [freshness]
                max_parallel_searches = 7

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        assert_eq!(config.search.max_concurrent_repo_searches, 7);
    }

    #[test]
    fn search_block_takes_precedence_over_legacy_parallelism_alias() {
        let dir = temp_dir("search-precedence");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "voyage"

                [milvus]
                address = "localhost:19530"

                [freshness]
                max_parallel_searches = 7

                [search]
                max_concurrent_repo_searches = 3

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        assert_eq!(config.search.max_concurrent_repo_searches, 3);
    }

    #[test]
    fn invalid_search_limits_are_rejected() {
        let dir = temp_dir("invalid-search");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "voyage"

                [milvus]
                address = "localhost:19530"

                [search]
                max_concurrent_requests = 0

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
        );

        let error = Config::load_from_path(&config_path).unwrap_err();
        assert!(error.to_string().contains("search.max_concurrent_requests"));
    }

    #[test]
    fn duplicate_group_ids_are_rejected() {
        let dir = temp_dir("duplicate-group");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "voyage"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]

                [[groups]]
                id = "workspace"
                repos = ["/tmp/b"]
            "#,
        );

        let error = Config::load_from_path(&config_path).unwrap_err();
        assert!(error.to_string().contains("duplicate group id"));
    }

    #[test]
    fn empty_group_repo_list_is_rejected() {
        let dir = temp_dir("empty-group");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "voyage"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = []
            "#,
        );

        let error = Config::load_from_path(&config_path).unwrap_err();
        assert!(error.to_string().contains("must contain at least one repo"));
    }

    #[test]
    fn relative_repo_scope_is_rejected() {
        let path = normalize_absolute_path("./repo").unwrap_err();
        assert!(
            path.to_string()
                .contains("must be absolute or start with `~`")
        );
    }

    #[test]
    fn mcp_scope_rejects_unconfigured_absolute_repo_paths() {
        let dir = temp_dir("mcp-repo-scope");
        let config_path = write_config(
            &dir,
            r#"
                [embedding]
                provider = "voyage"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/configured-repo"]
            "#,
        );

        let config = Config::load_from_path(&config_path).unwrap();
        let error = config
            .resolve_mcp_scope(None, Some("/tmp/unconfigured-repo"))
            .unwrap_err();

        assert!(error.to_string().contains("is not configured"));
        assert!(
            config
                .resolve_mcp_scope(None, Some("/tmp/configured-repo"))
                .is_ok()
        );
    }

    #[cfg(unix)]
    #[test]
    fn openai_key_file_rejects_insecure_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = temp_dir("openai-key-perms");
        let key_path = dir.join("openai_key");
        fs::write(&key_path, "secret").unwrap();
        fs::set_permissions(&key_path, fs::Permissions::from_mode(0o644)).unwrap();

        let config_path = write_config(
            &dir,
            &format!(
                r#"
                [embedding]
                provider = "openai"

                [embedding.openai]
                api_key_env = "THIS_ENV_SHOULD_NOT_EXIST_AGENT_CONTEXT"
                key_file = "{}"
                base_url = "https://api.openai.com/v1"

                [milvus]
                address = "localhost:19530"

                [[groups]]
                id = "workspace"
                repos = ["/tmp/a"]
            "#,
                key_path.display()
            ),
        );

        let config = Config::load_from_path(&config_path).unwrap();
        let profile = config.embedding.profile("default").unwrap();
        let error = profile.api_key().unwrap_err();
        assert!(error.to_string().contains("permissions are too open"));
    }
}
