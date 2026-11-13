use std::sync::{Arc, Mutex};

use aionui_ai_agent::agent_task::{AgentInstance, IAgentTask};
use aionui_ai_agent::protocol::events::FinishEventData;
use aionui_ai_agent::types::{BuildTaskOptions, SendMessageData};
use aionui_ai_agent::{AgentError, AgentSendError, AgentStreamEvent, IMockAgent, IWorkerTaskManager};
use aionui_api_types::WebSocketMessage;
use aionui_channel::channel_settings::ChannelSettingsService;
use aionui_channel::error::ChannelError;
use aionui_channel::message_service::ChannelMessageService;
use aionui_channel::types::PluginType;
use aionui_common::{AgentKillReason, AgentType, ConversationStatus, TimestampMs};
use aionui_conversation::ConversationService;
use aionui_conversation::skill_resolver::{ResolvedAgentSkill, SkillResolver};
use aionui_db::models::AssistantSessionRow;
use aionui_db::models::UpsertAssistantDefinitionParams;
use aionui_db::{
    IAcpSessionRepository, IAssistantDefinitionRepository, IClientPreferenceRepository, IConversationRepository,
    SqliteAcpSessionRepository, SqliteAgentMetadataRepository, SqliteAssistantDefinitionRepository,
    SqliteAssistantOverlayRepository, SqliteAssistantPreferenceRepository, SqliteClientPreferenceRepository,
    SqliteConversationRepository, init_database_memory,
};
use aionui_realtime::EventBroadcaster;
use async_trait::async_trait;
use tokio::sync::broadcast;

const TEST_OWNER_USER_ID: &str = "system_default_user";

struct TestBroadcaster {
    events: Mutex<Vec<WebSocketMessage<serde_json::Value>>>,
}

impl TestBroadcaster {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }
}

impl EventBroadcaster for TestBroadcaster {
    fn broadcast(&self, event: WebSocketMessage<serde_json::Value>) {
        self.events.lock().unwrap().push(event);
    }
}

struct NoopSkillResolver;

#[async_trait]
impl SkillResolver for NoopSkillResolver {
    async fn auto_inject_names(&self) -> Vec<String> {
        Vec::new()
    }

    async fn resolve_skills(&self, _names: &[String]) -> Vec<ResolvedAgentSkill> {
        Vec::new()
    }
}

struct ScriptedAgent {
    conversation_id: String,
    /// The workspace the agent was built with, reported back like a real agent's cwd.
    workspace: String,
    event_tx: broadcast::Sender<AgentStreamEvent>,
}

impl ScriptedAgent {
    fn new(conversation_id: &str, workspace: &str) -> Self {
        let (event_tx, _) = broadcast::channel(16);
        Self {
            conversation_id: conversation_id.to_owned(),
            workspace: workspace.to_owned(),
            event_tx,
        }
    }
}

#[async_trait]
impl IAgentTask for ScriptedAgent {
    fn agent_type(&self) -> AgentType {
        AgentType::Aionrs
    }

    fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn status(&self) -> Option<ConversationStatus> {
        Some(ConversationStatus::Finished)
    }

    fn last_activity_at(&self) -> TimestampMs {
        0
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentStreamEvent> {
        self.event_tx.subscribe()
    }

    async fn send_message(&self, _data: SendMessageData) -> Result<(), AgentSendError> {
        let _ = self.event_tx.send(AgentStreamEvent::Finish(FinishEventData::default()));
        Ok(())
    }

    async fn cancel(&self) -> Result<(), AgentError> {
        Ok(())
    }

    fn kill(&self, _reason: Option<AgentKillReason>) -> Result<(), AgentError> {
        Ok(())
    }
}

impl IMockAgent for ScriptedAgent {}

struct RecordingTaskManager {
    agents: Mutex<std::collections::HashMap<String, AgentInstance>>,
}

impl RecordingTaskManager {
    fn new() -> Self {
        Self {
            agents: Mutex::new(std::collections::HashMap::new()),
        }
    }
}

#[async_trait]
impl IWorkerTaskManager for RecordingTaskManager {
    fn get_task(&self, conversation_id: &str) -> Option<AgentInstance> {
        self.agents.lock().unwrap().get(conversation_id).cloned()
    }

    async fn get_or_build_task(
        &self,
        conversation_id: &str,
        options: BuildTaskOptions,
    ) -> Result<AgentInstance, AgentError> {
        let mut agents = self.agents.lock().unwrap();
        if let Some(agent) = agents.get(conversation_id) {
            return Ok(agent.clone());
        }

        let agent = AgentInstance::Mock(Arc::new(ScriptedAgent::new(
            conversation_id,
            &options.context.workspace.path,
        )));
        agents.insert(conversation_id.to_owned(), agent.clone());
        Ok(agent)
    }

    fn kill(&self, conversation_id: &str, _reason: Option<AgentKillReason>) -> Result<(), AgentError> {
        self.agents.lock().unwrap().remove(conversation_id);
        Ok(())
    }

    fn kill_and_wait(
        &self,
        conversation_id: &str,
        reason: Option<AgentKillReason>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        let _ = self.kill(conversation_id, reason);
        Box::pin(std::future::ready(()))
    }

    async fn clear(&self) {
        self.agents.lock().unwrap().clear();
    }

    fn active_count(&self) -> usize {
        self.agents.lock().unwrap().len()
    }

    fn collect_idle(&self, _idle_threshold_ms: TimestampMs) -> Vec<String> {
        Vec::new()
    }
}

fn bare_assistant_definition_params<'a>(
    definition_id: &'a str,
    assistant_id: &'a str,
    agent_id: &'a str,
) -> UpsertAssistantDefinitionParams<'a> {
    UpsertAssistantDefinitionParams {
        id: definition_id,
        assistant_id,
        source: "generated",
        owner_type: "system",
        source_ref: Some(assistant_id),
        name: assistant_id,
        name_i18n: "{}",
        description: Some("Channel bare assistant"),
        description_i18n: "{}",
        avatar_type: "emoji",
        avatar_value: Some("🤖"),
        agent_id,
        rule_resource_type: "user_file",
        rule_resource_ref: None,
        recommended_prompts: "[]",
        recommended_prompts_i18n: "{}",
        default_model_mode: "auto",
        default_model_value: None,
        default_permission_mode: "auto",
        default_permission_value: None,
        default_thought_level_mode: "auto",
        default_thought_level_value: None,
        default_skills_mode: "auto",
        default_skill_ids: "[]",
        custom_skill_names: "[]",
        default_disabled_builtin_skill_ids: "[]",
        default_mcps_mode: "auto",
        default_mcp_ids: "[]",
    }
}

#[tokio::test]
async fn send_to_agent_warms_cold_task_before_returning_stream_subscription() {
    let db = init_database_memory().await.unwrap();
    let pool = db.pool().clone();
    let work_root = tempfile::tempdir().unwrap();

    let task_manager: Arc<dyn IWorkerTaskManager> = Arc::new(RecordingTaskManager::new());
    let conversation_svc = Arc::new(ConversationService::new(
        work_root.path().to_path_buf(),
        Arc::new(TestBroadcaster::new()),
        Arc::new(NoopSkillResolver),
        Arc::clone(&task_manager),
        Arc::new(SqliteConversationRepository::new(pool.clone())),
        Arc::new(SqliteAgentMetadataRepository::new(pool.clone())),
        Arc::new(SqliteAcpSessionRepository::new(pool.clone())),
    ));

    let settings = Arc::new(ChannelSettingsService::new(Arc::new(
        SqliteClientPreferenceRepository::new(pool),
    )));
    let message_svc = ChannelMessageService::new(
        conversation_svc,
        Arc::clone(&task_manager),
        settings,
        work_root.path().to_path_buf(),
    );

    let session = AssistantSessionRow {
        id: "session-1".to_owned(),
        user_id: "channel-user-1".to_owned(),
        agent_type: "aionrs".to_owned(),
        conversation_id: None,
        workspace: None,
        chat_id: Some("7088048016".to_owned()),
        created_at: 1,
        last_activity: 1,
    };

    for platform in [
        PluginType::Telegram,
        PluginType::Lark,
        PluginType::Dingtalk,
        PluginType::Weixin,
    ] {
        let result = message_svc
            .send_to_agent(TEST_OWNER_USER_ID, &session, "hello", platform)
            .await
            .unwrap();

        assert!(
            result.stream_rx.is_some(),
            "channel relay must have an agent stream receiver after cold start for {platform:?}"
        );
        assert!(task_manager.get_task(&result.conversation_id).is_some());
    }
}

#[tokio::test]
async fn send_to_agent_persists_assistant_snapshot_for_channel_bound_assistant() {
    let db = init_database_memory().await.unwrap();
    let pool = db.pool().clone();
    let work_root = tempfile::tempdir().unwrap();

    let task_manager: Arc<dyn IWorkerTaskManager> = Arc::new(RecordingTaskManager::new());
    let conversation_repo = Arc::new(SqliteConversationRepository::new(pool.clone()));
    let conversation_repo_trait: Arc<dyn IConversationRepository> = conversation_repo.clone();
    let acp_session_repo = Arc::new(SqliteAcpSessionRepository::new(pool.clone()));
    let conversation_svc = Arc::new(ConversationService::new(
        work_root.path().to_path_buf(),
        Arc::new(TestBroadcaster::new()),
        Arc::new(NoopSkillResolver),
        Arc::clone(&task_manager),
        conversation_repo_trait,
        Arc::new(SqliteAgentMetadataRepository::new(pool.clone())),
        acp_session_repo.clone(),
    ));

    let pref_repo = Arc::new(SqliteClientPreferenceRepository::new(pool.clone()));
    let definition_repo = Arc::new(SqliteAssistantDefinitionRepository::new(pool.clone()));
    let overlay_repo = Arc::new(SqliteAssistantOverlayRepository::new(pool.clone()));
    let assistant_preference_repo = Arc::new(SqliteAssistantPreferenceRepository::new(pool.clone()));
    conversation_svc.with_assistant_definition_repo(definition_repo.clone());
    conversation_svc.with_assistant_state_repo(overlay_repo.clone());
    conversation_svc.with_assistant_preference_repo(assistant_preference_repo);
    definition_repo
        .upsert(&bare_assistant_definition_params(
            "asstdef-channel-claude",
            "bare-claude",
            "claude",
        ))
        .await
        .unwrap();
    pref_repo
        .upsert_batch(
            TEST_OWNER_USER_ID,
            &[(
                "assistant.telegram.agent",
                r#"{"assistant_id":"bare-claude","name":"Claude"}"#,
            )],
        )
        .await
        .unwrap();

    let settings = Arc::new(ChannelSettingsService::new(pref_repo).with_assistant_repos(definition_repo, overlay_repo));
    let message_svc = ChannelMessageService::new(
        conversation_svc,
        Arc::clone(&task_manager),
        settings,
        work_root.path().to_path_buf(),
    );

    let session = AssistantSessionRow {
        id: "session-assisted".to_owned(),
        user_id: "channel-user-1".to_owned(),
        agent_type: "aionrs".to_owned(),
        conversation_id: None,
        workspace: None,
        chat_id: Some("7088048016".to_owned()),
        created_at: 1,
        last_activity: 1,
    };

    let result = message_svc
        .send_to_agent(TEST_OWNER_USER_ID, &session, "hello", PluginType::Telegram)
        .await
        .unwrap();

    let snapshot = conversation_repo
        .get_assistant_snapshot(TEST_OWNER_USER_ID, &result.conversation_id)
        .await
        .unwrap();
    assert!(
        snapshot.is_some(),
        "channel-created conversation should persist an assistant snapshot when the platform is bound to an assistant"
    );
    let snapshot = snapshot.unwrap();
    let conversation = conversation_repo
        .get(TEST_OWNER_USER_ID, &result.conversation_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(conversation.r#type, AgentType::Acp.serde_name());
    let session_row = acp_session_repo
        .get_for_user(TEST_OWNER_USER_ID, &result.conversation_id)
        .await
        .unwrap()
        .expect("acp_session row should exist for ACP assistant conversations");
    assert_eq!(session_row.agent_id, "2d23ff1c");
    assert_eq!(snapshot.assistant_id, "bare-claude");
    assert_eq!(snapshot.agent_id, "2d23ff1c");
    assert_eq!(conversation.name, "Claude");
}

#[tokio::test]
async fn send_to_agent_rejects_unresolvable_channel_assistant_binding() {
    let db = init_database_memory().await.unwrap();
    let pool = db.pool().clone();
    let work_root = tempfile::tempdir().unwrap();

    let task_manager: Arc<dyn IWorkerTaskManager> = Arc::new(RecordingTaskManager::new());
    let conversation_repo = Arc::new(SqliteConversationRepository::new(pool.clone()));
    let conversation_repo_trait: Arc<dyn IConversationRepository> = conversation_repo.clone();
    let acp_session_repo = Arc::new(SqliteAcpSessionRepository::new(pool.clone()));
    let conversation_svc = Arc::new(ConversationService::new(
        work_root.path().to_path_buf(),
        Arc::new(TestBroadcaster::new()),
        Arc::new(NoopSkillResolver),
        Arc::clone(&task_manager),
        conversation_repo_trait,
        Arc::new(SqliteAgentMetadataRepository::new(pool.clone())),
        acp_session_repo,
    ));

    let pref_repo = Arc::new(SqliteClientPreferenceRepository::new(pool.clone()));
    pref_repo
        .upsert_batch(
            TEST_OWNER_USER_ID,
            &[(
                "assistant.telegram.agent",
                r#"{"assistant_id":"missing-assistant","name":"Missing"}"#,
            )],
        )
        .await
        .unwrap();
    let definition_repo = Arc::new(SqliteAssistantDefinitionRepository::new(pool.clone()));
    let overlay_repo = Arc::new(SqliteAssistantOverlayRepository::new(pool.clone()));
    let settings = Arc::new(ChannelSettingsService::new(pref_repo).with_assistant_repos(definition_repo, overlay_repo));
    let message_svc = ChannelMessageService::new(
        conversation_svc,
        Arc::clone(&task_manager),
        settings,
        work_root.path().to_path_buf(),
    );

    let session = AssistantSessionRow {
        id: "session-assisted-missing".to_owned(),
        user_id: "channel-user-missing".to_owned(),
        agent_type: "aionrs".to_owned(),
        conversation_id: None,
        workspace: None,
        chat_id: Some("7088048017".to_owned()),
        created_at: 1,
        last_activity: 1,
    };

    let err = message_svc
        .send_to_agent(TEST_OWNER_USER_ID, &session, "hello", PluginType::Telegram)
        .await
        .unwrap_err();
    assert!(matches!(err, ChannelError::MessageSendFailed(_)));
    assert!(
        err.to_string().contains("missing-assistant"),
        "error should surface the unresolved assistant identity"
    );
}

#[tokio::test]
async fn send_to_agent_without_saved_binding_defaults_to_bare_aionrs_assistant() {
    let db = init_database_memory().await.unwrap();
    let pool = db.pool().clone();
    let work_root = tempfile::tempdir().unwrap();

    let task_manager: Arc<dyn IWorkerTaskManager> = Arc::new(RecordingTaskManager::new());
    let conversation_repo = Arc::new(SqliteConversationRepository::new(pool.clone()));
    let conversation_repo_trait: Arc<dyn IConversationRepository> = conversation_repo.clone();
    let acp_session_repo = Arc::new(SqliteAcpSessionRepository::new(pool.clone()));
    let conversation_svc = Arc::new(ConversationService::new(
        work_root.path().to_path_buf(),
        Arc::new(TestBroadcaster::new()),
        Arc::new(NoopSkillResolver),
        Arc::clone(&task_manager),
        conversation_repo_trait,
        Arc::new(SqliteAgentMetadataRepository::new(pool.clone())),
        acp_session_repo,
    ));

    let pref_repo = Arc::new(SqliteClientPreferenceRepository::new(pool.clone()));
    let definition_repo = Arc::new(SqliteAssistantDefinitionRepository::new(pool.clone()));
    let overlay_repo = Arc::new(SqliteAssistantOverlayRepository::new(pool.clone()));
    let assistant_preference_repo = Arc::new(SqliteAssistantPreferenceRepository::new(pool.clone()));
    conversation_svc.with_assistant_definition_repo(definition_repo.clone());
    conversation_svc.with_assistant_state_repo(overlay_repo.clone());
    conversation_svc.with_assistant_preference_repo(assistant_preference_repo);
    definition_repo
        .upsert(&bare_assistant_definition_params(
            "asstdef-channel-aionrs",
            "bare-aionrs",
            "aionrs",
        ))
        .await
        .unwrap();

    let settings = Arc::new(ChannelSettingsService::new(pref_repo).with_assistant_repos(definition_repo, overlay_repo));
    let message_svc = ChannelMessageService::new(
        conversation_svc,
        Arc::clone(&task_manager),
        settings,
        work_root.path().to_path_buf(),
    );

    let session = AssistantSessionRow {
        id: "session-assisted-default-aionrs".to_owned(),
        user_id: "channel-user-default".to_owned(),
        agent_type: "aionrs".to_owned(),
        conversation_id: None,
        workspace: None,
        chat_id: Some("7088048018".to_owned()),
        created_at: 1,
        last_activity: 1,
    };

    let result = message_svc
        .send_to_agent(TEST_OWNER_USER_ID, &session, "hello", PluginType::Telegram)
        .await
        .unwrap();

    let snapshot = conversation_repo
        .get_assistant_snapshot(TEST_OWNER_USER_ID, &result.conversation_id)
        .await
        .unwrap()
        .expect("channel-created conversation should default to a bare assistant snapshot");
    let conversation = conversation_repo
        .get(TEST_OWNER_USER_ID, &result.conversation_id)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(snapshot.assistant_id, "bare-aionrs");
    assert_eq!(snapshot.agent_id, "632f31d2");
    assert_eq!(conversation.r#type, AgentType::Aionrs.serde_name());
    assert_eq!(conversation.name, "tg-aionrs-70880480");
}

#[tokio::test]
async fn send_to_agent_without_assistant_name_falls_back_to_legacy_channel_name() {
    let db = init_database_memory().await.unwrap();
    let pool = db.pool().clone();
    let work_root = tempfile::tempdir().unwrap();

    let task_manager: Arc<dyn IWorkerTaskManager> = Arc::new(RecordingTaskManager::new());
    let conversation_repo = Arc::new(SqliteConversationRepository::new(pool.clone()));
    let conversation_repo_trait: Arc<dyn IConversationRepository> = conversation_repo.clone();
    let acp_session_repo = Arc::new(SqliteAcpSessionRepository::new(pool.clone()));
    let conversation_svc = Arc::new(ConversationService::new(
        work_root.path().to_path_buf(),
        Arc::new(TestBroadcaster::new()),
        Arc::new(NoopSkillResolver),
        Arc::clone(&task_manager),
        conversation_repo_trait,
        Arc::new(SqliteAgentMetadataRepository::new(pool.clone())),
        acp_session_repo,
    ));

    let pref_repo = Arc::new(SqliteClientPreferenceRepository::new(pool.clone()));
    let definition_repo = Arc::new(SqliteAssistantDefinitionRepository::new(pool.clone()));
    let overlay_repo = Arc::new(SqliteAssistantOverlayRepository::new(pool.clone()));
    let assistant_preference_repo = Arc::new(SqliteAssistantPreferenceRepository::new(pool.clone()));
    conversation_svc.with_assistant_definition_repo(definition_repo.clone());
    conversation_svc.with_assistant_state_repo(overlay_repo.clone());
    conversation_svc.with_assistant_preference_repo(assistant_preference_repo);
    definition_repo
        .upsert(&bare_assistant_definition_params(
            "asstdef-channel-codex",
            "bare-codex",
            "codex",
        ))
        .await
        .unwrap();
    pref_repo
        .upsert_batch(
            TEST_OWNER_USER_ID,
            &[("assistant.telegram.agent", r#"{"assistant_id":"bare-codex"}"#)],
        )
        .await
        .unwrap();

    let settings = Arc::new(ChannelSettingsService::new(pref_repo).with_assistant_repos(definition_repo, overlay_repo));
    let message_svc = ChannelMessageService::new(
        conversation_svc,
        Arc::clone(&task_manager),
        settings,
        work_root.path().to_path_buf(),
    );

    let session = AssistantSessionRow {
        id: "session-assisted-fallback-name".to_owned(),
        user_id: "channel-user-2".to_owned(),
        agent_type: "aionrs".to_owned(),
        conversation_id: None,
        workspace: None,
        chat_id: Some("7088048016".to_owned()),
        created_at: 1,
        last_activity: 1,
    };

    let result = message_svc
        .send_to_agent(TEST_OWNER_USER_ID, &session, "hello", PluginType::Telegram)
        .await
        .unwrap();

    let conversation = conversation_repo
        .get(TEST_OWNER_USER_ID, &result.conversation_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(conversation.name, "tg-acp-codex-70880480");
}

// ── Per-channel shared workspace ───────────────────────────────────────

/// Bare-aionrs channel wiring with one shared work root for both services, as in production.
struct ChannelWorkspaceFixture {
    work_root: tempfile::TempDir,
    conversation_svc: Arc<ConversationService>,
    message_svc: ChannelMessageService,
}

impl ChannelWorkspaceFixture {
    async fn new() -> Self {
        let db = init_database_memory().await.unwrap();
        let pool = db.pool().clone();
        // Keep the in-memory pool alive for the whole test.
        std::mem::forget(db);
        let work_root = tempfile::tempdir().unwrap();

        let task_manager: Arc<dyn IWorkerTaskManager> = Arc::new(RecordingTaskManager::new());
        let conversation_repo: Arc<dyn IConversationRepository> =
            Arc::new(SqliteConversationRepository::new(pool.clone()));
        let conversation_svc = Arc::new(ConversationService::new(
            work_root.path().to_path_buf(),
            Arc::new(TestBroadcaster::new()),
            Arc::new(NoopSkillResolver),
            Arc::clone(&task_manager),
            conversation_repo,
            Arc::new(SqliteAgentMetadataRepository::new(pool.clone())),
            Arc::new(SqliteAcpSessionRepository::new(pool.clone())),
        ));

        let pref_repo = Arc::new(SqliteClientPreferenceRepository::new(pool.clone()));
        let definition_repo = Arc::new(SqliteAssistantDefinitionRepository::new(pool.clone()));
        let overlay_repo = Arc::new(SqliteAssistantOverlayRepository::new(pool.clone()));
        conversation_svc.with_assistant_definition_repo(definition_repo.clone());
        conversation_svc.with_assistant_state_repo(overlay_repo.clone());
        conversation_svc.with_assistant_preference_repo(Arc::new(SqliteAssistantPreferenceRepository::new(pool)));
        definition_repo
            .upsert(&bare_assistant_definition_params(
                "asstdef-channel-aionrs",
                "bare-aionrs",
                "aionrs",
            ))
            .await
            .unwrap();

        let settings =
            Arc::new(ChannelSettingsService::new(pref_repo).with_assistant_repos(definition_repo, overlay_repo));
        let message_svc = ChannelMessageService::new(
            Arc::clone(&conversation_svc),
            task_manager,
            settings,
            work_root.path().to_path_buf(),
        );
        Self {
            work_root,
            conversation_svc,
            message_svc,
        }
    }

    fn channel_dir(&self) -> std::path::PathBuf {
        self.work_root.path().join("channels").join("Lark-Intl")
    }

    async fn send(&self, session: &AssistantSessionRow) -> Result<String, ChannelError> {
        self.message_svc
            .send_to_agent(TEST_OWNER_USER_ID, session, "hello", PluginType::LarkIntl)
            .await
            .map(|result| result.conversation_id)
    }

    async fn extra(&self, conversation_id: &str) -> serde_json::Value {
        self.conversation_svc
            .get(TEST_OWNER_USER_ID, conversation_id)
            .await
            .unwrap()
            .extra
    }
}

fn channel_session(id: &str, conversation_id: Option<String>) -> AssistantSessionRow {
    AssistantSessionRow {
        id: id.to_owned(),
        user_id: "channel-user-intl".to_owned(),
        agent_type: "aionrs".to_owned(),
        conversation_id,
        workspace: None,
        chat_id: Some(format!("oc_{id}")),
        created_at: 1,
        last_activity: 1,
    }
}

#[tokio::test]
async fn channel_conversations_share_the_channel_workspace() {
    let fx = ChannelWorkspaceFixture::new().await;

    let first = fx.send(&channel_session("s1", None)).await.unwrap();
    let second = fx.send(&channel_session("s2", None)).await.unwrap();
    assert_ne!(first, second);

    let expected = fx.channel_dir().to_string_lossy().into_owned();
    for id in [&first, &second] {
        let extra = fx.extra(id).await;
        assert_eq!(extra["workspace"], expected.as_str());
        assert_eq!(extra["is_temporary_workspace"], false);
    }
    assert!(fx.channel_dir().is_dir());
    // A per-conversation temp workspace would live under `<root>/conversations/...`.
    assert!(!fx.work_root.path().join("conversations").exists());
}

#[tokio::test]
async fn deleting_a_channel_conversation_keeps_the_channel_workspace() {
    let fx = ChannelWorkspaceFixture::new().await;
    let first = fx.send(&channel_session("s1", None)).await.unwrap();
    let second = fx.send(&channel_session("s2", None)).await.unwrap();

    fx.conversation_svc.delete(TEST_OWNER_USER_ID, &first).await.unwrap();

    assert!(fx.channel_dir().is_dir());
    assert_eq!(
        fx.extra(&second).await["workspace"],
        fx.channel_dir().to_string_lossy().as_ref()
    );
}

#[tokio::test]
async fn channel_workspace_failure_falls_back_to_temp_workspace() {
    let fx = ChannelWorkspaceFixture::new().await;
    std::fs::write(fx.work_root.path().join("channels"), b"not a dir").unwrap();

    let id = fx.send(&channel_session("s1", None)).await.unwrap();

    assert_eq!(fx.extra(&id).await["is_temporary_workspace"], true);
}

#[tokio::test]
async fn bound_session_recreates_a_deleted_channel_workspace() {
    let fx = ChannelWorkspaceFixture::new().await;
    std::fs::create_dir_all(fx.channel_dir()).unwrap();
    // `create` never builds an agent, so this conversation's agent is cold.
    let req: aionui_api_types::CreateConversationRequest = serde_json::from_value(serde_json::json!({
        "type": "aionrs",
        "source": "lark",
        "extra": {
            "session_mode": "yolo",
            "workspace": fx.channel_dir().to_string_lossy(),
        }
    }))
    .unwrap();
    let conversation = fx.conversation_svc.create(TEST_OWNER_USER_ID, req).await.unwrap();
    std::fs::remove_dir_all(fx.channel_dir()).unwrap();

    // One send only: a second send on a warm agent would skip the workspace check.
    let id = fx
        .send(&channel_session("bound", Some(conversation.id.clone())))
        .await
        .expect("send must self-heal the channel dir before warmup");

    assert_eq!(id, conversation.id);
    assert!(fx.channel_dir().is_dir());
}
