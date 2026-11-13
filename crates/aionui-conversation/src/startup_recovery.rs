use aionui_common::ErrorChain;
use aionui_db::models::MessageRow;
use aionui_db::{MessagePageDirection, MessagePageParams, MessageRowUpdate};
use tracing::{info, warn};

use crate::runtime_persistence::RuntimeWriteKind;
use crate::service::ConversationService;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartupRecoveryAction {
    FinishVisibleOutput,
    FinishEmptyPlaceholder,
    /// A tool_call/tool_group row left on `work` by a previous process. At
    /// startup this is dead BY DEFINITION — the CLI process group died with the
    /// old aioncore, so no terminal will ever be reported (a hard kill runs
    /// neither the pump's teardown settle nor leaves anyone to resume). The
    /// row's CONTENT status must be rewritten too: the frontend renders the
    /// embedded status, and `hasRunningToolMessages` would keep the View Steps
    /// spinner alive off a `finish` row whose content still says running.
    SettleToolRow,
    /// A `tool_call` row for an AskUserQuestion the user never answered. Unlike
    /// a dead Bash/edit call it is NOT dead by definition: the question is still
    /// askable — the frontend derives a question card from this row and resends
    /// the answer as a message (the CLI's own resume marks the dangling tool_use
    /// as interrupted and reads the message). Settling it to `canceled` would
    /// erase the only record that the user still owes an answer (live
    /// 2026-10-12: every restart turned the stuck question into a dead row).
    /// Left exactly as persisted.
    KeepPendingUserPrompt,
}

impl ConversationService {
    pub async fn recover_stale_runtime_state_on_startup(&self) {
        let rows = match self.conversation_repo().list_stale_runtime_messages().await {
            Ok(rows) => rows,
            Err(error) => {
                warn!(
                    error = %ErrorChain(&error),
                    "startup recovery skipped because stale runtime message query failed"
                );
                return;
            }
        };

        let mut recovered = 0usize;
        for stale in rows {
            let row = stale.message;
            if !self
                .runtime_persistence()
                .allows(&row.conversation_id, RuntimeWriteKind::StartupRecovery)
            {
                continue;
            }

            let mut action = classify_recovery_action(&row);
            if action == StartupRecoveryAction::KeepPendingUserPrompt
                && self.has_user_message_after(&stale.user_id, &row).await
            {
                // The user already moved on (answered by message / asked something
                // else): the prompt is no longer owed, so settle it like any other
                // dead tool row — otherwise the View Steps spinner keyed off this
                // row would survive every restart.
                action = StartupRecoveryAction::SettleToolRow;
            }
            if action == StartupRecoveryAction::KeepPendingUserPrompt {
                info!(
                    conversation_id = %row.conversation_id,
                    msg_id = ?row.msg_id,
                    message_type = %row.r#type,
                    recovery_action = ?action,
                    "startup recovery kept a pending user prompt"
                );
                continue;
            }
            let content = match action {
                StartupRecoveryAction::SettleToolRow => settle_tool_row_content(&row.r#type, &row.content),
                _ => None,
            };
            let update = MessageRowUpdate {
                content,
                status: Some(Some("finish".to_owned())),
                hidden: Some(matches!(action, StartupRecoveryAction::FinishEmptyPlaceholder)),
            };

            match self
                .conversation_repo()
                .update_message(&stale.user_id, &row.conversation_id, &row.id, &update)
                .await
            {
                Ok(()) => {
                    recovered += 1;
                    info!(
                        conversation_id = %row.conversation_id,
                        msg_id = ?row.msg_id,
                        message_type = %row.r#type,
                        recovery_action = ?action,
                        "startup recovery closed stale runtime message"
                    );
                }
                Err(error) => {
                    warn!(
                        conversation_id = %row.conversation_id,
                        msg_id = ?row.msg_id,
                        error = %ErrorChain(&error),
                        "startup recovery failed to close stale runtime message"
                    );
                }
            }
        }

        if recovered > 0 {
            info!(recovered, "startup recovery completed for stale runtime messages");
        }
    }

    /// Is there a user (`position = right`) message newer than `row`? Looks at
    /// the latest page only — a prompt buried deeper than that is long past.
    async fn has_user_message_after(&self, user_id: &str, row: &MessageRow) -> bool {
        let params = MessagePageParams {
            limit: 200,
            direction: MessagePageDirection::InitialLatest,
        };
        match self
            .conversation_repo()
            .list_messages_page(user_id, &row.conversation_id, &params)
            .await
        {
            Ok(page) => page
                .items
                .iter()
                .any(|m| m.position.as_deref() == Some("right") && m.created_at > row.created_at),
            Err(error) => {
                warn!(
                    conversation_id = %row.conversation_id,
                    error = %ErrorChain(&error),
                    "startup recovery could not check for a later user message; keeping the prompt"
                );
                false
            }
        }
    }
}

fn classify_recovery_action(row: &MessageRow) -> StartupRecoveryAction {
    if row.r#type == "tool_call" && is_pending_user_prompt(&row.content) {
        return StartupRecoveryAction::KeepPendingUserPrompt;
    }
    if matches!(row.r#type.as_str(), "tool_call" | "tool_group") {
        return StartupRecoveryAction::SettleToolRow;
    }
    if message_has_visible_content(row) {
        StartupRecoveryAction::FinishVisibleOutput
    } else {
        StartupRecoveryAction::FinishEmptyPlaceholder
    }
}

/// Rewrite a stale tool row's embedded status to its terminal form. Returns
/// `None` when the content needs no change (already terminal, or unparsable —
/// the row-level `finish` still applies either way).
///
/// The two channels speak different status vocabularies (see `ToolGroupStatus`):
/// `tool_call` content is snake_case, `tool_group` entries are PascalCase.
fn settle_tool_row_content(row_type: &str, content: &str) -> Option<String> {
    let mut value: serde_json::Value = serde_json::from_str(content).ok()?;
    match row_type {
        "tool_call" => {
            let stale = value
                .get("status")
                .and_then(|s| s.as_str())
                .is_none_or(|s| !matches!(s, "completed" | "error" | "canceled"));
            if !stale {
                return None;
            }
            value["status"] = serde_json::Value::String("canceled".into());
            Some(value.to_string())
        }
        "tool_group" => {
            let entries = value.as_array_mut()?;
            let mut changed = false;
            for entry in entries {
                let stale = entry
                    .get("status")
                    .and_then(|s| s.as_str())
                    .is_none_or(|s| !matches!(s, "Success" | "Error" | "Canceled"));
                if stale {
                    entry["status"] = serde_json::Value::String("Canceled".into());
                    changed = true;
                }
            }
            changed.then(|| value.to_string())
        }
        _ => None,
    }
}

/// An AskUserQuestion `tool_call` whose embedded status is still non-terminal:
/// the user never answered it. Unparsable content is never a prompt.
fn is_pending_user_prompt(content: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(content) else {
        return false;
    };
    value.get("name").and_then(|name| name.as_str()) == Some("AskUserQuestion")
        && value
            .get("status")
            .and_then(|status| status.as_str())
            .is_none_or(|status| !matches!(status, "completed" | "error" | "canceled"))
}

fn message_has_visible_content(row: &MessageRow) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&row.content) else {
        return !row.content.trim().is_empty();
    };

    value
        .get("content")
        .and_then(|content| content.as_str())
        .is_some_and(|content| !content.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use aionui_db::models::MessageRow;

    use super::*;

    #[test]
    fn visible_text_finishes_as_visible_output() {
        let row = MessageRow {
            id: "msg-1".into(),
            conversation_id: "conv-1".into(),
            msg_id: Some("msg-1".into()),
            r#type: "text".into(),
            content: serde_json::json!({ "content": "hello" }).to_string(),
            position: Some("left".into()),
            status: Some("work".into()),
            hidden: false,
            created_at: 1,
            backend_turn_id: None,
        };

        assert_eq!(
            classify_recovery_action(&row),
            StartupRecoveryAction::FinishVisibleOutput
        );
    }

    #[test]
    fn stale_tool_rows_classify_as_settle() {
        for ty in ["tool_call", "tool_group"] {
            let row = MessageRow {
                id: "m".into(),
                conversation_id: "c".into(),
                msg_id: Some("m".into()),
                r#type: ty.into(),
                content: "{}".into(),
                position: Some("left".into()),
                status: Some("work".into()),
                hidden: false,
                created_at: 1,
                backend_turn_id: None,
            };
            assert_eq!(classify_recovery_action(&row), StartupRecoveryAction::SettleToolRow);
        }
    }

    fn tool_row(name: &str, status: &str) -> MessageRow {
        MessageRow {
            id: "m".into(),
            conversation_id: "c".into(),
            msg_id: Some("m".into()),
            r#type: "tool_call".into(),
            content: serde_json::json!({
                "call_id": "m",
                "name": name,
                "args": {"questions": [{"question": "Proceed?", "options": [{"label": "Yes"}]}]},
                "status": status
            })
            .to_string(),
            position: Some("left".into()),
            status: Some("work".into()),
            hidden: false,
            created_at: 1,
            backend_turn_id: None,
        }
    }

    /// The stuck-question residue (live 2026-10-12): an unanswered
    /// AskUserQuestion is a pending user prompt, not a dead tool call — it must
    /// survive startup untouched so the frontend can still derive its card.
    #[test]
    fn pending_ask_user_question_rows_are_kept() {
        let row = tool_row("AskUserQuestion", "running");
        assert_eq!(
            classify_recovery_action(&row),
            StartupRecoveryAction::KeepPendingUserPrompt
        );
        assert!(is_pending_user_prompt(&row.content));
    }

    #[test]
    fn answered_ask_user_question_rows_still_settle() {
        for status in ["completed", "error", "canceled"] {
            let row = tool_row("AskUserQuestion", status);
            assert_eq!(classify_recovery_action(&row), StartupRecoveryAction::SettleToolRow);
            assert!(!is_pending_user_prompt(&row.content));
        }
    }

    #[test]
    fn other_running_tool_rows_still_settle() {
        let row = tool_row("Bash", "running");
        assert_eq!(classify_recovery_action(&row), StartupRecoveryAction::SettleToolRow);
        assert!(!is_pending_user_prompt("not json"));
    }

    /// The hard-kill residue (audit 2026-08-04): rows whose CONTENT still says
    /// running must be rewritten — the frontend renders the embedded status, so
    /// a row-level `finish` alone leaves the View Steps spinner alive.
    #[test]
    fn settle_rewrites_running_content_and_leaves_terminal_content_alone() {
        // tool_call: snake_case vocabulary.
        let card = serde_json::json!({"call_id": "t1", "name": "Bash", "status": "running"}).to_string();
        let out = settle_tool_row_content("tool_call", &card).expect("running → rewritten");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["status"], "canceled");
        assert_eq!(v["name"], "Bash", "only the status changes");

        let done = serde_json::json!({"call_id": "t1", "status": "completed"}).to_string();
        assert!(
            settle_tool_row_content("tool_call", &done).is_none(),
            "terminal content is left untouched"
        );

        // tool_group: PascalCase vocabulary, per entry.
        let group = serde_json::json!([
            {"call_id": "a", "name": "run:A", "status": "Executing"},
            {"call_id": "b", "name": "run:B", "status": "Success"}
        ])
        .to_string();
        let out = settle_tool_row_content("tool_group", &group).expect("one executing entry");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v[0]["status"], "Canceled");
        assert_eq!(v[1]["status"], "Success", "terminal entries keep their outcome");

        let all_done = serde_json::json!([{"call_id": "a", "status": "Success"}]).to_string();
        assert!(settle_tool_row_content("tool_group", &all_done).is_none());
    }

    #[test]
    fn empty_text_finishes_as_hidden_placeholder() {
        let row = MessageRow {
            id: "msg-1".into(),
            conversation_id: "conv-1".into(),
            msg_id: Some("msg-1".into()),
            r#type: "text".into(),
            content: serde_json::json!({ "content": "" }).to_string(),
            position: Some("left".into()),
            status: Some("work".into()),
            hidden: false,
            created_at: 1,
            backend_turn_id: None,
        };

        assert_eq!(
            classify_recovery_action(&row),
            StartupRecoveryAction::FinishEmptyPlaceholder
        );
    }
}
