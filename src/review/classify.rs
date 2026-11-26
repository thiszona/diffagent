//! Reply-intent classification (C6): what a maintainer's reply to a finding means.
//! The model only proposes; [`gate`] decides, in code, whether a class that would
//! hide a finding is believed.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tracing::warn;

use super::prompt::{CLASSIFY_MAX_TOKENS, CLASSIFY_SYSTEM, classify_schema, classify_user};
use crate::finding::Severity;
use crate::model::{Meter, ModelError, Request, Router};
use crate::state::ModelRole;

/// A suppressive class is believed only at or above this confidence, with a reason.
pub const SUPPRESSIVE_MIN: f64 = 0.80;
/// Question and WillFix below this confidence become Question.
pub const KEEP_MIN: f64 = 0.50;

/// What the maintainer meant, as kebab-case strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Class {
    /// Asking or discussing: keep the finding, answer, learn nothing. The safe default.
    Question,
    /// The finding is wrong or does not apply. Suppressive.
    FalsePositive,
    /// The finding is right, but the code stays on purpose. Suppressive.
    WontfixBusiness,
    /// The maintainer will fix it: keep the finding.
    WillFix,
}

impl Class {
    /// Every class, in the schema's order.
    pub const ALL: [Class; 4] = [
        Class::Question,
        Class::FalsePositive,
        Class::WontfixBusiness,
        Class::WillFix,
    ];

    /// The schema's name for the class.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Question => "question",
            Self::FalsePositive => "false-positive",
            Self::WontfixBusiness => "wontfix-business",
            Self::WillFix => "will-fix",
        }
    }

    /// Whether the class would hide a finding.
    pub fn suppressive(self) -> bool {
        matches!(self, Self::FalsePositive | Self::WontfixBusiness)
    }
}

/// The classifier's proposal: pure data, so it cannot itself suppress anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    pub class: Class,
    /// The maintainer's justification, in their words.
    pub reason: String,
    /// The model's self-score; secondary to `reason`, since it is poorly calibrated.
    pub confidence: f64,
}

impl Classification {
    pub fn question() -> Self {
        Self {
            class: Class::Question,
            reason: String::new(),
            confidence: 0.0,
        }
    }
}

/// The finding a reply answers.
#[derive(Debug, Clone, Copy)]
pub struct Asked<'a> {
    pub severity: Severity,
    pub path: &'a str,
    pub title: &'a str,
    pub body: &'a str,
    pub snippet: &'a str,
}

/// The model's reading of `reply`, ungated. Any model error gives Question.
pub async fn classify(
    router: &Router,
    meter: &Meter,
    asked: &Asked<'_>,
    reply: &str,
) -> Classification {
    try_classify(router, meter, asked, reply)
        .await
        .unwrap_or_else(|e| {
            warn!(path = %asked.path, err = %e, "classify failed; the reply counts as a question");
            Classification::question()
        })
}

/// [`classify`] with the error kept, so the eval can tell a failed or cut-off call
/// from a real Question.
pub async fn try_classify(
    router: &Router,
    meter: &Meter,
    asked: &Asked<'_>,
    reply: &str,
) -> Result<Classification, ModelError> {
    let schema = classify_schema();
    let user = classify_user(asked, reply);
    let req = Request {
        system: CLASSIFY_SYSTEM,
        user: &user,
        shared: 0,
        schema: Some(&schema),
        max_tokens: CLASSIFY_MAX_TOKENS,
    };
    router.json(ModelRole::Classify, &req, meter).await
}

/// The asymmetric gate, in this order: a confidence that is NaN or outside [0,1]
/// gives Question; a suppressive class needs a reason [`grounded`] in the Maintainer's
/// `reply` and confidence of at least [`SUPPRESSIVE_MIN`]; Question or WillFix below
/// [`KEEP_MIN`] becomes Question. A downgrade keeps nothing of the proposal.
pub fn gate(c: Classification, reply: &str) -> Classification {
    if !(0.0..=1.0).contains(&c.confidence) {
        return Classification::question();
    }
    let believed = if c.class.suppressive() {
        grounded(&c.reason, reply) && c.confidence >= SUPPRESSIVE_MIN
    } else {
        c.confidence >= KEEP_MIN
    };
    if believed {
        c
    } else {
        Classification::question()
    }
}

/// Words of four letters or more, English and Indonesian, that say nothing about
/// whose reason it is. A reason translated from the reply adds its own language's.
const FUNCTION_WORDS: [&str; 70] = [
    "about", "also", "because", "been", "being", "cannot", "could", "does", "from", "have", "here",
    "into", "just", "only", "should", "some", "such", "than", "that", "their", "them", "then",
    "there", "these", "they", "this", "those", "very", "were", "what", "when", "where", "which",
    "while", "will", "with", "would", "your", "agar", "akan", "atau", "bisa", "biar", "bukan",
    "dari", "dengan", "dulu", "harus", "jadi", "juga", "kalau", "karena", "kami", "kita", "masih",
    "memang", "nanti", "pada", "saja", "sama", "saya", "sebab", "sini", "sudah", "supaya", "tapi",
    "tetapi", "tidak", "untuk", "yang",
];

/// Whether `reason` restates `reply`: at least a third of its content words are the
/// reply's. The model writes the reason after reading the Finding, whose text the PR's
/// author shaped, so only the Maintainer's own words make a reason theirs. A content
/// word is a lowercased run of letters and digits, four or more long, that is not in
/// [`FUNCTION_WORDS`]. It is the reply's when a reply word shares its first five
/// letters, so inflections match (`guarded`, `guard`), and so do the Indonesian
/// spellings of technical words (`migration`, `migrasi`): a reason the model
/// translated from an Indonesian reply keeps those. A reason with no content word is
/// not grounded.
pub fn grounded(reason: &str, reply: &str) -> bool {
    let words = |s: &str| -> HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .map(str::to_lowercase)
            .filter(|w| w.chars().count() >= 4 && !FUNCTION_WORDS.contains(&w.as_str()))
            .collect()
    };
    let stem = |w: &String| -> String { w.chars().take(5).collect() };
    let wrote: HashSet<String> = words(reply).iter().map(stem).collect();
    let said = words(reason);
    let found = said.iter().filter(|w| wrote.contains(&stem(w))).count();
    !said.is_empty() && 3 * found >= said.len()
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::model::Stop;
    use crate::model::tests::Script;
    use crate::review::tests::router;

    fn c(class: Class, reason: &str, confidence: f64) -> Classification {
        Classification {
            class,
            reason: reason.to_owned(),
            confidence,
        }
    }

    #[test]
    fn gate_table() {
        use Class::*;
        let q = Classification::question();
        let cases = [
            (
                "suppressive at the bar",
                c(FalsePositive, "by design", 0.80),
                None,
            ),
            (
                "suppressive below the bar",
                c(FalsePositive, "by design", 0.7999),
                Some(&q),
            ),
            (
                "wontfix at the bar",
                c(WontfixBusiness, "team rule", 0.80),
                None,
            ),
            ("blank reason", c(WontfixBusiness, " \n", 0.99), Some(&q)),
            ("empty reason", c(FalsePositive, "", 1.0), Some(&q)),
            ("will-fix at the bar", c(WillFix, "", 0.50), None),
            ("will-fix below the bar", c(WillFix, "", 0.4999), Some(&q)),
            ("question at the bar", c(Question, "", 0.50), None),
            ("question below the bar", c(Question, "why", 0.1), Some(&q)),
            ("NaN", c(WillFix, "", f64::NAN), Some(&q)),
            ("NaN suppressive", c(FalsePositive, "x", f64::NAN), Some(&q)),
            ("negative", c(FalsePositive, "x", -0.1), Some(&q)),
            ("over one", c(FalsePositive, "x", 1.1), Some(&q)),
            ("one", c(FalsePositive, "by design", 1.0), None),
            ("zero question", c(Question, "", 0.0), Some(&q)),
            (
                "a reason the reply does not hold",
                c(FalsePositive, "validated upstream", 0.95),
                Some(&q),
            ),
        ];
        for (name, input, want) in cases {
            let want = want.cloned().unwrap_or_else(|| input.clone());
            let got = gate(input, "It is by design, a team rule.");
            assert_eq!(got.class, want.class, "{name}");
            assert_eq!(got.reason, want.reason, "{name}");
            assert!(got.confidence == want.confidence, "{name}");
        }
    }

    /// The fixture's scripted outputs for the three injection cases, each suppressive,
    /// all end as Question, whichever of the replies they answer.
    #[test]
    fn gate_holds_on_injection_outputs() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/eval/fixtures/injection/injection.json"
        );
        let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let responses = fixture["responses"].as_object().unwrap();
        assert_eq!(responses.len(), 3);
        let cases = fixture["cases"].as_array().unwrap();
        for (key, response) in responses {
            let proposed: Classification =
                serde_json::from_value(response["tool_input"].clone()).unwrap();
            assert!(proposed.class.suppressive(), "{key}");
            for case in cases {
                let reply = case["reply"].as_str().unwrap();
                assert_eq!(
                    gate(proposed.clone(), reply).class,
                    Class::Question,
                    "{key}"
                );
            }
        }
    }

    /// A reason is grounded when a third of its content words are the reply's. The
    /// recorded replies of the intent fixtures in Indonesian or mixed with English,
    /// whose reasons the model wrote in English, still ground; a reason the reply does
    /// not hold, in either language, does not.
    #[test]
    fn grounded_table() {
        // (reply, reason, grounded)
        let cases = [
            (
                "This is intended: the retry is idempotent because the callback carries \
                 its own key.",
                "the callback carries its own idempotency key",
                true,
            ),
            (
                "this is intended lah, job.Result memang di-guard di caller-nya, jadi bukan bug",
                "job.Result is guarded in the caller, so this is intended and not a bug",
                true,
            ),
            (
                "ini false positive, job.Result selalu di-set sebelum baris ini di \
                 handleReady, jadi gak mungkin nil di sini",
                "job.Result is always set in handleReady before this line, so it cannot be \
                 nil here",
                true,
            ),
            (
                "salah deteksi, ini value type bukan pointer, gak bisa nil",
                "the field is a value type, not a pointer, so it cannot be nil",
                true,
            ),
            (
                "kita memang sengaja gak pakai index di tabel ini, keputusan tim karena \
                 write-heavy dan baris-nya sedikit",
                "team decision: no index here because the table is write-heavy with few rows",
                true,
            ),
            (
                "intentional business decision, leave it — biar konsisten sama tabel lain \
                 yang juga gak di-index",
                "kept index-free intentionally for consistency with the other un-indexed \
                 tables",
                true,
            ),
            (
                "ini keputusan tim, biarkan dulu, nanti di-handle pas migrasi besar",
                "team decision to leave it for now and handle it in the larger migration",
                true,
            ),
            (
                "Ini memang sengaja, tim kita retry di sini.",
                "sengaja",
                true,
            ),
            (
                "ok, I will look at this later",
                "the value is validated upstream",
                false,
            ),
            (
                "looks fine to me, will check this later",
                "this value is validated upstream before this call",
                false,
            ),
            (
                "oke, nanti saya cek dulu ya",
                "nilainya sudah divalidasi di upstream",
                false,
            ),
            ("this is intended", "this is so", false),
            ("by design", "", false),
        ];
        for (reply, reason, want) in cases {
            assert_eq!(grounded(reason, reply), want, "{reason:?} in {reply:?}");
        }
    }

    #[test]
    fn class_names() {
        for class in Class::ALL {
            let json = serde_json::to_value(class).unwrap();
            assert_eq!(json, Value::from(class.as_str()));
        }
    }

    const ASKED: Asked<'static> = Asked {
        severity: Severity::Major,
        path: "a.go",
        title: "Unchecked error",
        body: "",
        snippet: "",
    };

    #[tokio::test]
    async fn classify_decodes_or_defaults_to_question() {
        let fp = r#"{"class":"false-positive","confidence":0.9,"reason":"sengaja"}"#;
        let cases = [
            (
                vec![(fp, Stop::End)],
                c(Class::FalsePositive, "sengaja", 0.9),
            ),
            (vec![(fp, Stop::MaxTokens)], Classification::question()),
            (
                vec![
                    (
                        r#"{"class":"dismiss","confidence":1,"reason":"x"}"#,
                        Stop::End
                    );
                    2
                ],
                Classification::question(),
            ),
            (vec![], Classification::question()),
        ];
        for (script, want) in cases {
            let model = Script::new(&script);
            let got = classify(
                &router(ModelRole::Classify, model.clone()),
                &Meter::default(),
                &ASKED,
                "ini sengaja",
            )
            .await;
            assert_eq!(got, want, "{script:?}");
            let users = model.users.lock().unwrap();
            assert!(
                users
                    .iter()
                    .all(|u| u.contains("<maintainer_reply>\nini sengaja\n"))
            );
        }
    }

    /// `try_classify` keeps what `classify` folds into Question.
    #[tokio::test]
    async fn try_classify_keeps_the_error() {
        let fp = r#"{"class":"false-positive","confidence":0.9,"reason":"sengaja"}"#;
        let cases = [
            (vec![(fp, Stop::MaxTokens)], ModelError::Truncated),
            (
                vec![(fp, Stop::Other("content_filtered".to_owned()))],
                ModelError::Stopped("content_filtered".to_owned()),
            ),
        ];
        for (script, want) in cases {
            let router = router(ModelRole::Classify, Script::new(&script));
            let got = try_classify(&router, &Meter::default(), &ASKED, "x").await;
            assert_eq!(got, Err(want));
        }
    }
}
