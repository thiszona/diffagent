//! The Finding: one problem the reviewer reports at a location in a changed file.

use serde::{Deserialize, Serialize};

/// How much a Finding matters. Ordered from least to most severe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Nitpick,
    Minor,
    Major,
    Critical,
}

/// The kind of problem, picked by the model from a fixed list. It is part of a
/// Finding's identity, so two problems in one function stay distinct without
/// relying on the model's wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Category {
    NilSafety,
    ErrorHandling,
    Concurrency,
    ResourceLeak,
    Security,
    Validation,
    Logic,
    Performance,
    Maintainability,
}

impl Category {
    /// The stable identifier used in identities and prompts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NilSafety => "nil-safety",
            Self::ErrorHandling => "error-handling",
            Self::Concurrency => "concurrency",
            Self::ResourceLeak => "resource-leak",
            Self::Security => "security",
            Self::Validation => "validation",
            Self::Logic => "logic",
            Self::Performance => "performance",
            Self::Maintainability => "maintainability",
        }
    }
}

/// One reported problem. Line numbers are NEW-file lines as claimed by the model,
/// so they are untrusted: possibly zero, negative, or far outside the file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    pub path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub severity: Severity,
    pub category: Category,
    pub title: String,
    pub body: String,
    /// Verbatim copy of the flagged line, used to verify the anchor.
    #[serde(default)]
    pub snippet: String,
    /// Replacement text for the anchored span, if the model proposed one.
    #[serde(default)]
    pub suggestion: String,
}
