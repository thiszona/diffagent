//! diffagent: a GitHub App that reviews pull requests, answers questions in its
//! review threads, and learns per-repo rules from maintainer corrections.

pub mod anchor;
pub mod config;
pub mod context;
pub mod diff;
pub mod eval;
pub mod finding;
pub mod github;
pub mod identity;
pub mod jira;
pub mod job;
pub mod lang;
pub mod learn;
pub mod model;
pub mod plan;
pub(crate) mod proc;
pub mod render;
pub mod review;
pub mod scan;
pub mod state;
pub mod suggest;
pub mod webhook;
pub mod worker;
pub mod workspace;
