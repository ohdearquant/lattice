//! Model query builder.

use super::registry::ModelRegistry;
use crate::registry::model::{ModelStatus, RegisteredModel};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Model query builder
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelQuery {
    name: Option<String>,
    status: Option<ModelStatus>,
    min_accuracy: Option<f32>,
    tags: Vec<String>,
    limit: Option<usize>,
}

impl ModelQuery {
    /// Create a new query
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Filter by status
    pub fn status(mut self, status: ModelStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Filter by minimum accuracy
    pub fn min_accuracy(mut self, accuracy: f32) -> Self {
        self.min_accuracy = Some(accuracy);
        self
    }

    /// Filter by tag
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Limit results
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Execute query against registry
    ///
    /// Returns owned `RegisteredModel` values (cloned from the registry snapshot).
    pub fn execute(&self, registry: &ModelRegistry) -> Vec<RegisteredModel> {
        let mut results: Vec<RegisteredModel> = registry
            .list_all()
            .into_iter()
            .filter(|m| {
                // Name filter
                if let Some(ref name) = self.name {
                    if &m.name != name {
                        return false;
                    }
                }

                // Status filter
                if let Some(ref status) = self.status {
                    if &m.status != status {
                        return false;
                    }
                }

                // Accuracy filter
                if let Some(min_acc) = self.min_accuracy {
                    if let Some(acc) = m.metadata.validation_accuracy {
                        if acc < min_acc {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }

                // Tag filter
                for tag in &self.tags {
                    if !m.metadata.tags.contains(tag) {
                        return false;
                    }
                }

                true
            })
            .collect();

        // Apply limit
        if let Some(limit) = self.limit {
            results.truncate(limit);
        }

        results
    }
}
