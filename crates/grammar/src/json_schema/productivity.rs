use super::SchemaError;
use crate::pda::{CompiledGrammar, Symbol};

// Schema compilation uses no alternatives for an impossible value, whereas the
// PDA treats a zero-alternative rule as epsilon. Remove every path to those rules
// before exposing the grammar; explicit empty alternatives still derive epsilon.
pub(super) fn prune_empty_alternatives(grammar: &mut CompiledGrammar) -> Result<(), SchemaError> {
    let mut productive = vec![false; grammar.rules.len()];
    let mut dependents = vec![Vec::new(); grammar.rules.len()];
    let mut remaining = Vec::with_capacity(grammar.rules.len());
    let mut pending = Vec::new();

    for (rule_id, rule) in grammar.rules.iter().enumerate() {
        let mut counts = Vec::with_capacity(rule.alts.len());
        for (alt_id, alt) in rule.alts.iter().enumerate() {
            let mut count = 0usize;
            for symbol in alt {
                if let Symbol::NonTerminal(target) = symbol {
                    dependents
                        .get_mut(*target)
                        .ok_or_else(|| {
                            SchemaError("invalid compiled schema rule reference".into())
                        })?
                        .push((rule_id, alt_id));
                    count += 1;
                }
            }
            counts.push(count);
            if count == 0 && !productive[rule_id] {
                productive[rule_id] = true;
                pending.push(rule_id);
            }
        }
        remaining.push(counts);
    }

    // A cycle becomes productive only when a finite derivation reaches it.
    // Count occurrences, not distinct targets: one alternative can use a rule twice.
    while let Some(target) = pending.pop() {
        for &(rule_id, alt_id) in &dependents[target] {
            remaining[rule_id][alt_id] -= 1;
            if remaining[rule_id][alt_id] == 0 && !productive[rule_id] {
                productive[rule_id] = true;
                pending.push(rule_id);
            }
        }
    }

    if !productive.first().copied().unwrap_or(false) {
        return Err(SchemaError(
            "unsatisfiable schema: no finite JSON value".into(),
        ));
    }
    for rule in &mut grammar.rules {
        rule.alts.retain(|alt| {
            alt.iter().all(|symbol| match symbol {
                Symbol::NonTerminal(target) => productive[*target],
                Symbol::Terminal(_) | Symbol::AnyByte => true,
            })
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pda::{GrammarBuilder, GrammarState, StepResult, advance_byte};

    fn accepts(grammar: &CompiledGrammar, input: &[u8]) -> bool {
        let mut state = GrammarState::initial();
        for &byte in input {
            match advance_byte(&mut state, grammar, byte) {
                StepResult::Accepted => {}
                StepResult::Rejected => return false,
                StepResult::StackLimitExceeded => panic!("unexpected PDA limit"),
            }
        }
        state.is_complete()
    }

    #[test]
    fn repeated_dependencies_and_unreachable_cycle_keep_their_language() {
        let mut builder = GrammarBuilder::new();
        let root = builder.reserve("root");
        let word = builder.add_rule("word", vec![vec![Symbol::Terminal(b'x')]]);
        let dead = builder.reserve("dead");
        builder
            .set_alts(dead, vec![vec![Symbol::NonTerminal(dead)]])
            .unwrap();
        builder
            .set_alts(
                root,
                vec![vec![Symbol::NonTerminal(word), Symbol::NonTerminal(word)]],
            )
            .unwrap();
        let mut grammar = builder.build();
        prune_empty_alternatives(&mut grammar).unwrap();
        assert!(accepts(&grammar, b"xx"));
        assert!(!accepts(&grammar, b"x"));
        assert!(!accepts(&grammar, b"xxx"));
    }

    #[test]
    fn dead_dependency_cannot_turn_into_epsilon() {
        let mut builder = GrammarBuilder::new();
        let root = builder.reserve("root");
        let dead = builder.reserve("dead");
        builder
            .set_alts(
                root,
                vec![
                    vec![Symbol::Terminal(b'y')],
                    vec![Symbol::NonTerminal(dead), Symbol::Terminal(b'x')],
                ],
            )
            .unwrap();
        let mut grammar = builder.build();
        assert!(
            accepts(&grammar, b"x"),
            "PDA still interprets zero alternatives as epsilon"
        );
        prune_empty_alternatives(&mut grammar).unwrap();
        assert!(accepts(&grammar, b"y"));
        assert!(!accepts(&grammar, b"x"));
    }

    #[test]
    fn one_productive_dependency_does_not_satisfy_the_whole_alternative() {
        let mut builder = GrammarBuilder::new();
        let root = builder.reserve("root");
        let word = builder.add_rule("word", vec![vec![Symbol::Terminal(b'x')]]);
        let cycle = builder.reserve("cycle");
        builder
            .set_alts(cycle, vec![vec![Symbol::NonTerminal(cycle)]])
            .unwrap();
        builder
            .set_alts(
                root,
                vec![vec![Symbol::NonTerminal(word), Symbol::NonTerminal(cycle)]],
            )
            .unwrap();
        assert!(prune_empty_alternatives(&mut builder.build()).is_err());
    }
}
