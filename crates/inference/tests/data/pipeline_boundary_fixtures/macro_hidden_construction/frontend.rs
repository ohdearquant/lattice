use crate::neutral::PlainOptions;

macro_rules! harmless {
    () => {};
}

some_unclassified_macro!(crate::model::ConcreteModel);

pub fn consume(_options: PlainOptions) {}
