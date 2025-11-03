use super::*;

/// Render the CFG of a program into a DOT graph
pub fn graph(ir: &Ir) -> String {
    let mut retval = String::from("digraph {\n");
    for (name, function) in &ir.functions {
        let decl = &ir.function_decls[name];
        retval.push_str(&function_subgraph(decl, function));
    }
    retval.push_str("}\n");
    retval
}

/// Render the CFG of a function into a DOT subgraph cluster
pub fn function_subgraph(decl: &FunctionDecl, ir: &Function) -> String {
    let subgraph_name = format!("function_{}", decl.name.value);
    let entry_name = format!("{subgraph_name}_entry");
    let mut retval = format!(
        "subgraph cluster_{subgraph_name} {{\nlabel = \"{}\"\n{entry_name} [label=\"Entry\"]\nnode [shape=record];\n{entry_name} -> {:?}\n",
        decl.name.value, ir.entry,
    );

    for (&basic_block_id, basic_block) in &ir.basic_blokcs {
        retval.push_str(&format!(
            "{basic_block_id:?} [label=\"{}\"]\n",
            escape_label(&make_basic_block_label(basic_block_id, basic_block))
        ));
        match &basic_block.terminator {
            Terminator::Jump { to } => {
                retval.push_str(&format!("{basic_block_id:?} -> {to:?}\n"));
            }
            Terminator::CondJump {
                cond: _,
                if_true,
                if_false,
            } => {
                retval.push_str(&format!("{basic_block_id:?} -> {if_true:?}\n"));
                retval.push_str(&format!("{basic_block_id:?} -> {if_false:?}\n"));
            }
            Terminator::Return { .. } | Terminator::Unreachable => (),
        }
    }

    retval.push_str("}\n");
    retval
}

/// Escape a string to be used as a DOT label
fn escape_label(label: &str) -> String {
    let mut escaped = String::new();
    for ch in label.chars() {
        match ch {
            '<' => escaped.push_str("\\<"),
            '>' => escaped.push_str("\\>"),
            '{' => escaped.push_str("\\{"),
            '}' => escaped.push_str("\\}"),
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\l"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Render the basic block into a label string (unescaped)
fn make_basic_block_label(id: BasicBlockId, basic_block: &BasicBlock) -> String {
    let mut label = format!("{id:?}{:?}:\n", basic_block.args);
    for instruction in &basic_block.instructions {
        label.push_str(&format!("{instruction:?}\n"));
    }
    match &basic_block.terminator {
        Terminator::Jump { .. } => (),
        Terminator::CondJump {
            cond,
            if_true,
            if_false,
        } => {
            label.push_str(&format!(
                "if {cond:?} jump {if_true:?} else jump {if_false:?}\n"
            ));
        }
        Terminator::Return { value } => {
            label.push_str(&format!("return {value:?}\n"));
        }
        Terminator::Unreachable => {
            label.push_str("unreachable\n");
        }
    }
    label
}
