//! Simplifies the control flow graph
//!
//! - Removes unreachable blocks.
//! - Merges successors of in-degree 1 with their predecessor of out-degree 1.
//! - Merges empty jump blocks into its predecessor of out-degree 1.
//! - Breaks critical edges.

use super::*;

pub fn pass(function: &mut Function) {
    let Some(body) = &mut function.body else { return };

    eliminate_unreachable(body);

    let mut predecessor_map = build_predecessor_map(body);

    let mut queue = body.postorder();
    let mut removed = HashSet::new();
    let mut rename_map = RenameMap::default();
    while let Some(block_id) = queue.pop() {
        if removed.contains(&block_id) {
            continue;
        }

        if let [succ_id] = *body.basic_blocks[&block_id].terminator.successors()
            && predecessor_map[&succ_id].len() == 1
        {
            let succ = body.basic_blocks.remove(&succ_id).unwrap();
            removed.insert(succ_id);
            queue.push(block_id);

            let block = body.basic_blocks.get_mut(&block_id).unwrap();
            block.instructions.extend(succ.instructions);
            let Terminator::Jump { to: _, args: succ_args } = std::mem::replace(&mut block.terminator, succ.terminator)
            else {
                unreachable!()
            };
            rename_map.map_args(&succ.args, &succ_args);
            predecessor_map = build_predecessor_map(body);
            continue;
        }

        if block_id != body.entry
            && body.basic_blocks[&block_id].instructions.is_empty()
            && body.basic_blocks[&block_id].args.is_empty()
            && let &Terminator::Jump { to, .. } = &body.basic_blocks[&block_id].terminator
            && to != block_id
        {
            let block = body.basic_blocks.remove(&block_id).unwrap();
            removed.insert(block_id);

            let Terminator::Jump {
                to: jump_to,
                args: jump_args,
            } = block.terminator
            else {
                unreachable!()
            };

            for pred_id in &predecessor_map[&block_id] {
                match &mut body.basic_blocks.get_mut(pred_id).unwrap().terminator {
                    Terminator::Jump { to, args } => {
                        *to = jump_to;
                        *args = jump_args.clone();
                    }
                    Terminator::CondJump {
                        cond: _,
                        if_true,
                        if_true_args,
                        if_false,
                        if_false_args,
                    } => {
                        if *if_true == block_id {
                            *if_true = jump_to;
                            *if_true_args = jump_args.clone();
                        } else if *if_false == block_id {
                            *if_false = jump_to;
                            *if_false_args = jump_args.clone();
                        }
                    }
                    Terminator::Return(_) | Terminator::Unreachable => unreachable!(),
                }
            }
            predecessor_map = build_predecessor_map(body);
            continue;
        }
    }

    for block in body.basic_blocks.values_mut() {
        for insn in &mut block.instructions {
            insn.kind.visit_operands_mut(|operand| {
                rename_map.rename(operand);
            });
        }
        block.terminator.visit_operands_mut(|operand| {
            rename_map.rename(operand);
        });
    }

    break_critical_edges(body, &predecessor_map);
}

fn eliminate_unreachable(body: &mut FunctionBody) {
    let reachable_blocks: HashSet<_> = body.postorder().into_iter().collect();
    body.basic_blocks.retain(|id, _bb| reachable_blocks.contains(id));
}

fn build_predecessor_map(body: &FunctionBody) -> HashMap<BasicBlockId, Vec<BasicBlockId>> {
    let mut predecessor_map = HashMap::<BasicBlockId, Vec<BasicBlockId>>::new();
    for (pred_id, block) in &body.basic_blocks {
        for succ in block.terminator.successors() {
            predecessor_map.entry(succ).or_default().push(*pred_id);
        }
    }
    predecessor_map
}

#[derive(Default)]
struct RenameMap {
    map: HashMap<DefinitionId, Value>,
}

impl RenameMap {
    fn map_args(&mut self, args: &[DefinitionId], arg_vals: &[Value]) {
        for (from, to) in args.iter().zip(arg_vals) {
            self.insert(from.clone(), to.clone());
        }
    }

    fn insert(&mut self, from: DefinitionId, to: Value) {
        for x in self.map.values_mut() {
            if let Value::Definition(x_def_id) = x
                && *x_def_id == from
            {
                *x = to.clone();
            }
        }
        self.map.insert(from, to);
    }

    fn rename(&self, operand: &mut Value) {
        if let Value::Definition(operand_def_if) = operand
            && let Some(renamed_to) = self.map.get(operand_def_if)
        {
            *operand = renamed_to.clone();
        }
    }
}

fn break_critical_edges(body: &mut FunctionBody, predecessor_map: &HashMap<BasicBlockId, Vec<BasicBlockId>>) {
    for block_id in body.basic_blocks.keys().copied().collect::<Vec<_>>() {
        let successors = body.basic_blocks[&block_id].terminator.successors();
        if successors.len() > 1 {
            for successor_id in successors {
                if predecessor_map.get(&successor_id).is_some_and(|preds| preds.len() > 1) {
                    break_critical_edge(body, block_id, successor_id);
                }
            }
        }
    }
}

fn break_critical_edge(body: &mut FunctionBody, from: BasicBlockId, to: BasicBlockId) {
    let new_block_id = BasicBlockId::new();
    let args = match &mut body.basic_blocks.get_mut(&from).unwrap().terminator {
        Terminator::Jump { .. } | Terminator::Return(_) | Terminator::Unreachable => unreachable!(),
        Terminator::CondJump {
            cond: _,
            if_true,
            if_true_args,
            if_false,
            if_false_args,
        } => {
            if *if_true == to {
                *if_true = new_block_id;
                std::mem::take(if_true_args)
            } else {
                assert_eq!(*if_false, to);
                *if_false = new_block_id;
                std::mem::take(if_false_args)
            }
        }
    };
    body.basic_blocks.insert(
        new_block_id,
        BasicBlock {
            args: Vec::new(),
            instructions: Vec::new(),
            terminator: Terminator::Jump { to, args },
        },
    );
}
