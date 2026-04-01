pub mod drop_zst;
pub mod simplify_cfg;

use super::*;

pub fn basic_passes(function: &mut Function) {
    drop_zst::pass(function);
    simplify_cfg::pass(function);
}
