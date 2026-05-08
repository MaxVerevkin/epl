# Compiler Architecture

This document shortly describes the architecture of the EPL compiler: the
pipeline that the source code goes through, the intermediate representations
(IRs) used during compilartion, and the rationales behind them.

## Overview

`src` -> `lexical tokens` -> `AST` -> `IR_TREE` -> `IR` -> `LLVM IR` -> `exe`

## Lexical Analysis

This stage transforms the source code text into a stream of lexical tokens, and
is standard for compilers and interpreters. There are four kinds of tokens:
literals, identifiers, keywords and punctuation. Whitespace and comments are
skipped. Comments start with `#` and continue to the end of the line.

For example. `fn main() -> i32 { printf("hello"); 0 }` becomes `fn` `main` `(` `)`
`->` `i32` `{` `printf` `(` `"hello"` `)` `;` `0` `}`. This stream is much more
convenient to parse than plain text.

EPL uses a hand-written lexer.

## Parsing

The parser in `src/ast.rs` is a hand-written recursive descent parser. It turns
the token stream into an abstract syntax tree (AST). The AST deliberately stays
close to the source:

- Names are still strings with spans.
- Type names are unresolved.
- Most semantic checks are not performed yet.
- Expression precedence has been made explicit by tree shape.

The parser does a small amount of syntax validation, such as rejecting duplicate
annotations on one item, requiring `...` to be the final function parameter, and
requiring `let name;` to include either a type or an initializer.

The earlier `hello` example is parsed as:

```text
Ast {
    items: [
        Item {
            annotations: {},
            kind: Item::Function {
                name: main@3..7,
                args: [],
                is_variadic: false,
                return_ty: Some(
                    i32@13..16,
                ),
                body: Some(
                    BlockExpr {
                        statements: [
                            Expr(
                                FunctionCallExpr {
                                    name: printf@19..25,
                                    args: [
                                        "hello"@26..33,
                                    ],
                                    args_span: 25..34,
                                },
                            ),
                        ],
                        final_expr: Some(
                            0@36..37,
                        ),
                        opening_brace_span: 17..18,
                        closing_brace_span: 38..39,
                    },
                ),
            },
        },
    ],
}
```

## AST to IR_TREE

`IR_TREE` is built by `ir_tree::Module::from_ast`. This is the main semantic
analysis stage. The representation is still expression-oriented and tree-shaped,
but it is no longer source syntax:

- Every expression has a static type.
- Every local variable, function, and loop has an entity ID instead of relying
  on a source name.
- Struct and array types have computed layouts.
- Place expressions, such as variables, fields, indexed arrays, and pointer
  dereferences, are represented explicitly.
- Some source constructs are rewritten into simpler core forms.

`Module::from_ast` performs several passes over the AST:

1. Seed the type namespace with built-ins and collect struct definitions.
2. Collect function declarations, including signatures, variadic flags, return
   types, and the `@pure` annotation.
3. Lower each function body using those complete type and function namespaces.
4. Run semantic checkers, including the required `fn main() -> i32` ABI check,
   `comptime` purity checks, and `@pure` function checks.
5. Run basic tree optimizations.
6. Evaluate every remaining `comptime` expression and replace it with a
   constant.

IR_TREE lowering is where most language rules in `doc/language.md` are enforced:
name lookup, type checking, local inference, assignment validity, call arity,
field existence, array indexing type, loop result types, cast validity, and so
on.

### Lowered Constructs

IR_TREE removes several source-level conveniences:

- Function arguments become ordinary local variables initialized from
  `Argument(n)` expressions. This means assignment to an argument is handled the
  same way as assignment to a local variable.
- `while cond { body }` becomes a `loop` containing an `if cond { body } else {
  break }`.
- `for i in start..end { body }` becomes a block with hidden counter and target
  variables, plus a normal loop. The visible `i` is assigned a per-iteration
  copy.
- `a || b` becomes `if a { true } else { b }`.
- `a && b` becomes `if a { b } else { false }`.
- Unary negation becomes subtraction from zero.

The same `hello` example becomes:

```text
fn main() -> i32
    BLOCK TYPE=i32
    |   FUNCTION_CALL("printf") TYPE=unit
    |   |   CONST_STRING("hello") TYPE=*i8
    |   CONST 0 TYPE=i32
```

### Compile-Time Evaluation

`comptime expr` is first lowered and type checked like any other expression,
then the checker verifies that it is pure to run during compilation. The
evaluator in `src/ir_tree/evaluator.rs` interprets the IR_TREE expression and
returns an `ir_tree::Constant`.

The evaluator supports normal control flow by using internal `Return`, `Break`,
and `Continue` signals. It also has a small constant-memory model: local
variables are stored as byte buffers using the same type layouts as the
compiler, so compile-time structs and arrays can be mutated through local
places.

For example, `examples/comptime_primes.epl` computes an array at compile time.
In the IR_TREE snapshot, the whole `comptime` block has already become one
constant:

```text
STORE TYPE=unit
|   VARIABLE(var_1) [PLACE] TYPE=[i32; 20]
|   CONST [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73] TYPE=[i32; 20]
```

The evaluator may call `@pure` functions, including recursive pure functions, by
interpreting their IR_TREE bodies with constant arguments. It rejects operations
that would depend on run-time state, such as accessing variables outside the
`comptime` expression, calling impure functions, dereferencing pointers, or
using string literals.

### IR_TREE Clean Up

The current IR_TREE optimizer is small and local, and performs these
simplifications:

- `&ptr.*` becomes `ptr`.
- `(&place).*` becomes `place`.
- Empty blocks and single-expression blocks with no local variables are
  collapsed.
- Code after the first `!` expression in a block is removed.
- Pure unused expressions before the final expression of a block are removed.
- Redundant trailing `unit` constants are removed.

## IR_TREE to IR

IR is control flow grapth (CFG) signle static assignment (SSA) intermediate
representation.

- Function bodies are control-flow graphs.
- Each basic block has instructions and a terminator.
- Values are constants or `DefinitionId`s produced by instructions or block
  arguments.
- Local variables are represented as stack allocation slots (`allocas`).
- Loads, stores, calls, integer operations, comparisons, pointer offsets, and
  integer casts are explicit instructions.

This lowering also collapses signed and unsigned integer types of the same
width. IR_TREE has distinct types such as `i32` and `u32`, but lower IR has only
`I8`, `I32`, and `I64`. Signedness is not stored in the IR integer type itself,
it is copied from the typed IR_TREE operands into the specific arithmetic,
comparison, and extension instructions that need to know whether the operation
is signed.

### IR Clean Up

After IR lowering, `ir::opt::basic_passes` runs two passes:

- `drop_zst` removes operations on zero-sized types. It drops zero-sized
  arguments, block arguments, allocas, loads, stores, and normalizes zero-sized
  returns to `unit`.
- `simplify_cfg` removes unreachable blocks, merges a block with its only
  successor when the successor has a single predecessor, and redirects empty jump
  blocks into their targets.

## IR to LLVM IR

LLVM generation happens in two phases:

1. Declare every function and remember its LLVM function type. This allows calls
   to functions defined later in the file.
2. Emit every function body.

For each function body, the compiler:

- Creates LLVM basic blocks for every IR basic block.
- Emits entry-block `alloca` instructions for IR allocation slots.
- Creates LLVM `phi` nodes for non-entry block arguments.
- Emits each IR instruction as the corresponding LLVM instruction.
- Emits block terminators and fills phi incoming edges from jump arguments.

The type mapping is direct:

- `unit` is an empty LLVM struct.
- `bool` is `i1`.
- IR integer types `I8`, `I32`, and `I64` map to LLVM integer types of the same
  width.
- Pointers map to opaque LLVM pointers.
- Structs and arrays map to LLVM aggregate types.

After construction, the LLVM module is verified. The `llvm-ir` command prints
the module. The `llvm-obj` command asks LLVM to emit a native object file named
`a.out.o`.

## Linking

The compiler currently stops at an object file. To produce and run an executable,
link it with a system C compiler:

```sh
cargo run -- llvm-obj examples/hello_world.epl
cc a.out.o
./a.out
```

External declarations such as `fn printf(fmt: *i8, ...);` are emitted as LLVM
function declarations. The final link step is responsible for resolving those
symbols from libc or another object/library.

## Diagnostics

Lexer, parser, IR_TREE, and IR errors are all adapted to a shared diagnostic
interface in `src/diagnostics.rs`. Most errors carry a `Span`, so the command
line can print:

- The diagnostic message.
- The file and line.
- The relevant source line.
- A caret underline for the span.

Semantic diagnostics are mostly produced during AST to IR_TREE lowering, because
that is the first stage with enough information to know what each name and type
means.

## Testing and Snapshots

The helper crate in `epl_test/` runs compiler tests over `examples/` and
`tests/`.

For each `.epl` file it currently:

- Runs `ir_tree` and compares the output with the neighboring `.epl.ir_tree`
  snapshot, creating a new snapshot if one is missing.
- Runs `ir` as a smoke test.

Run the test harness with:

```sh
cargo b
cargo r -p epl_test ./target/debug/epl examples tests
```

The IR_TREE snapshots are useful documentation for the lowering rules. For
example, the snapshots show `for` loops lowered to hidden counter variables,
`comptime` expressions replaced by constants, and redundant `&`/`.*` pairs
simplified away.
