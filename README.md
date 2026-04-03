# My Final Year Project

## Project proposal

Title: Design and Implementation of an Ahead-of-Time Compiled Systems Programming Language with Advanced Compile-Time Execution and Modular Code Generation Backends.

Motivation: Many modern languages implement some sort of compile time evaluation and macro system. I would like to explore the state of the art implementations, their abilities and limitations. Then I want to design a minimal language (akin to C) but with powerful meta-programming capabilities. To make it “useful” I would support calling external C libraries, and to make the generated executables performant, I will incorporate LLVM / Cranelift / GCC for the machine code generation. However I would also include a minimal aarch64 (and maybe x86_64) backend to make the projects self-sufficient. Note that those backends do not have a goal to be anywhere near performant.

## Docs

- [Syntax reference](doc/syntax.md)
- [Building with LLVM](doc/llvm.md)

## Current architecture overview

The current pipeline is as follows:

1. Source file is lexically analyzed and parsed into an abstract syntax tree (AST).
2. AST is lowered to IR_TREE. Name resolution and type checking is performed in this step. IR_TREE resembles the original source code  but is fully typed and certain constructs are expressed in simpler building blocks (such as for loop -> regular loop, logical and/or -> if expression).
3. Very basic optimizations are performed on IR_TREE.
4. IR_TREE is lowered to IR, whith is a low-level, single static assignment (SSA), basic-block-based reperesentation.
5. The following optimization passes are run on the IR:
    - `drop_zst` - Operations on zero sized types (ZSTs) are eliminated.
    - `simplify_cfg` - Remove unreachable basic blocks and merge.
    - `mem2reg` - _comming soon._
6. IR is trivially lowered to LLVM IR.
7. LLVM performs its optimizations and emits an object file.

## Roadmap

- [ ] Basic, documented, usable language with LLVM backend and no metaprogramming capabilities.
- [ ] Basic optimizations operating on IR.
- [ ] Research and document the current state of the art solutions.
- [ ] Design the model of compile time evaluation, reflection, and macros.
- [ ] Implement the design above.
- [ ] Extra backends (Cranelift, GCC, custom).
- [ ] More advanced optimizations.

## Useful commands

Inspect the intermediate representation in the contro-flow-graph form (requires `graphviz`):

```
cargo run -- cfg <file> > /tmp/cfg.dot && dot -Tsvg /tmp/cfg.dot -o /tmp/cfg.svg && open /tmp/cfg.svg
```

Print the LLVM IR:

```
cargo run -- llvm-ir <file>
```

Compile and run

```
cargo run -- llvm-obj <file> && cc a.out.o && ./a.out
```
