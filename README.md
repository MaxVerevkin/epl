# My Final Year Project

<details>
<summary> Original project proposal </summary>

Title: Design and Implementation of an Ahead-of-Time Compiled Systems Programming Language with Advanced Compile-Time Execution and Modular Code Generation Backends.

Motivation: Many modern languages implement some sort of compile-time evaluation and macro system. I would like to explore the state-of-the-art implementations, their abilities and limitations. Then I want to design a minimal language (akin to C) but with powerful meta-programming capabilities. To make it “useful” I would support calling external C libraries, and to make the generated executables performant, I will incorporate LLVM / Cranelift / GCC for the machine code generation. However I would also include a minimal aarch64 (and maybe x86_64) backend to make the project self-sufficient. Note that those backends do not have a goal to be anywhere near performant.
</details>

## Docs

- [Syntax reference](doc/syntax.md)
- [Language reference](doc/language.md)
- [Compiler architecture](doc/compiler.md)
- [Building with LLVM](doc/llvm.md)

## Useful commands

To run tests:

```
cargo b && cargo r -p epl_test ./target/debug/epl examples tests
```

Inspect the intermediate representation in control-flow-graph form (requires `graphviz`):

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
