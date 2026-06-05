## Docs

- [Syntax reference](doc/syntax.md)
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
