# Language Reference

This document describes the language semantics implemented by the EPL compiler.
See also `doc/syntax.md` for the grammar-level syntax reference.

EPL is a single-file, statically typed, expression-based, ahead-of-time
compiled language. It currently supports mutable local variables, functions,
structs, fixed-size arrays, raw pointers, external function declarations, and
compile-time evaluation.

## Program Structure

An EPL source file is a sequence of top-level items:

- Function definitions and declarations.
- Struct definitions.

Functions and types use separate global namespaces:

- Function names must be unique in the function namespace.
- Struct names live in the type namespace and must not duplicate built-in types
  or earlier struct names.
- Built-in types are always present in the type namespace.
- Function declarations are collected before function bodies are checked, so
  functions may call later functions and may be recursive.
- Struct definitions are processed in source order. A struct field may refer
  only to a type that is already known when the struct is defined.
- Function signatures and bodies may use any struct type declared in the file,
  because all struct definitions are collected before functions are lowered.

## Types and Values

Every expression has a statically known type. EPL does not perform implicit
numeric conversions.

Built-in types:

- `unit`: the zero-sized type used for expressions that produce no meaningful
  value.
- `bool`: boolean values `true` and `false`.
- `i8`, `i32`, `i64`: signed integers.
- `u8`, `u32`, `u64`: unsigned integers.
- `ptr`: an opaque pointer type.
- `!`: the never type, used for expressions that do not complete normally.

Composite types:

- `*T`: a typed pointer to `T`.
- `[T; N]`: an array of `N` elements of type `T`.
- User-defined structs.

Array lengths in type expressions must currently be number literals. Signed and
unsigned integer types of the same width are distinct types.

Integer literals have one of the supported integer types. A suffix selects the
type explicitly. Without a suffix, an integer literal uses the expected integer
type when there is one, otherwise it defaults to `i32`.

String literals have type `*i8`. They are intended for simple C-style string
arguments and are not supported as compile-time constants.

The `undefined` literal requires an expected type from context and produces an
undefined value of that type.

## Variables, Scope, and Inference

`let` introduces a mutable local variable. Assignment and compound assignment
may update local variables, function arguments, and other place expressions.

Blocks introduce lexical scopes. Name lookup starts in the current scope and
then searches outer scopes. A later `let` with the same name shadows an earlier
binding for subsequent uses in that scope.

Supported `let` forms:

- `let name = expr;` infers the variable type from `expr`.
- `let name: T = expr;` checks `expr` against `T`.
- `let name: T;` declares storage of type `T` without an initializer.

Reading a local value before assigning to it is undefined.

Type inference is local and context-driven. Expected types flow into these
expressions:

- Typed `let` initializers.
- Assignment right-hand sides.
- Function call arguments that correspond to fixed parameters.
- Function return expressions.
- Block final expressions.
- `if` branches when an expected result type is known.
- Unnamed struct initializers.
- Array initializers with an expected array type.
- `undefined`.
- Unsuffixed integer literals.

Additional inference rules:

- For binary arithmetic and comparison operators, the left operand determines
  the expected type of the right operand.
- For `for i in start..end`, `start` must be an integer expression. Its type is
  used for `end`, the hidden loop counter, and the visible loop variable.
- For an array initializer with no expected type, the element type is inferred
  from the first element expression that has a normal value type. If no element
  provides such a type, the element type is `!`.

Type conflicts are compile-time errors.

## Functions and ABI

Function parameters must have explicit types. A return type may be omitted, in
which case it is `unit`. Argument names must be unique, and function arguments
behave as mutable local variables inside the function body.

A function may have a body or may be a declaration ending in `;`. A declaration
without a body names an external function that may be called from EPL if the
final program supplies a compatible definition.

Calls to non-variadic functions must provide exactly the declared arguments.
Calls to variadic functions must provide at least the fixed arguments declared
by the callee. EPL may declare and call variadic functions, but defining a
variadic function body is not supported.

The compiler enforces this entry-point signature:

```epl
fn main() -> i32
```

EPL code can call any EPL function signature supported by the language. The C
ABI is not fully implemented: C FFI currently works only for a subset of
external function signatures. Existing examples use simple calls such as
`printf` and `exit`, but this should be treated as limited coverage rather than
complete ABI support. By-value aggregate arguments or returns, exact C `void`
and noreturn behavior, and other platform-specific ABI details should not be
assumed to match C.

Function names are used as written. EPL has no overload-dependent name
resolution and no name mangling.

## Annotations

Annotations are parsed on top-level items, but the only accepted annotation is
`@pure` on functions. Unknown annotations are rejected, and struct annotations
are not supported.

A `@pure` function must have a body. Pure functions may be evaluated by
`comptime` when called from a compile-time expression, subject to the purity
rules in [Compile-Time Evaluation](#compile-time-evaluation).

## Expressions

Expressions are evaluated in source order. Function call arguments, array
initializer elements, and struct initializer fields are evaluated left to right.

### Blocks

A block evaluates its statements in order. If the block ends with a final
expression, the block's type and value are the type and value of that
expression. If there is no final expression, the block has type `unit`.

Expression statements evaluate their expression and discard the value. If a
block is used in a context that expects a non-`unit` type, the block must end in
a final expression compatible with that type.

### Control Flow

`return` exits the current function and has type `!`. Returning without a value
is valid only in a function whose return type is `unit`, otherwise the returned
expression must match the function return type.

`if` conditions must be `bool`. An `if` without `else` is valid only where a
`unit` result is acceptable, its false branch is then `unit`. The branch types
must match.

`loop` repeats its body until control leaves through `break`, `return`, or
another `!` expression. A loop with no reachable `break` has type `!`.
Otherwise, its type is the common type of its `break` values. A plain `break`
contributes `unit`.

`break` and `continue` are valid only inside loops and have type `!`. All
breaks for the same loop must be compatible with the loop's expected result
type.

`while` checks its boolean condition before each body execution and stops when
the condition is false. A `while` expression has type `unit`.

`for` currently supports only half-open integer ranges of the form
`start..end`. The start and end expressions are evaluated once before the loop.
The loop continues while the current value is less than the end value and
increments by one each iteration. The visible loop variable is a per-iteration
copy, assigning to it does not change the hidden loop counter. A `for`
expression has type `unit`.

Range expressions are only supported as `for` iterators.

### Operators

Arithmetic operators are supported for integer operands only. Both operands
must have the same type, and the result has that integer type. Supported
operators are addition, subtraction, multiplication, division, remainder, and
unary negation for signed integers.

Signedness affects division, remainder, comparison, and integer extension during
casts.

Compound assignment performs the corresponding integer arithmetic operation in
place and has type `unit`.

Comparison operators return `bool`. Integer values support all comparison
operators. `bool` values support equality and inequality. Pointer comparisons
are not currently supported.

`&&` and `||` require `bool` operands and return `bool`. They short-circuit:

- `a && b` evaluates `b` only when `a` is true.
- `a || b` evaluates `b` only when `a` is false.

`!x` requires `x: bool` and returns the logical negation.

## Data Access and Mutation

Assignment has type `unit`. The left-hand side must be a place expression.

Supported places:

- Local variables.
- Struct fields of places.
- Array elements of places.
- Dereferenced typed pointers.

Field access is valid only on struct values. Array indexing is valid only on
array values. Array indices must be `u64`, and indexing does not perform bounds
checks.

### Pointers

`&place` returns a typed pointer to a place. The operand must be a place.

`ptr.*` dereferences a typed pointer. Dereferencing the opaque `ptr` type is
rejected because it has no pointee type.

Pointer operations are raw. The language does not implement lifetime checking,
alias checking, null checking, pointer arithmetic, or bounds checking.

### Structs

Struct field names must be unique within a struct definition.

Struct initializers must provide every field exactly once. Field order in the
initializer does not have to match the definition order. Duplicate fields,
missing fields, and unknown fields are rejected.

An initializer with an explicit struct name determines its own type. An
initializer without a struct name requires an expected struct type from context.

Struct values are passed, returned, assigned, and stored by value.

### Arrays

Array initializers create array values. If an expected array type is present,
the initializer length must match it and every element must match the expected
element type.

Without an expected array type, the element type is inferred from the elements.
Array values are passed, returned, assigned, and stored by value.

### Casts

The supported casts are:

- Integer to integer.
- Pointer to pointer.

Integer-to-integer casts resize the value to the destination width. Narrowing
casts truncate. Widening casts sign-extend signed source integers and
zero-extend unsigned source integers.

Pointer-to-pointer casts preserve the pointer value. Casts between pointers and
integers are not supported.

## Compile-Time Evaluation

`comptime expr` evaluates `expr` during compilation and replaces it with the
resulting constant value. The expression is type checked in its surrounding
expected type before it is evaluated, and it must satisfy the compile-time
purity rules.

A `comptime` expression may use:

- Literals except string literals.
- Arithmetic, comparison, boolean operators, and integer casts.
- Blocks, conditionals, and loops.
- Local variables and assignment to locals declared inside the `comptime`
  expression.
- Arrays and structs.
- Calls to `@pure` functions.
- `break` and `continue` for loops inside the `comptime` expression.

A `comptime` expression may not:

- Access function arguments or variables from the surrounding run-time context.
- Use string literals.
- Take addresses, dereference pointers, or otherwise use pointer operations.
- Call non-`@pure` functions.
- Return from the enclosing function.

Pure functions are checked with similar restrictions. They may use their own
arguments, local variables, local mutation, and normal function `return`, they
may call only other pure functions. They may not use string literals, take
addresses, or dereference pointers.

A compile-time result must be representable as a compile-time constant. The
supported compile-time constant forms are `unit`, `bool`, integers, arrays,
structs, and `undefined` values. Pointer and string values are not supported as
compile-time constants.

## Current Language Limits

The most important current limits are:

- Only one source file is compiled at a time.
- There are no modules or imports.
- There is no implicit conversion between numeric types.
- Array type lengths must be number literals, not arbitrary constant
  expressions.
- `for` iteration only supports half-open integer ranges.
- Range expressions have no meaning outside `for`.
- Variadic functions can be declared but not defined.
- The C ABI is implemented only for a subset of external function signatures.
- There are no array bounds checks or pointer safety checks.
- `comptime` cannot inspect or capture surrounding run-time state.
