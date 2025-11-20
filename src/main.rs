mod ast;
mod common;
mod diagnostics;
mod entity_id;
mod ir;
mod lex;
mod llvm;

fn main() {
    let mut args = std::env::args();
    let arg0 = args.next().expect("arg0 is always present");
    let Some(command) = args.next() else {
        print_usage(&arg0, 1)
    };

    match &*command {
        "help" => {
            print_usage(&arg0, 0);
        }
        "lex" => {
            let Some(file) = args.next() else { print_usage(&arg0, 1) };
            if args.next().is_some() {
                print_usage(&arg0, 1)
            }

            let src = std::fs::read_to_string(file).unwrap();

            for x in lex::Lexer::new(&src) {
                let (span, token) = x.unwrap();
                println!("{}..{}: {token:?}", span.start, span.end);
            }
        }
        "ast" => {
            let Some(file) = args.next() else { print_usage(&arg0, 1) };
            if args.next().is_some() {
                print_usage(&arg0, 1)
            }

            let src = std::fs::read_to_string(&file).unwrap();

            let ast = ast::Parser::new(&src).parse().unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            println!("{ast:#?}");
        }
        "ir" => {
            let Some(file) = args.next() else { print_usage(&arg0, 1) };
            if args.next().is_some() {
                print_usage(&arg0, 1)
            }

            let src = std::fs::read_to_string(&file).unwrap();

            let ast = ast::Parser::new(&src).parse().unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            let ir = ir::Ir::from_ast(&ast).unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            println!("{ir:#?}");
        }
        "cfg" => {
            let Some(file) = args.next() else { print_usage(&arg0, 1) };
            if args.next().is_some() {
                print_usage(&arg0, 1)
            }

            let src = std::fs::read_to_string(&file).unwrap();

            let ast = ast::Parser::new(&src).parse().unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            let ir = ir::Ir::from_ast(&ast).unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            let graph = ir::graphviz::graph(&ir);
            print!("{graph}");
        }
        "llvm-ir" => {
            let Some(file) = args.next() else { print_usage(&arg0, 1) };
            if args.next().is_some() {
                print_usage(&arg0, 1)
            }

            let src = std::fs::read_to_string(&file).unwrap();

            let ast = ast::Parser::new(&src).parse().unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            let ir = ir::Ir::from_ast(&ast).unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            let llvm_module = llvm::LlvmModule::from_ir(&ir).unwrap_or_else(|err| {
                eprintln!("LLVM verification error: {}", err.to_string_lossy());
                std::process::exit(1);
            });

            println!("{llvm_module}");
        }
        "llvm-obj" => {
            let Some(file) = args.next() else { print_usage(&arg0, 1) };
            if args.next().is_some() {
                print_usage(&arg0, 1)
            }

            let src = std::fs::read_to_string(&file).unwrap();

            let ast = ast::Parser::new(&src).parse().unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            let ir = ir::Ir::from_ast(&ast).unwrap_or_else(|err| {
                diagnostics::print_error(&file, &src, err);
                std::process::exit(1);
            });

            let llvm_module = llvm::LlvmModule::from_ir(&ir).unwrap_or_else(|err| {
                eprintln!("LLVM verification error: {}", err.to_string_lossy());
                std::process::exit(1);
            });

            llvm_module.compile().unwrap_or_else(|err| {
                eprintln!("LLVM compilation error: {}", err.to_string_lossy());
                std::process::exit(1);
            });
        }
        other => {
            println!("Unknown command {other:?}");
            println!();
            print_usage(&arg0, 1);
        }
    }
}

fn print_usage(arg0: &str, status_code: i32) -> ! {
    println!("Usage: {arg0} <command> [command args]");
    println!();
    println!("Commands:");
    println!("  - help");
    println!("  - lex <file>");
    println!("  - ast <file>");
    println!("  - ir <file>");
    println!("  - cfg <file>");
    println!("  - llvm-ir <file>");
    println!("  - llvm-obj <file>");
    std::process::exit(status_code);
}
