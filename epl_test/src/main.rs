use std::io;
use std::path::Path;

struct Stats {
    success: u32,
    failed: u32,
}

fn main() {
    let mut args = std::env::args();
    let arg0 = args.next().unwrap();

    let Some(epl_exe) = args.next() else { usage(&arg0) };
    let Some(test_dir) = args.next() else { usage(&arg0) };

    if args.next().is_some() {
        usage(&arg0);
    }

    let mut stats = Stats { success: 0, failed: 0 };
    run_tests_in_dir(&mut stats, Path::new(&test_dir), &epl_exe);

    println!("DONE:");
    println!("passed: {}", stats.success);
    println!("failed: {}", stats.failed);
    println!("tatal: {}", stats.success + stats.failed);

    if stats.failed > 0 {
        std::process::exit(1);
    }
}

fn usage(arg0: &str) -> ! {
    println!("USAGE: {arg0} [epl exe] [test dir]");
    std::process::exit(1);
}

fn run_tests_in_dir(stats: &mut Stats, dir: &Path, epl_exe: &str) {
    for entry in std::fs::read_dir(dir).expect("could not read tests dir") {
        let entry = entry.expect("could not read entry in the tests dir");
        let path = entry.path();
        if path.is_dir() {
            run_tests_in_dir(stats, &path, epl_exe);
        } else if path.extension().is_some_and(|ext| ext == "epl") {
            run_tests_on_src(stats, &path, epl_exe);
        }
    }
}

fn run_tests_on_src(stats: &mut Stats, path: &Path, epl_exe: &str) {
    let result = std::process::Command::new(epl_exe)
        .arg("ir_tree")
        .arg(path)
        .output()
        .expect("compiler invocation failed");

    if !result.status.success() {
        stats.failed += 1;
        println!(
            "{}/IR_TREE FAIL: compiler existed with non zero exit code",
            path.display()
        );
        return;
    }

    let Ok(result_str) = String::from_utf8(result.stdout) else {
        stats.failed += 1;
        println!("{}/IR_TREE FAIL: compiler produced non-utf8 output", path.display());
        return;
    };

    let ir_tree_path = path.with_added_extension("ir_tree");
    match std::fs::read_to_string(&ir_tree_path) {
        Ok(expected_ir_tree) => {
            if expected_ir_tree == result_str {
                stats.success += 1;
                println!("{}/IR_TREE PASS", path.display());
            } else {
                stats.failed += 1;
                println!("{}/IR_TREE FAIL: outptu does not match", path.display());
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            std::fs::write(ir_tree_path, result_str).expect("could not write expcted ir_tree output");
        }
        Err(e) => {
            panic!("could not read expected ir_tree output: {e:?}");
        }
    }
}
