# LLVM Setup

```sh
# Create installation directory, this is just example.
mkdir /opt/llvm-21.1.5

# Get the source code
mkdir /opt/llvm-src
git clone https://github.com/llvm/llvm-project -b llvmorg-21.1.5

# Configure LLVM build
cd llvm-project/llvm
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/llvm-21.1.5 -DCMAKE_BUILD_TYPE=Release

# Compile and install LLVM
cmake --build . --target install -j 8
```

# Compiling EPL with LLVM

`LLVM_SYS_211_PREFIX` environment variable must point to the LLVM's instalation directory during the build, e.g.

```
LLVM_SYS_211_PREFIX="/opt/llvm-21.1.5" cargo build
```

For convenience, you may set in in your local `.cargo/config.toml` under `[env]` section.
