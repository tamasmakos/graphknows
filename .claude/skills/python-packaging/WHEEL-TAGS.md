# Wheel compatibility tags & binary extensions

Filename: `{distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl`. Spaces become `_`. `pip-23.3.1-py3-none-any.whl` = any Python 3, no ABI dependency, any platform — the pure-Python shape.

| Component | Examples | Meaning |
|---|---|---|
| Python tag | `py3`, `cp39`, `pp39` | Implementation + version (`py` generic, `cp` CPython, `pp` PyPy, `ip` IronPython, `jy` Jython) |
| ABI tag | `none`, `abi3`, `cp39` | Required Python ABI; `abi3` = CPython stable ABI |
| Platform tag | `any`, `win32`, `manylinux_2_24_x86_64`, `musllinux_1_2_x86_64`, `macosx_11_0_arm64` | OS/arch |

**manylinux** — `manylinux_x_y_arch` means glibc x.y or newer. Legacy aliases: `manylinux1` (2.5), `manylinux2010` (2.12), `manylinux2014` (2.17); the first two are EOL. Forward-compatible, never backward.

**musllinux** — `musllinux_x_y_arch`, for musl libc (Alpine).

**macOS** — `macosx_x_y_arch`, with multi-arch identifiers `universal2` (arm64 + x86_64), `intel`, `fat`. Android (`android_apilevel_abi`) and iOS tags are recent additions.

**abi3** — compile against the Limited API (`#define Py_LIMITED_API`) and one `cp3X-abi3-…` wheel covers every later Python 3.x, collapsing the build matrix to one row per OS/arch. Drop it when you need unstable C API. Wheels ship no `.pyc`; installers generate them.

**purelib vs platlib** — wheels are laid out by sysconfig path, so pure and platform-specific content install to the right roots.

**Building the matrix** — use **cibuildwheel** in CI: one wheel per Python version × OS × arch, unless `abi3` collapses the version axis. multibuild is the older alternative. Always ship an sdist as the fallback for platforms you don't build.
