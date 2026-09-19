# -*- coding: utf-8 -*-
"""
把CT 4pi投影生成器打包成一个 Windows exe。

    .venv\\Scripts\\python.exe build_exe.py            # 文件夹版 (dist/CT4PiProjector/)
    .venv\\Scripts\\python.exe build_exe.py --onefile   # 单文件 exe

默认打文件夹版而不是单文件：warp-lang 运行时会JIT编译CUDA/C++核函数, 打成
单文件每次启动都要先解压整个依赖树(几百MB, PySide6+matplotlib+scipy+warp的
native工具链)到临时目录, 启动会明显变慢；warp自己的kernel cache在
%LOCALAPPDATA%\\NVIDIA\\warp\\Cache 下, 跟打包模式无关, 不受影响。

几个打包上的坑:
  · warp-lang 除了.py代码还带一堆native二进制/头文件(JIT编译工具链), 普通
    PyInstaller静态扫描找不全, 必须用 --collect-all warp。
  · trimesh 也带一些资源文件(resources/*.json等), 同样用 --collect-all trimesh。
  · matplotlib 的字体等数据文件用 --collect-data matplotlib 保底。
  · **重要**: warp 的 @wp.kernel 装饰器在导入时要用 inspect/linecache 读被装饰
    函数的*真实源码文本*去做AST转译, 但PyInstaller打包后的模块只是编译好的
    字节码, 压缩在归档里没有可读的.py文件——装饰器只记得一个裸文件名(比如
    "raytrace_gpu.py", 不带目录), 运行时靠当前工作目录去找它。以前只是"侥幸
    没崩"：本脚本自己的冒烟测试是从项目源码目录里调用exe的, 所以那个裸文件名
    刚好能在cwd里找到真实的 raytrace_gpu.py, 掩盖了问题; 真的双击部署好的exe
    (cwd是exe自己的目录, 没有源码)就会崩 (RuntimeError: Directly evaluating
    Warp code defined as a string...)。修法是两步都做: (1) projector_gui.py
    启动时如果是frozen就强制chdir到exe所在目录 (兜底--onefile时chdir到
    _MEIPASS), (2) 这个脚本把 raytrace_gpu.py 真实源码文件复制到exe旁边(以及
    --onefile模式下用 --add-data 打包进临时解压目录), 让上面那个裸文件名查找
    总能命中真实文件。

打完之后会自己跑一遍 `exe --selftest`（真的加载STL、跑一次GPU光追预览, 不是
只看窗口能不能开), 而且是从一个跟项目源码无关的临时目录里调用exe的 (故意
不让"cwd里恰好有源码"这种侥幸情况发生, 逼真模拟用户双击运行)，看退出码；
过不了就不算打包成功。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

NAME = "CT4PiProjector"
ENTRY = "projector_gui.py"

COLLECT_ALL = ["warp", "trimesh", "astra", "skimage"]
COLLECT_DATA = ["matplotlib"]
HIDDEN = [
    "scipy.spatial.transform._rotation_groups",
    "networkx",
]
# Needs its real .py source on disk at runtime -- see the big comment above and in
# projector_gui.py (warp's @wp.kernel needs inspect/linecache-readable source text).
RUNTIME_SOURCE_FILES = ["raytrace_gpu.py"]
# --collect-all warp pulls in warp's whole package tree (tests/examples/jax/render/usd
# submodules included) even though we only ever use its core kernel + mesh-query API.
# None of this project's code imports these, so excluding them is safe and cuts bloat.
EXCLUDE = [
    "warp.tests", "warp.examples", "warp.jax_experimental", "warp.render",
    "warp.fem", "warp.sim",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onefile", action="store_true", help="打成单文件 (默认文件夹版)")
    ap.add_argument("--console", action="store_true", help="保留控制台窗口 (调试用)")
    ap.add_argument("--clean", action="store_true", help="先清掉 build/ dist/")
    ap.add_argument("--skip-selftest", action="store_true", help="跳过打包后的冒烟测试")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    python = os.path.join(root, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(python):
        python = sys.executable

    if args.clean:
        for d in ("build", "dist"):
            shutil.rmtree(d, ignore_errors=True)

    cmd = [
        python, "-m", "PyInstaller", "--noconfirm",
        "--name", NAME,
        "--onefile" if args.onefile else "--onedir",
        "--console" if args.console else "--windowed",
    ]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    for pkg in COLLECT_DATA:
        cmd += ["--collect-data", pkg]
    for mod in HIDDEN:
        cmd += ["--hidden-import", mod]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    for src in RUNTIME_SOURCE_FILES:
        # covers --onefile (lands in the sys._MEIPASS extraction dir); --onedir gets a
        # second, more reliable copy placed directly next to the exe further below.
        cmd += ["--add-data", f"{src};."]
    cmd.append(ENTRY)

    print("打包命令：\n  " + " ".join(cmd) + "\n")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("PyInstaller 失败")
        return r.returncode

    exe = (
        os.path.join("dist", NAME, NAME + ".exe")
        if not args.onefile
        else os.path.join("dist", NAME + ".exe")
    )
    if not os.path.exists(exe):
        print(f"没找到产物 {exe}")
        return 1

    if not args.onefile:
        # --onedir: put a real copy of the source right next to the exe. cwd on a
        # normal double-click launch is the exe's own directory, so this is what
        # linecache actually finds -- --add-data above lands inside _internal/, which
        # is NOT on that lookup path.
        exe_dir = os.path.dirname(exe)
        for src in RUNTIME_SOURCE_FILES:
            shutil.copy2(src, os.path.join(exe_dir, src))

    size = os.path.getsize(exe) / 1e6
    print(f"\n产物：{exe}  ({size:.1f} MB)")

    if args.skip_selftest:
        return 0

    print("跑冒烟测试 (--selftest, 真的加载STL+跑一次GPU光追预览, 从跟源码无关的目录里调用)...")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    # Run from an unrelated cwd (not the project root) so a bug like "only works because
    # cwd happens to contain the source checkout" can't hide from this test again.
    r = subprocess.run(
        [os.path.abspath(exe), "--selftest"],
        capture_output=True, timeout=180, env=env, cwd=tempfile.gettempdir(),
    )
    ok = r.returncode == 0
    print(
        ("  通过：" if ok else "  失败：")
        + (r.stdout or b"").decode("utf-8", "replace").strip()
        + "\n"
        + (r.stderr or b"").decode("utf-8", "replace").strip()[-2000:]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
