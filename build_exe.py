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

打完之后会自己跑一遍 `exe --selftest`（真的加载STL、跑一次GPU光追预览, 不是
只看窗口能不能开)，看退出码；过不了就不算打包成功。
"""

import argparse
import os
import shutil
import subprocess
import sys

NAME = "CT4PiProjector"
ENTRY = "projector_gui.py"

COLLECT_ALL = ["warp", "trimesh", "astra", "skimage"]
COLLECT_DATA = ["matplotlib"]
HIDDEN = [
    "scipy.spatial.transform._rotation_groups",
    "networkx",
]
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
    size = os.path.getsize(exe) / 1e6
    print(f"\n产物：{exe}  ({size:.1f} MB)")

    if args.skip_selftest:
        return 0

    print("跑冒烟测试 (--selftest, 真的加载STL+跑一次GPU光追预览)...")
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    r = subprocess.run([exe, "--selftest"], capture_output=True, timeout=180, env=env)
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
