# -*- coding: utf-8 -*-
"""
把CT重建查看器打包成一个独立的 Windows exe。

    .venv\\Scripts\\python.exe build_recon_viewer_exe.py            # 文件夹版
    .venv\\Scripts\\python.exe build_recon_viewer_exe.py --onefile   # 单文件 exe

这个跟 build_exe.py (投影生成器) 是两个独立的打包目标。加了"跟参考STL算SSIM"
功能之后, 这个查看器又需要Warp做GPU ground truth占据判定了 (`--collect-all warp`
重新加回来), 所以体积不再比投影生成器小太多——省不掉Warp那部分了；如果你不用
SSIM功能(只是导入/重建/看等值面), Warp的JIT编译器不会被触发, 但exe本身还是
带着它。

打完之后自己跑一遍 `exe --selftest`（真的做一次等值面提取+一次ASTRA重建+一次
跟合成参考网格的SSIM对比, 不是只看窗口能不能开）。
"""

import argparse
import os
import shutil
import subprocess
import sys

NAME = "CT4PiReconViewer"
ENTRY = "recon_viewer_gui.py"

COLLECT_ALL = ["trimesh", "astra", "skimage", "warp"]
COLLECT_DATA = ["matplotlib"]
HIDDEN = [
    "scipy.spatial.transform._rotation_groups",
]
EXCLUDE = [
    "warp.tests", "warp.examples", "warp.jax_experimental", "warp.render",
    "warp.fem", "warp.sim",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onefile", action="store_true", help="打成单文件 (默认文件夹版)")
    ap.add_argument("--console", action="store_true", help="保留控制台窗口 (调试用)")
    ap.add_argument("--clean", action="store_true", help="先清掉 build/ dist/ 里这个目标的产物")
    ap.add_argument("--skip-selftest", action="store_true", help="跳过打包后的冒烟测试")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    python = os.path.join(root, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(python):
        python = sys.executable

    if args.clean:
        shutil.rmtree(os.path.join("build", NAME), ignore_errors=True)
        shutil.rmtree(os.path.join("dist", NAME), ignore_errors=True)
        spec = f"{NAME}.spec"
        if os.path.isfile(spec):
            os.remove(spec)

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

    print("跑冒烟测试 (--selftest, 真的做一次等值面提取+ASTRA重建+SSIM对比)...")
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
