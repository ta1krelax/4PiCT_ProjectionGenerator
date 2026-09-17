# CT 4π 锥束投影生成器 (CT 4π Projector)

从 SolidWorks 导出的 STL 模型出发，生成 4π 空间任意方向采样的锥束 X-ray 投影
(cone-beam projection)，用于后续 4π 采样密度 / 采样方式 vs 样品复杂度的重建实验。

核心方法论：投影生成阶段直接对 STL 三角网格做**解析光线-三角面片求交**
（GPU 加速，NVIDIA Warp 的 BVH 光追），按 Beer-Lambert 定律累加射线穿过实体的
路径长度得到吸收值 —— 不对模型体素化，因此和后续 ASTRA 重建用的离散化模型是
两套独立体系，避免 inverse crime 导致重建效果被"美化"。

基于 Python + NVIDIA Warp (GPU 光追) + trimesh (STL 处理) + PySide6 (GUI) + matplotlib (预览)。

## 安装

本机已经建好独立的虚拟环境 `.venv`（Python 3.14，含 GPU 版 warp-lang），
直接用它运行即可，不需要重新装：

```powershell
cd ct_4pi_projector
.venv\Scripts\python.exe projector_gui.py
```

在别的机器上用，先建虚拟环境装依赖（需要 NVIDIA 显卡 + 驱动才能用 GPU 加速，
没有 GPU 时 Warp 会自动退回 CPU 计算，结果一致只是慢一些）：

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe projector_gui.py
```

## 界面结构

主窗口只放三块内容：**中间 3D 模型预览**（左键拖拽旋转模型本身、坐标轴不动、
滚轮缩放）、**右侧投影预览画廊**（每次点"预览单张投影"新增一张缩略图，双击
弹出大图看细节/悬浮读像素值）、**下方日志**。

所有参数都收在顶部一行 tab 按钮里（输入/输出、旋转中心、锥束几何、材料、
采样与设备），点一下弹出一个独立小窗口调那部分参数，再点一下或关掉那个小
窗口就收起；几个参数窗口互不影响，可以同时开几个对照着调。

## 使用说明

1. **输入 / 输出**：点顶部"输入 / 输出" tab 弹出窗口，选择 STL 文件，点
   "加载并预览模型"（中间 3D 视图会显示网格，窗口里显示顶点数/面片数/
   水密性/包围盒/质心）；再选择输出目录。
   - 如果提示"非水密"，说明 STL 导出时可能不是严格的封闭实体（SolidWorks 导出
     STL 常见问题），路径长度积分可能有误差，建议回 CAD 里检查实体导出设置。
2. **旋转中心**：默认用模型质心（几何中心，自动算好），也可以切到"包围盒中心"
   或"手动输入" X/Y/Z 自己填。这个点就是 4π 采样时锥束绕转的中心，也是保存的
   ASTRA 几何向量的坐标原点。
3. **锥束几何（放大倍率）**：填 SOD（源到旋转中心的距离）和放大倍率 M
   （= SDD/SOD，SDD 会自动算出并显示），对应"beam source 和 detector 的相对
   位置"。
4. **像素密度**：探测器像素物理尺寸（mm/px）+ 分辨率（行 x 列），决定输出图像
   的像素密度和物理覆盖范围。
5. **材料**：给一个线性吸收系数 mu（1/mm），STL 本身没有材料信息，这里假设
   单一均匀材料的实体。
6. **4π 采样方案**：
   - `fibonacci_4pi`：全 4π 球面近均匀采样（推荐，用于采样密度实验）
   - `random_4pi`：全 4π 球面随机采样
   - `circular_x/y/z`：单轴 360° 圆周扫描（传统 CT 基线，用于对比/快速验证几何是否正确）
   - 填投影角度数 N。
7. **计算设备**：有 GPU 就选 cuda:0（自动检测），没有就用 cpu，两者输出结果一致。
8. **预览单张投影**：用当前参数只算第一个采样方向的投影，右侧显示灰度图，
   用来在跑全部之前先确认几何/曝光参数（放大倍率、mu、分辨率）是否合理。
9. **生成全部 4π 投影**：后台线程批量计算全部 N 个投影（不卡界面，有进度条），
   完成后保存到输出目录：
   - `projections.npy`：形状 `(N, rows, cols)`，float32，值是 `mu * 路径长度`
     （即 line integral，可直接喂给 FBP/SIRT/CGLS，不用再取对数）
   - `astra_vectors.npy`：形状 `(N, 12)`，ASTRA `cone_vec` 几何格式，坐标已经
     以旋转中心为原点
   - `metadata.json`：记录所有参数（角度方案、SOD/SDD/放大倍率、像素尺寸、
     mu、旋转中心等），以及一段可以直接抄的 ASTRA 导入代码

### 导入 ASTRA 重建（下一步）

```python
import astra, numpy as np
vectors = np.load('astra_vectors.npy')
proj = np.load('projections.npy')       # (n_views, rows, cols)
rows, cols = proj.shape[1], proj.shape[2]
proj_geom = astra.create_proj_geom('cone_vec', rows, cols, vectors)
sino = np.transpose(proj, (1, 0, 2))     # astra 要的顺序是 (row, view, col)
proj_id = astra.data3d.create('-sino', proj_geom, sino)
```

## 已验证

用解析球体（半径 R=10mm）做了单元测试：过球心的射线路径长度应为 2R=20mm，
GPU 计算结果与解析值误差 <0.03mm（来自网格离散化本身，非算法误差），GPU/CPU
两种设备结果一致。GUI 端到端流程（加载模型 -> 预览 -> 批量生成 -> 落盘）已用
离屏模式跑通。

## 已知限制 / 后续可扩展

- 假设模型是单一均匀材料（mu 是一个标量），STL 本身不带材料信息。
- 非水密网格（non-watertight STL）会导致路径长度积分出错，目前只做提示，不
  自动修复；需要的话可以后续接 `trimesh.repair` 或 pymeshfix。
- 旋转中心目前只能用数值输入，暂不支持在 3D 预览里鼠标点选。
- 未加噪声模型（Poisson noise 等），当前输出是理想无噪声的线积分。
- 采样方案 / 重建算法都是可插拔设计（`geometry.SAMPLING_SCHEMES` 字典），后续
  接 ASTRA 重建、对比 SIRT/CGLS 对稀疏 4π 采样的鲁棒性，可以直接在这套代码上扩展。
