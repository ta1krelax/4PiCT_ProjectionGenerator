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

### 打包成独立 exe

不想装 Python 环境的话，可以打包成一个独立的 Windows exe（文件夹版，双击
`CT4PiProjector.exe` 就能跑，不需要另装 Python/依赖；仍然需要 NVIDIA 显卡+
驱动才能用 GPU，没有的话自动退回 CPU）：

```powershell
.venv\Scripts\python.exe build_exe.py
```

产物在 `dist\CT4PiProjector\`（整个文件夹一起分发，别只拷 exe）。打包脚本
打完会自动跑一次 `--selftest`（真的加载一个STL、跑一次GPU光追预览，不是只看
窗口能不能开），失败会报非零退出码。加 `--onefile` 可以打单文件版，但体积大
（warp-lang 自带完整JIT编译工具链），启动会明显更慢。

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
10. **样品评估**：对已加载的STL计算一组形态学参数（16个，按对"需要多少投影角度N"
    的预测重要性排序），1-7 默认勾选，8-16 默认不勾选，可以自己增减。点"计算并
    导出形态参数"后，连同当前的生成参数(SOD/放大倍率/像素尺寸/旋转中心等)一起
    写入输出目录下的 `shape_metrics.json`。
    - 1 Projection number N：直接读当前设的投影角度数，不是算出来的。
    - 2/4/6/7 L1/T5、Major dimension L1、Local thickness T5、L2/L1、L3/L2：
      L1/L2/L3 是 Krumbein/Sneed-Folk 意义下的三个互相垂直的最大尺度轴（最长轴、
      垂直于它的最长轴、再垂直于前两者的最长轴）；T5 是局部厚度分布的第5百分位
      （Hildebrand & Rüegsegger 1997 的最大内切球方法，体素化+距离变换实现，
      不是简单的"到最近表面距离"，有做"覆盖球"修正）。
    - 3 Equivalent diameter / Volume：D_eq=(6V/π)^(1/3)。
    - 8 Sphericity Ψ：Wadell(1932)公式, Ψ=π^(1/3)(6V)^(2/3)/A。
    - 9 SA/V：表面积/体积。
    - 10 Porosity：假设最大体积的壳是外壳，其余壳都是内部空腔，
      porosity=空腔体积/外壳体积（对多壳网格有效，单壳网格返回0）。
    - 11 Convexity/Solidity：V/凸包体积、凸包面积/实际面积。
    - 12 Euler characteristic：直接用 trimesh 的 `euler_number`。
    - 13 Surface roughness：**近似值** —— 原网格顶点相对Taubin平滑后网格的
      位移 Sa/Sq，不是真正的光学轮廓仪测量（任意实体没有一个"参考平面"）。
    - 14 Fractal dimension：体素化后做 box-counting，光滑物体在有限体素分辨率
      下算出来不会精确等于理论值(如实心球应趋于3)，这个参数本身在源表格里也
      标注"第一版不建议"，结果只适合同分辨率下样品间的相对比较。
    - 15 Tortuosity：**近似代理指标** —— 真正的多孔介质弯曲度需要一个流动通道，
      对任意实心样品没有标准定义；这里算的是"表面测地距离/直线距离"（两个
      L1端点之间沿网格表面走的最短路径长度，除以它们的直线距离），仅供参考。
    - 16 Principal-axis orientation：惯性主轴方向 + 相对XYZ全局轴的夹角。
    - 计算 T5/fractal dimension 需要先把STL体素化，面板里可以调"体素分辨率"
      （默认64，沿最长边的体素数，越大越准但越慢）。
11. **序列生成**：做"采样密度扫描"实验时，不用每次手动改N再点一次生成——填
    多段 `(起始N, 结束N, 步长)`（比如 100-1000 步长100，1000-3000 步长200），
    点"预览N列表"看合并去重后的完整N值列表，确认没问题再点"开始批量生成序列"。
    - 每个N单独生成到 `out_dir/N_00100/` 这样的子文件夹（5位数字补零），互不
      覆盖；其余参数（几何/采样方式/mu等）全部沿用当前设置，只有N变。
    - 后台线程跑，不卡界面；有总进度条（第几个N）和当前N内部进度条两层。
    - 勾选"跳过已生成过的N"（默认勾选）：某个N的子文件夹里已经有
      `projections.npy` 就直接跳过，方便序列跑到一半中断后接着跑，不用从头来。
    - 中途可以点"停止"，当前正在跑的N会跑完，后面还没开始的N不再跑。
    - 全部跑完（或被停止）后，在 `out_dir` 下生成一份 `sequence_summary.json`
      汇总表：每个N对应的状态（ok/skipped_existing/failed/cancelled）、子文件夹
      路径、耗时；单个N失败不会中断整个序列，错误详情记在对应条目里。

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

形态参数模块也用解析形状验证过：实心球（R=5mm）算出 sphericity≈0.9998、
SA/V≈0.6006（理论3/R=0.6）、solidity/convexity≈1.0、euler=2、L1≈L2≈L3≈10mm、
局部厚度T5≈10.2mm（符合"整个实心球都是一个最大内切球, 所以处处厚度=直径"的
理论预期）；长方体（10x8x6mm）算出 L1=14.142mm（=空间对角线）、solidity=1.0、
SA/V 精确匹配解析值；用两层同心球壳模拟含内部空腔的样品，porosity 算出0.216，
和解析值(4/3π3³)/(4/3π5³)=0.216 一致。

注：Local thickness 对尖锐直角边非常敏感——一个没有圆角的长方体，其5%分位
厚度会因为8个角/12条棱附近的"最大内切球"迅速缩小到体素尺度而显得很小，这是
Hildebrand-Rüegsegger算法本身的正确行为（棱角处确实局部很"薄"），不是bug；
真实CAD零件通常有圆角，不会这么极端。

## 已知限制 / 后续可扩展

- 假设模型是单一均匀材料（mu 是一个标量），STL 本身不带材料信息。
- 非水密网格（non-watertight STL）会导致路径长度积分出错，目前只做提示，不
  自动修复；需要的话可以后续接 `trimesh.repair` 或 pymeshfix。
- 旋转中心目前只能用数值输入，暂不支持在 3D 预览里鼠标点选。
- 未加噪声模型（Poisson noise 等），当前输出是理想无噪声的线积分。
- 采样方案 / 重建算法都是可插拔设计（`geometry.SAMPLING_SCHEMES` 字典），后续
  接 ASTRA 重建、对比 SIRT/CGLS 对稀疏 4π 采样的鲁棒性，可以直接在这套代码上扩展。
- 形态参数里 tortuosity 用 networkx 在网格的顶点图上跑最短路径，顶点数很大
  （几万以上）时可能明显变慢；local thickness 的体素网格分辨率越高也越慢——
  面板里的"体素分辨率"和要不要勾选这两项，都可以按需要调整。
- surface roughness / fractal dimension / tortuosity 三个都是**代理指标**
  （没有对任意实体的标准定义），已经在README和结果的 `note` 字段里写清楚了
  各自的近似方式，不要直接当成物理光学轮廓仪/多孔介质弯曲度的测量值来用。
