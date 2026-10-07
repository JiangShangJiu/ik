# 展示复现与主页发布

完整作品正文只维护 [个人主页项目页](homepage/inverse-kinematics.md)。本文说明如何重新生成展示、打包现有素材，以及复制到 Astro 站点。

## 完整复现

需要 Python 3.10+、MuJoCo Menagerie 中的四台模型，以及中文字体和 `ffmpeg`。复现配置使用 Python 3.11.7；下列 constraints 固定直接依赖，模型固定到生成展示时使用的 commit。在仓库根目录运行：

```bash
python -m pip install -e '.[showcase,test]' -c requirements-reproduce.txt
export MUJOCO_MENAGERIE=/path/to/mujoco_menagerie
git clone https://github.com/google-deepmind/mujoco_menagerie.git "$MUJOCO_MENAGERIE"
git -C "$MUJOCO_MENAGERIE" checkout bf756430b615819654b640f321c71ba5c3ebeef8
export MUJOCO_GL=egl

python -m scripts.showcase.build
python -m pytest -q
```

模型目录可自行选择；已有克隆时跳过 `git clone`，执行相同的 `checkout` 即可。Panda 模型也支持通过 `PANDA_XML` 单独指定；本页复现使用上述锁定版本的 `franka_emika_panda/panda_nohand.xml`。

`build` 是当前展示的统一入口：重新计算实验数据，渲染结构、分支、固定末端多解、整圈可行性与末端路径，再调用 `export_homepage --from-render` 更新发布素材和本地压缩包。

原始 PNG、MP4 与中间 JSON 写入 `build/showcase/`，不纳入版本控制。页面实际引用的 WebP、MP4 和 JSON 保留在 [docs/homepage/public/](homepage/public/)，README 也引用这一份资源。`build/ik-homepage.zip` 是可重新生成的本地分发包，不提交到远程仓库。

## 打包与复制到 Astro

已有发布素材只需重新打包时运行：

```bash
python -m scripts.showcase.export_homepage
```

默认从已保存的 `docs/homepage/public/` 打包，无需模型或 OpenGL 渲染。当 `build/showcase/` 已具备页面引用的完整一套原始结果时，可用 `python -m scripts.showcase.export_homepage --from-render` 从 `build/showcase/` 转换并导出页面引用集；它会检查所有输入，不能用于仅有一组渲染的部分导入。

[inverse-kinematics.md](homepage/inverse-kinematics.md) 按现有 Astro 项目的 `projects` 字段编写，包含 `heroImage`、`featureVideo`、`resultFigures`、`highlights` 和 `math: true`。发布素材只保留该页面引用的 **37 个文件：16 张 WebP、16 段 MP4、5 份 JSON**。图片包括封面、技术图、静态构型图集及视频封面；视频包括固定末端多解、直线、末端圆周与 iiwa 整圈诊断；数据对应下文五个实验阶段。

本地生成的 `build/ik-homepage.zip` 内部使用站点的目录结构：

- `src/content/projects/inverse-kinematics.md`
- `public/assets/img/projects/inverse-kinematics/`
- `public/assets/videos/inverse-kinematics/`
- `public/assets/data/inverse-kinematics/`

将压缩包解压到个人主页仓库根目录，再按正常 Astro 流程构建、预览和发布。也可以把本仓库的 `docs/homepage/inverse-kinematics.md` 复制到站点的 `src/content/projects/`，再将 `docs/homepage/public/assets/` 下对应的三个资源目录合并到站点 `public/assets/`。正文使用 `/assets/` 绝对路径，项目页路由保持为 `/projects/inverse-kinematics/`。其他 Astro 站点需先适配自己的内容 schema 和页面组件，数学公式渲染需支持 `math: true`。

渲染和打包命令只操作本仓库，不会修改或部署个人站点。

## 管线与数据

`portfolio` 生成结构、闭式候选、Panda 几何初值与修正、短路径和近奇异对照图，记录在 [portfolio_metrics.json](homepage/public/assets/data/inverse-kinematics/portfolio_metrics.json)。这些数据属于选定目标与短路径的展示实验。

`solution_gallery` 生成四型号静态多解总览与每台八构型图集。UR5e、Lite 6 展示离散闭式候选；iiwa 按臂型角选择代表样本；Panda 扫描 q7，再从通过数值求解与验收的内部候选中选取样本。记录见 [solution_metrics.json](homepage/public/assets/data/inverse-kinematics/solution_metrics.json)。

`pose_gallery` 生成当前首页主动画：四段各 240 帧、20 fps、12 秒，共 960 帧。UR5e、Lite 6 的八个独立闭式候选各停留 1.5 秒，直接切换，不插值；iiwa 在 ψ = 6°–156° 的稳定合法展示弧往返；Panda 逐帧给定 q7，仅对其余六关节做 DLS 修正。固定末端实验的逐帧误差、限位和相邻构型变化见 [pose_metrics.json](homepage/public/assets/data/inverse-kinematics/pose_metrics.json)。六轴跳切不表示连续换支轨迹。

`swivel_gallery` 生成当前 iiwa 目标的整圈诊断图和解释动画。0.25° 网格含两端共 1441 个采样角，每角有 2–8 个合法候选，共 9414 条合法采样记录，包含周期端点，不能称为互异解数量。保留所有返回的合法候选，分别用 0.15 / 0.30 rad 邻接阈值检查连接图，并检查肘圆水平投影与肩轴、J1 行程的关系。沿当前支的最后合法采样点为 172.75°，173° 因 J7 低限越界被拒绝；最近替代候选与末个合法构型之差约 4.43 rad，只针对该下一采样角。动画前 9 秒延拓，后 3 秒停住；重播需要跳切，不能计作连续循环。结论限定于本目标、模型限位与扫描设置，数据见 [swivel_metrics.json](homepage/public/assets/data/inverse-kinematics/swivel_metrics.json)。

`motion_gallery` 生成八条固定姿态的末端路径，每条 240 帧、20 fps、12 秒，共 1920 帧。UR5e、iiwa、Panda 的直线长 1.12 m、圆半径 40 cm；Lite 6 的直线长 68 cm、圆半径 26 cm。iiwa 固定 ψ = −0.5 rad，Panda 固定 q3 = 0。轨迹、圆面方向、位姿误差、限位与循环接缝分别记录在 [path_metrics.json](homepage/public/assets/data/inverse-kinematics/path_metrics.json)。帧率用于展示，未做时间参数化或完整碰撞验证。

实验记录包含各阶段的源码与输入溯源；`portfolio_metrics.json` 另记录模型 XML 哈希和依赖版本。完整 `build` 按 `portfolio → solution_gallery → pose_gallery → swivel_gallery → motion_gallery → export_homepage` 顺序运行，后续实验读取本轮生成的数据。单独使用 `--reuse-data` 时，输入优先取 `build/showcase/`，没有本地生成记录才回退到发布 JSON；应明确自己使用的是哪份输入。数值复现不要求时间戳、绝对路径、耗时或渲染文件字节相同；重跑后需用新 JSON 核对正文摘要。

本次已从空 `build/showcase/` 完整运行构建，生成全部 37 项发布资源，并通过 115 项测试。五份 JSON 已更新为本次生成的源码与输入溯源；重建前后的数值及布尔验收结果一致。

历史方法笔记的 `docs/img/` 插图及 `scripts/visualize_ik.py` 等旧脚本另行保留；旧脚本可输出 GIF，但不属于当前主页的 37 项发布资源，也不并入上述五份实验统计。

## 离屏渲染环境

本次环境版本：Python 3.11.7、NumPy 1.26.4、MuJoCo 3.4.0、Matplotlib 3.8.0、Pillow 10.2.0、pytest 7.4.0；直接依赖约束保存在 [requirements-reproduce.txt](../requirements-reproduce.txt)。视频编码使用系统 ffmpeg 4.4.2（`libx264`），中文字体使用 Noto Sans CJK Regular / Bold。字体、驱动和编码器会影响画面细节与文件字节。

Linux 无窗口渲染使用 `MUJOCO_GL=egl`；有桌面窗口时也可以用 `glfw`。中文插图使用 Noto Sans CJK 或文泉驿字体，MP4 编码需要 `ffmpeg`。

本机使用 Anaconda Python 3.11 / MuJoCo 3.4.0 时，Anaconda 自带的旧 `libstdc++` 曾与系统 Mesa 的 LLVM 依赖冲突，出现 `GLIBCXX_3.4.30` 和 EGL 初始化错误。渲染管线已用以下进程级环境前缀解决该冲突；运行总入口时可使用：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/ik-mpl \
python -m scripts.showcase.build
```

这些库路径对应本机 Linux 环境，只作用于当前进程。其他机器应按各自 OpenGL 驱动安装位置设置。
