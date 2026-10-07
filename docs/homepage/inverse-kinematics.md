---
title: "机械臂逆运动学：从机构结构到连续运动"
subtitle: "一个目标位姿，可以有多种构型；一条连续轨迹，还需要稳定地选解。"
description: "在 MuJoCo 中实现四台机械臂的结构识别、闭式求解、冗余参数化和阻尼数值求解，用可复现的分支、轨迹与近奇异实验检查位姿误差、关节连续性和限位。"
heroImage: "/assets/img/projects/inverse-kinematics/portfolio_card.webp"
heroImageAlt: "四台机械臂的逆运动学结构与求解路线"
featureVideo:
  src: "/assets/videos/inverse-kinematics/pose_quad.mp4"
  poster: "/assets/img/projects/inverse-kinematics/pose_quad_first.webp"
  caption: "固定末端位姿：六轴直接切换闭式候选，iiwa 改变臂型角，Panda 按指定 q7 做 DLS 自运动。"
resultFigures:
  - src: "/assets/img/projects/inverse-kinematics/solutions_quad.webp"
    alt: "四型号各自固定末端位姿的多解对照"
    caption: "六轴的离散候选与七轴的有限冗余采样：每型号各展示两个代表构型。"
  - src: "/assets/img/projects/inverse-kinematics/portfolio_tracking.webp"
    alt: "Panda 路径对照与四台机器人的相邻关节变化"
    caption: "上图对照 Panda 的逐点 IK 与关节插值，下图检查四台机器人的相邻关节变化。"
order: 5
role: "独立完成（求解 / 实验验证 / 可视化）"
period: "2026.10"
status: "展示项目 · 可复现"
metrics:
  - "四台机械臂 · 三条求解路线"
  - "位置 / 姿态 / 限位逐点验收"
  - "固定种子实验与原始数据"
stack:
  - "Python"
  - "NumPy"
  - "MuJoCo"
  - "几何 IK"
  - "阻尼最小二乘"
  - "冗余运动学"
highlights:
  - value: "4 台"
    label: "UR5e / Lite 6 / iiwa 14 / Panda，在统一模型接口下对照"
  - value: "960 / 960"
    label: "固定 TCP 位姿的四段动画，共 960 帧通过位置、姿态与限位验收"
  - value: "< 0.1 rad"
    label: "直线与末端圆周路径的最大相邻关节变化范数，含循环衔接；未时间参数化"
  - value: "7 个"
    label: "几何初值补充实验返回的 Panda 构型，其中 2 个接近限位"
buttons:
  - label: "相关笔记：逆运动学原理"
    url: "/posts/逆运动学原理/"
    icon: "lucide:book-open"
    primary: true
  - label: "固定末端动画数据"
    url: "/assets/data/inverse-kinematics/pose_metrics.json"
    icon: "lucide:file-json"
  - label: "相关笔记：机械臂运动学"
    url: "/posts/机械臂运动学/"
    icon: "lucide:book-open"
tags:
  - "逆运动学"
  - "运动学"
  - "MuJoCo"
  - "闭式解"
math: true
---

我实现了一套机械臂逆运动学实验，覆盖 UR5e、UFACTORY Lite 6、KUKA iiwa 14 和 Franka Panda。项目从模型中的关节轴关系出发，选择闭式、臂型角参数化或几何初值与数值修正，再用正运动学检验结果。这里展示的是方法选择、误差验收和连续性处理。

**Python · NumPy · MuJoCo · 几何求解 · DLS · 冗余运动学**

## 1. 先看机构，再选求解方法

![四台机械臂的关节轴结构与求解路线](/assets/img/projects/inverse-kinematics/portfolio_structure.webp)

这四台机械臂共享位姿和雅可比接口，几何求解器则按结构分别实现。我从 MuJoCo 的关节轴线检查连续三轴的平行或汇交关系，避免仅凭外观套用公式。

这里用 **Pieper 准则**作结构筛选：对 6R 串联机械臂，**三个连续关节轴交于同一点**，或**三个连续关节轴相互平行**，都是构造闭式逆解的充分条件。它们是**充分条件，不是必要条件**；未检测到这两种结构，不能据此断言不存在其他闭式方法。

四型号对照同时包含两条路线：UR5e 展示连续三轴平行，**Lite 6 的 J4–J6 就是三轴共点的球腕**。iiwa 也有球腕，但它是 7R 冗余机构，不能把上述 6R 判据直接当作七关节的完整解法。

- **UR5e，6 轴**：J2–J4 平行，利用平面几何构造肩、肘、腕的闭式分支。
- **Lite 6，6 轴**：J4–J6 汇交于腕心，先求腕心位置，再分解腕部姿态。
- **iiwa 14，7 轴**：球肩、单肘、球腕构成 S-R-S 结构，臂型角提供剩余的一维自由度。
- **Panda，7 轴**：腕部存在偏移，这里的 S-R-S 解耦只能提供几何初值；随后用阻尼最小二乘修正完整位姿。

Panda 的腕部偏移约为 0.088 m。这个事实解释了**本项目为什么采用混合求解**；其他解析方法可见 [He 与 Liu（2021）的 Panda IK 实现](https://github.com/ffall007/franka_analytical_ik)，以 `q7` 为冗余参数构造几何解。几何常量由模型标定或测量；Panda 的 MDH 参数表仍在代码中显式定义，并与 MuJoCo 正运动学交叉验证。

## 2. 区分离散分支与连续冗余

六轴机械臂在一般非奇异目标下对应离散分支。七轴机械臂求解完整六维位姿时，在雅可比满行秩的局部通常留下**一维自运动解集**。iiwa 可以直接用臂型角描述；Panda 同样冗余，只是这里通过有限几何采样与数值修正取得候选。

### 末端位姿不变，构型怎样变化

<figure class="video-embed my-6">
  <div class="video-embed__frame">
    <video controls autoplay muted loop playsinline preload="metadata" poster="/assets/img/projects/inverse-kinematics/pose_quad_first.webp" title="固定末端位姿：离散闭式候选与连续冗余自运动">
      <source src="/assets/videos/inverse-kinematics/pose_quad.mp4" type="video/mp4" />
    </video>
  </div>
  <figcaption class="text-base-content/55 mt-2 text-center text-xs italic">每型号内部固定同一目标位置与姿态：UR5e、Lite 6 直接切换离散闭式候选；iiwa 14 随臂型角闭式求解，Panda 按指定 q7 做 DLS 数值延拓。</figcaption>
</figure>

这组动画在每一型号内部使用**同一个目标位置与姿态**，固定相机并逐帧回代验证；四台机械臂分别选择自己的目标。六轴展示离散候选之间的直接切换，七轴通过连续变化的冗余参数逐帧求解自运动，画面直接标明各自的真实解法。

每段 **240 帧、20 fps、12 秒**，UR5e、Lite 6 的八个候选各停留 **1.5 秒**。四段共 **960 帧**，全部通过 `10⁻⁵ m / 10⁻⁵ rad` 位姿容差和关节限位验收；最大位置、姿态误差分别约 **1.94 × 10⁻⁶ m**、**1.08 × 10⁻⁸ rad**。

- **UR5e：离散闭式解切换。** 利用平行轴结构求解肩、肘、腕的离散候选；停留后直接切到另一个已验收构型，构型之间不插值。
- **Lite 6：球腕闭式解切换。** 先解腕心位置，再分解腕部姿态；同样直接切换独立候选，不能把关节插值当作固定末端的连续解。
- **iiwa 14：臂型角闭式自运动。** 本段动画选用 **6°–156°** 的稳定合法区间往返，每帧闭式求解完整关节构型；这是一段展示区间，并非限位的精确边界。肘部几何圆半径约 **19.75 cm**，画面同时标出这段 **150°** 运动弧；这不承诺同一连续解支可转完整圈。
- **Panda：指定 q7 的 DLS 数值延拓。** `q7` 在约 **−1.13454 至 −0.13454 rad** 的区间往返。每次求解固定当前指定的 `q7`，仅对其余六个关节做 DLS 修正，并用上一帧热启动；求解器报告每帧 **1–5 次迭代**。

单独播放：[UR5e · 离散闭式解切换](/assets/videos/inverse-kinematics/pose_ur5e.mp4) · [Lite 6 · 球腕闭式解切换](/assets/videos/inverse-kinematics/pose_lite6.mp4) · [iiwa 14 · 臂型角闭式自运动](/assets/videos/inverse-kinematics/pose_iiwa14.mp4) · [Panda · 指定 q7 的 DLS 数值延拓](/assets/videos/inverse-kinematics/pose_panda.mp4)。

UR5e、Lite 6 的跳切只比较独立有效的构型，**不表示可执行的连续换支轨迹**。iiwa、Panda 的动画则检查相邻帧构型变化，逐帧保持末端目标位姿。iiwa、Panda 的最大相邻关节变化范数分别约 **0.0381、0.1094 rad**，最小限位裕度分别约 **0.5717、0.0869 rad**，并单独检查了循环接缝。这些自运动区间属于本例目标下的局部解族，未覆盖全部冗余构型。动态实验记录见 [pose_metrics.json](/assets/data/inverse-kinematics/pose_metrics.json)。

### 每个臂型角有解，为什么仍不能连续转一圈

这里讨论的是 **iiwa 14 的当前 TCP 目标与当前模型关节限位**。以 **0.25°** 步长扫描 0°–360°，含两端共 **1441 个采样角**，每个采样角都有 **2–8 个合法 IK 候选**。这只说明逐点有解；连续转一圈还需要这些候选能够沿臂型角连接起来，关节角始终连续并满足限位。

![当前 iiwa 目标的臂型角候选连接与肘圆水平投影](/assets/img/projects/inverse-kinematics/swivel_feasibility.webp)

我保留每个采样角上返回的所有合法候选，在相邻角之间按实际关节坐标的变化范数建立连接图，分别使用 **0.15 rad、0.30 rad** 阈值，不把越限的角度跳变按 2π 折回。本例两种阈值下都没有贯穿 0°–360° 的候选连接链；其他角度上的解可能属于另一支，不能仅凭“那里有解”就把两支拼成连续运动。

本例还检查肘部圆的水平投影：投影围住肩轴且不穿过肩轴，采样点到肩轴的最小水平距离约 **8.81 mm**。因此沿同一连续肩分支走完一圈时，J1 需要累计旋转 **360°**，而模型的 J1 限位约为 **±170°，总行程 340°**。这是**当前目标的几何关系与限位共同形成的障碍**；换一个目标时，投影关系可能改变，结论需要重新检查。

主动画采用的 **6°–156°** 是稳定展示段，**不是硬边界**。继续沿当前解支扫描，本次的**最后合法采样点为 172.75°**；到下一采样角 **173°** 时，J7 超出下限约 **0.00272 rad**。从末个合法构型直接切到该角最近的合法替代候选，关节向量变化范数约为 **4.43 rad**，因此这不是可直接拼接的连续下一帧。

下面的解释动画用 **9 秒** 沿当前解支延拓至末个合法采样点，再停留 **3 秒**；每帧都重新求解 IK。结束后的重播是说明性重启，不表示首尾连续的回程运动。

<figure class="video-embed my-6">
  <div class="video-embed__frame">
    <video controls muted loop playsinline preload="metadata" poster="/assets/img/projects/inverse-kinematics/swivel_iiwa14_first.webp" title="当前 iiwa 目标：逐点有解与连续整圈的区别">
      <source src="/assets/videos/inverse-kinematics/swivel_iiwa14.mp4" type="video/mp4" />
    </video>
  </div>
  <figcaption class="text-base-content/55 mt-2 text-center text-xs italic">末端目标固定，沿当前解支接近限位后停住；解释结束后重新播放，不表示首尾连续的运动轨迹。</figcaption>
</figure>

**局部的一维冗余自由度，不等于全局的 360° 周期自由度。** 邻接图提供有限采样下的连接证据，几何解释限定于本例目标；二者都不能推广为所有机器人或所有目标均无法转整圈。此诊断与主动画、末端路径的验收统计分别记录。

诊断数据见 [swivel_metrics.json](/assets/data/inverse-kinematics/swivel_metrics.json)。

<details>
<summary>补充：四型号静态八构型图集与采样数据</summary>

![四型号各自固定末端位姿的静态多解总览](/assets/img/projects/inverse-kinematics/solutions_quad.webp)

静态图集各展示八个代表构型：UR5e、Lite 6 保留本目标下返回的八个离散候选；iiwa 在 16 个臂型角网格得到的 102 个合法构型中按 45° 间隔选八个；Panda 沿局部 `q7` 解族保留 50 个内部候选，再按几何差异选八个，所选样本的最小限位裕度为 **0.0567 rad**。

**UR5e · 八个离散闭式候选**

![UR5e：同一目标位姿下的八个离散闭式候选](/assets/img/projects/inverse-kinematics/solutions_ur5e.webp)

**Lite 6 · 八个离散闭式候选**

![Lite 6：同一目标位姿下的八个离散闭式候选](/assets/img/projects/inverse-kinematics/solutions_lite6.webp)

**iiwa 14 · 按臂型角分散选择的八个冗余样本**

![iiwa 14：同一目标位姿下的按臂型角分散选择的八个冗余样本](/assets/img/projects/inverse-kinematics/solutions_iiwa14.webp)

**Panda · 通过 q7 扫描选择的八个冗余样本**

![Panda：同一目标位姿下的通过 q7 扫描选择的八个冗余样本](/assets/img/projects/inverse-kinematics/solutions_panda.webp)

图集数据见 [solution_metrics.json](/assets/data/inverse-kinematics/solution_metrics.json)。七轴的八个展示样本不构成解集枚举，也不改变下面几何初值实验的返回数量。Panda 扫描在数值未收敛处停止，这不证明该处是解集边界。

</details>

### 几何初值与限位验收的补充实验

![UR5e 同一目标下的 8 个合法闭式候选](/assets/img/projects/inverse-kinematics/portfolio_branches.webp)

下面这组补充实验记录在 `portfolio_metrics.json`，沿用各型号的同一目标，重新计算并检查每个返回构型：UR5e 返回 **8 个**合法候选，Lite 6 返回 **8 个**，iiwa 返回 **16 个采样构型**，Panda 返回 **7 个采样构型**。所有返回构型通过本次的位置、姿态和限位验收；这是这些目标下的返回数量，不是解集完备性的证明。

Panda 的接口上限 `max_solutions=8` 是工程参数，不能解释为“理论上最多八个解”。本例 Panda 候选中有 **2 个**的最小限位裕度不足 `10⁻³ rad`，整体最小裕度为 **0 rad**：即使末端误差通过验收，选解时仍需要关心构型是否贴近限位。程序因此同时记录位置误差、姿态误差和最小限位裕度。

![Panda 同一个候选的真实几何初值与 DLS 修正后构型](/assets/img/projects/inverse-kinematics/portfolio_polish.webp)

这张对照使用同一次求解中真实保留的几何初值。经过 **5 次迭代**，位置误差从约 **175.8 mm** 降到 **9.31 × 10⁻⁸ m**，修正后构型的最小限位裕度约 **44.4°**。几何步骤提供有结构的初值，最终精度通过完整位姿回代验收；不同候选需要的修正次数各不相同。

## 3. 从直线到圆周，逐点跟踪末端路径

这两组动画使用明亮背景，直接画出目标路径和末端实际走过的轨迹。四格依次对应 UR5e、Lite 6、iiwa 14 与 Panda，各自在可达区域内跟踪一条路径；**末端姿态保持固定，位置沿路径逐点移动**。

UR5e、iiwa 14 和 Panda 的直线段长 **1.12 m**、圆周半径 **40 cm**（直径 **80 cm**）；Lite 6 的直线段长 **68 cm**、圆周半径 **26 cm**（直径 **52 cm**）。每段视频使用 **240 帧、20 fps**，循环时长 **12 秒**；直线沿线段往返，圆周沿整圈前进。

### 直线：沿目标线段逐点跟随

<figure class="video-embed my-6">
  <div class="video-embed__frame">
    <video controls muted loop playsinline preload="metadata" poster="/assets/img/projects/inverse-kinematics/path_line_quad_first.webp" title="四台机械臂固定姿态直线逐点跟踪">
      <source src="/assets/videos/inverse-kinematics/path_line_quad.mp4" type="video/mp4" />
    </video>
  </div>
  <figcaption class="text-base-content/55 mt-2 text-center text-xs italic">四台机械臂的固定姿态直线跟踪。完整目标线段始终可见，移动的目标标记与末端轨迹共同显示逐点求解的过程。</figcaption>
</figure>

### 圆周：末端绕完整一圈

这次移动的是**末端位置**：它沿笛卡尔空间的圆周走完一圈，姿态保持固定。第 2 节的 iiwa、Panda 自运动保持末端位姿不变，改变的是手臂构型，两者对应不同任务。

<figure class="video-embed my-6">
  <div class="video-embed__frame">
    <video controls muted loop playsinline preload="metadata" poster="/assets/img/projects/inverse-kinematics/path_circle_quad_first.webp" title="四台机械臂固定姿态末端圆周运动">
      <source src="/assets/videos/inverse-kinematics/path_circle_quad.mp4" type="video/mp4" />
    </video>
  </div>
  <figcaption class="text-base-content/55 mt-2 text-center text-xs italic">四台机械臂的固定姿态末端圆周跟踪。画面中的圆是末端目标路径；末端逐点经过整圈目标位置。</figcaption>
</figure>

<details>
<summary>按机器人查看单独视频</summary>

- **UR5e**：[直线跟踪](/assets/videos/inverse-kinematics/path_line_ur5e.mp4) · [末端圆周](/assets/videos/inverse-kinematics/path_circle_ur5e.mp4)
- **Lite 6**：[直线跟踪](/assets/videos/inverse-kinematics/path_line_lite6.mp4) · [末端圆周](/assets/videos/inverse-kinematics/path_circle_lite6.mp4)
- **iiwa 14**：[直线跟踪](/assets/videos/inverse-kinematics/path_line_iiwa14.mp4) · [末端圆周](/assets/videos/inverse-kinematics/path_circle_iiwa14.mp4)
- **Panda**：[直线跟踪](/assets/videos/inverse-kinematics/path_line_panda.mp4) · [末端圆周](/assets/videos/inverse-kinematics/path_circle_panda.mp4)

</details>

四台机器人各两条路径，共 **1920 帧**，全部通过 `10⁻⁵ m / 10⁻⁵ rad` 位姿容差和关节限位检查。各片相邻帧与循环衔接处的关节向量变化范数均小于 **0.1 rad**。轨迹与验收记录见 [path_metrics.json](/assets/data/inverse-kinematics/path_metrics.json)。播放帧率仅用于展示，尚未进行碰撞检测、时间参数化或动力学验证。

### 误差与相邻关节变化的对照

下面保留原先的位置／姿态插值短路径实验；它与上述固定姿态视频分开记录，图中数字来自 [portfolio_metrics.json](/assets/data/inverse-kinematics/portfolio_metrics.json)。

![上图为 Panda 路径对照，下图为四台机器人相邻关节变化](/assets/img/projects/inverse-kinematics/portfolio_tracking.webp)

关节空间的直线通常不会映射成笛卡尔空间的直线。我在线性位置插值和旋转插值产生的目标位姿上逐点求解：UR5e、Lite 6 和 iiwa 从候选中选取最接近上一点的关节构型，Panda 使用上一解热启动 DLS。

每条路径均匀采样 **48 点**。本次四条路径的所有采样点都通过 `10⁻⁵ m` 位置容差、`10⁻⁵ rad` 姿态容差和关节限位检查。图中上方显示 Panda 的路径对照：绿色是逐点 IK，橙色使用相同端点，只在关节角之间线性插值。四台机器人的关节插值路径到目标直线的最大偏离依次为 **4.03、3.60、13.94、6.94 mm**（UR5e、Lite 6、iiwa、Panda）。

图中下方检查四台机器人的相邻关节向量变化。本例四条路径的最大相邻变化范数为 **0.0112、0.0127、0.0320、0.0115 rad**，可以用来识别突然换支。路径参数尚未分配实际时间，因此这些值不能直接当作关节速度或加速度，也不保证采样点之间的全部约束成立。本次 Panda 轨迹的最小限位裕度为 **0.494 rad**。

## 4. 接近奇异时，把误差和步长一起看

![近伸直目标下三种数值更新的误差与步长](/assets/img/projects/inverse-kinematics/portfolio_singularity.webp)

这组实验固定在 iiwa 的一个近伸直目标：腕心距离约为两段臂长之和的 **99.67%**，肘部圆半径收缩到约 **33.3 mm**。它用来观察条件变差时的迭代行为，不代表所有机器人的奇异构型分类。

阻尼最小二乘使用

$$
\Delta q = J^T\left(JJ^T + \lambda^2 I\right)^{-1}e.
$$

阻尼抑制小奇异值方向上的增益，代价是收敛行为随阻尼、初值和容差变化。生产求解器还配合步长限幅、关节限位与误差下降验收，自适应调整阻尼。

图中的曲线是独立的 **80 步教学实验**：DLS 固定 `λ=0.05`，未启用生产求解器的步长限幅和误差下降验收。两套结果分开记录。在同一初值和目标下，实际 `solve_numerical` 的伪逆、DLS 分别在 **7、8 次迭代**内通过 `10⁻⁶ m / 10⁻⁶ rad` 验收；转置法在 80 次迭代后的位置误差仍约 **8.37 mm**。这组结果说明阻尼的作用和代价，也保留了伪逆在本例成功收敛的事实。

位置误差用米、姿态误差用弧度分别报告。雅可比的奇异值依赖平移与旋转行的尺度，避免把混合残差或奇异值当作具有统一物理单位的精度指标。

## 5. 用验证和复现界定结果

每张图都来自 MuJoCo 模型与本次求解记录。返回解的残差由正运动学重新计算，轨迹中的每个构型也单独检查关节限位。这里报告的是模型中的数值残差，实机精度还取决于标定、传感和控制。测试覆盖 Panda 的 MDH／MuJoCo 正运动学互校、雅可比有限差分、闭式回代、限位和不可达目标等。

冗余次级任务支持限位规避、姿态偏好和可操作度。阻尼下的 `N = I − J#J` 是近似零空间投影，不能保证次级运动严格不影响主任务；实现中对次级步限幅，并在接受前比较主任务误差。

实验数据保存在 [portfolio_metrics.json](/assets/data/inverse-kinematics/portfolio_metrics.json)：包含目标位姿、关节向量、采样设置、逐点误差、限位裕度、依赖版本和源码哈希。本次目标通过固定随机种子挑选，属于 **四个代表性目标和四条可达短路径**，没有据此推导工作空间成功率或求解速度结论。

项目目前完成运动学求解与仿真验证。碰撞检测、障碍物约束、时间参数化、动力学以及实机控制属于后续工作。

在项目仓库根目录复现（Python 3.10+）：

```bash
python -m pip install -e '.[showcase,test]' -c requirements-reproduce.txt
export MUJOCO_MENAGERIE=/path/to/mujoco_menagerie
git clone https://github.com/google-deepmind/mujoco_menagerie.git "$MUJOCO_MENAGERIE"
git -C "$MUJOCO_MENAGERIE" checkout bf756430b615819654b640f321c71ba5c3ebeef8
export MUJOCO_GL=egl

python -m scripts.showcase.build
python -m pytest -q
```

模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。已有模型克隆只需切换到上述 commit。Linux 无窗口环境使用 EGL；完整视频构建需要 `ffmpeg` 与中文字体。总入口将原始渲染写入本地 `build/showcase/`，再更新页面实际引用的发布资源。

只打包当前发布素材时，无需重新渲染：

```bash
python -m scripts.showcase.export_homepage
```

输出为本地 `build/ik-homepage.zip`，不纳入版本控制，也不会自动修改或部署个人站点。构建产出的 JSON 合计记录各阶段的源码与输入溯源，`portfolio_metrics.json` 另记录模型 XML 哈希和依赖版本。数值复现不要求时间戳、耗时、绝对路径和渲染文件字节相同；重跑后应核对正文摘要。

延伸阅读：[逆运动学原理](/posts/逆运动学原理/) · [机械臂运动学](/posts/机械臂运动学/) · [本次实验数据](/assets/data/inverse-kinematics/portfolio_metrics.json)
