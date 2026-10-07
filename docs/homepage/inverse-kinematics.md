---
title: "机械臂逆运动学：从机构结构到连续运动"
subtitle: "让末端沿轨迹移动，也让手臂在末端不动时改变构型。"
description: "我为 UR5e、Lite 6、iiwa 14 和 Panda 实现了逆运动学求解，用动画展示离散多解、冗余自运动与末端路径。"
heroImage: "/assets/img/projects/inverse-kinematics/portfolio_card.webp"
heroImageAlt: "四台机械臂的逆运动学结构与求解路线"
previewVideos:
  - src: "/assets/videos/inverse-kinematics/path_line_quad.mp4"
    poster: "/assets/img/projects/inverse-kinematics/path_line_quad_first.webp"
    aspectRatio: "12 / 11"
    caption: "直线运动｜四种机械臂沿同一类笛卡尔直线路径运动。"
    title: "直线运动"
  - src: "/assets/videos/inverse-kinematics/path_circle_quad.mp4"
    poster: "/assets/img/projects/inverse-kinematics/path_circle_quad_first.webp"
    aspectRatio: "12 / 11"
    caption: "圆周运动｜四种机械臂沿末端圆轨迹运动。"
    title: "圆周运动"
featureVideo:
  src: "/assets/videos/inverse-kinematics/pose_quad.mp4"
  poster: "/assets/img/projects/inverse-kinematics/pose_quad_first.webp"
  caption: "末端不动，手臂还能怎样变化？四台机械臂的多解与自运动。"
  aspectRatio: "12 / 11"
resultFigures:
  - src: "/assets/img/projects/inverse-kinematics/solutions_quad.webp"
    alt: "四型号各自固定末端位姿的多解对照"
    caption: "六轴的离散候选与七轴的有限冗余采样：每型号各展示两个代表构型。"
  - src: "/assets/img/projects/inverse-kinematics/portfolio_tracking.webp"
    alt: "Panda 路径对照与四台机器人的相邻关节变化"
    caption: "上图对照 Panda 的逐点 IK 与关节插值，下图检查四台机器人的相邻关节变化。"
order: 5
role: "独立开发 · 求解器与可视化"
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
    label: "从平行轴、球腕到七轴冗余"
  - value: "960 / 960"
    label: "固定末端动画的位置、姿态与限位检查"
  - value: "< 0.1 rad"
    label: "直线与圆周的相邻关节变化范数，含接缝"
buttons:
  - label: "查看源码"
    url: "https://github.com/JiangShangJiu/ik"
    icon: "lucide:github"
    external: true
    primary: true
  - label: "复现与素材发布"
    url: "https://github.com/JiangShangJiu/ik/blob/main/docs/REPRODUCING.md"
    icon: "lucide:terminal"
    external: true
  - label: "逆运动学原理"
    url: "/posts/逆运动学原理/"
    icon: "lucide:book-open"
  - label: "机械臂运动学"
    url: "/posts/机械臂运动学/"
    icon: "lucide:book-open"
tags:
  - "逆运动学"
  - "运动学"
  - "MuJoCo"
  - "闭式解"
math: true
---

我给 UR5e、Lite 6、iiwa 14 和 Panda 写了一套逆运动学求解器，想弄清楚同一个末端位姿能对应哪些构型，以及怎样沿着一条路径稳定地选解。上面的动画把末端的位置和朝向固定下来，展示手臂还能怎样改变构型。

六轴机械臂的不同解通常是离散分支，所以我让 UR5e 和 Lite 6 停在一个解上，再直接切到另一个解，没有在两个构型之间插值。七轴在一般非奇异位姿下还留有一维冗余，我沿着这维自由度逐帧求解，让 iiwa 和 Panda 在末端不动时连续活动。

<details>
<summary>分型号看固定末端动画与实验记录</summary>

**UR5e · 平行轴闭式解**

八个独立候选各停留 1.5 秒，直接切换。这是在比较有效构型，不表示一条可以执行的换支轨迹。

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="UR5e 固定末端的离散闭式解">
      <source src="/assets/videos/inverse-kinematics/pose_ur5e.mp4" type="video/mp4" />
    </video>
  </div>
</figure>

**Lite 6 · 球腕闭式解**

同样直接切换八个候选。关节之间的直线插值通常无法保持原来的末端位姿。

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="Lite 6 固定末端的球腕闭式解">
      <source src="/assets/videos/inverse-kinematics/pose_lite6.mp4" type="video/mp4" />
    </video>
  </div>
</figure>

**iiwa 14 · 臂型角闭式自运动**

我选取 ψ = 6°–156° 这段稳定区间往返，每帧重新求解。画出的肘部圆半径约 19.75 cm，运动弧为 150°；这段区间不是限位的精确边界。

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="iiwa 臂型角闭式自运动">
      <source src="/assets/videos/inverse-kinematics/pose_iiwa14.mp4" type="video/mp4" />
    </video>
  </div>
</figure>

**Panda · 指定 q7 的数值延拓**

q7 在约 −1.13454 至 −0.13454 rad 之间往返。每一帧固定当前指定的 q7，用上一帧热启动，只对其余六轴做 DLS；求解器报告每帧 1–5 次迭代。

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="Panda 指定 q7 的 DLS 自运动">
      <source src="/assets/videos/inverse-kinematics/pose_panda.mp4" type="video/mp4" />
    </video>
  </div>
</figure>

四段各 240 帧、20 fps、12 秒，共 960 帧，位置、姿态与关节限位全部通过本次检查。位姿容差为 10⁻⁵ m / 10⁻⁵ rad，最大误差约为 1.94 × 10⁻⁶ m / 1.08 × 10⁻⁸ rad。iiwa、Panda 的最大相邻关节变化范数分别约为 0.0381、0.1094 rad，最小限位裕度分别约为 0.5717、0.0869 rad；连续自运动还单独检查了循环接缝。这里展示的只是各自目标下的一段局部解族，逐帧记录见 [pose_metrics.json](/assets/data/inverse-kinematics/pose_metrics.json)。

</details>

## 1. 我为什么给四台机械臂选了不同的解法

给定目标位置 $p_d$ 和姿态 $R_d$，逆运动学要找到同时满足二者与关节限位的关节角。我采用模型指定的末端／法兰坐标系作为 TCP，把四台机械臂接到同一组位姿、雅可比接口上，再从 MuJoCo 中读取关节轴，检查连续三轴是否平行或共点。

![四台机械臂的关节轴结构与求解路线](/assets/img/projects/inverse-kinematics/portfolio_structure.webp)

对 6R 串联机构，**连续三轴共点**或**连续三轴相互平行**都是 Pieper 准则给出的闭式求解充分条件，条件范围可对照 [IK-Geo 原始论文](https://arxiv.org/abs/2211.05737)。UR5e 的 J2–J4 平行，Lite 6 的 J4–J6 共点，正好覆盖这两条路线。它们是充分条件，没检测到这些结构不能据此断言不存在闭式解；iiwa 虽然也有球腕，却是 7R，还要处理冗余参数。

### UR5e：先减成一个平面两杆问题

我先从目标位姿去掉工具偏移，用固定侧向偏移求肩角；再由目标末轴方向确定平面末段的方向。扣除这段之后，剩余位置问题可以交给平面 2R，用余弦定理求肘上、肘下两种构型，再恢复腕部旋转。

设两杆长度为 $l_1,l_2$，扣除偏移后的平面目标为 $d$，核心计算是：

$$
c_\delta=\frac{\|d\|^2-l_1^2-l_2^2}{2l_1l_2},\qquad
\delta=\pm\arccos c_\delta.
$$

$\delta$ 是几何夹角，转成模型关节角时还需处理零位偏角和轴方向。UR 的腕部没有三轴共点，所以这里还用到了目标姿态，不能直接套用球腕的位置／姿态解耦。具体分支处理见 [Ur5eSolver](https://github.com/JiangShangJiu/ik/blob/main/ik/closedform.py)。

### Lite 6：腕心定位之后，再解腕部姿态

球腕的三个关节都绕同一个腕心 $W$ 转，转动它们不会改变腕心的位置。我先根据目标求腕心：

$$
W=p_d-R_do_w.
$$

$o_w$ 是 TCP 坐标系中从腕心到 TCP 的固定偏移，由模型测量；本例 Lite 6 中为零。前三轴把腕心送到目标，剩下的旋转在实测腕轴基底中做 Z–Y–Z 分解。肩、肘、腕的选择组合成不同候选，再逐个做限位和正运动学检查。实现见 [Lite6Solver](https://github.com/JiangShangJiu/ik/blob/main/ik/closedform.py)。

### iiwa 14：给肘的位置加一个参数

固定肩心 $S$、腕心 $W$ 和两段臂长后，肘点必须同时落在两个球面上。一般非退化情况下，交线是一个圆；我用臂型角 $\psi$ 指定肘点在圆上的位置：

$$
E(\psi)=C+\rho\left(h\cos\psi+v\sin\psi\right).
$$

$C,\rho$ 是圆心和半径，$h,v$ 是圆面内的正交单位向量。指定肘点后，我用上臂方向和肘轴构造肩部旋转，再提取肩角、肘角与腕角。**臂型角是整条手臂的构型参数，并不是某一个关节角。** 同一个角度也可能对应多个分支，仍要检查限位，见 [SrsSolver](https://github.com/JiangShangJiu/ik/blob/main/ik/srs.py)。

<details>
<summary>肘部圆的圆心和半径怎样算</summary>

两段臂长为 $a,b$，令 $L=\|W-S\|$、$\hat u=(W-S)/L$，则：

$$
x=\frac{a^2+L^2-b^2}{2L},\qquad
C=S+x\hat u,\qquad \rho=\sqrt{a^2-x^2}.
$$

这描述的是肘点的几何可达圆，整圈是否有连续合法的关节解，还取决于分支和关节限位。

</details>

### Panda：几何给初值，完整模型给最终答案

Panda 有约 0.088 m 的腕部偏移，这里的理想 S-R-S 几何不能精确解耦。我扫描腕心偏移方向和臂型角，构造有几何结构的初值，再用完整位姿的 DLS 修正。几何种子还不是有效逆解，最终要回到模型里检查位置和姿态。

开头的 Panda 自运动用的是另一条明确的约束：每帧指定 $q_7=s$，删除雅可比第七列，只求其余六轴，并用上一帧作为初值。这样可以沿本例的一段局部解族延拓，固定关节在单次求解中始终固定。几何种子与延拓分别见 [analytic.py](https://github.com/JiangShangJiu/ik/blob/main/ik/analytic.py)、[pose_data.py](https://github.com/JiangShangJiu/ik/blob/main/scripts/showcase/pose_data.py)。

这是我的实现选择，不意味着 Panda 没有解析逆解。[He 与 Liu（2021）的实现](https://github.com/ffall007/franka_analytical_ik) 就使用 q7 作为冗余参数。项目中的 Panda MDH 参数仍在代码中显式定义，并与 MuJoCo 正运动学互校。

<details>
<summary>四台机械臂的静态多解图集</summary>

![四型号各自固定末端位姿的多解总览](/assets/img/projects/inverse-kinematics/solutions_quad.webp)

六轴图集保留本目标下返回的八个离散候选。iiwa 从 16 个臂型角网格得到的 102 个合法构型中按 45° 间隔选八个；Panda 沿局部 q7 解族保留 50 个内部候选，再按几何差异选八个，所选构型的最小限位裕度为 0.0567 rad。七轴的八个构型只是代表样本，不是全部解。

**UR5e**

![UR5e 同一目标下的八个离散闭式候选](/assets/img/projects/inverse-kinematics/solutions_ur5e.webp)

**Lite 6**

![Lite 6 同一目标下的八个离散闭式候选](/assets/img/projects/inverse-kinematics/solutions_lite6.webp)

**iiwa 14**

![iiwa 同一目标下按臂型角选择的八个冗余样本](/assets/img/projects/inverse-kinematics/solutions_iiwa14.webp)

**Panda**

![Panda 同一目标下沿 q7 解族选择的八个冗余样本](/assets/img/projects/inverse-kinematics/solutions_panda.webp)

Panda 扫描在数值未收敛处停止，不能据此认定解集边界。候选与筛选记录见 [solution_metrics.json](/assets/data/inverse-kinematics/solution_metrics.json)。

</details>

## 2. 让末端沿直线和圆周走起来

固定末端让我看清了多解，下一步是让末端移动。我先在笛卡尔空间画出目标，再对每一点求关节角，检查实际末端能否沿线走过去，同时保持朝向不变。

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 12 / 11; padding-bottom: 0;">
    <video controls muted loop playsinline preload="metadata" poster="/assets/img/projects/inverse-kinematics/path_line_quad_first.webp" title="四台机械臂的固定姿态直线跟踪">
      <source src="/assets/videos/inverse-kinematics/path_line_quad.mp4" type="video/mp4" />
    </video>
  </div>
  <figcaption class="text-base-content/55 mt-2 text-center text-xs italic">沿完整目标线段往返。移动标记是本帧目标，末端轨迹显示逐点求解的结果。</figcaption>
</figure>

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 12 / 11; padding-bottom: 0;">
    <video controls muted loop playsinline preload="metadata" poster="/assets/img/projects/inverse-kinematics/path_circle_quad_first.webp" title="四台机械臂的固定姿态末端圆周运动">
      <source src="/assets/videos/inverse-kinematics/path_circle_quad.mp4" type="video/mp4" />
    </video>
  </div>
  <figcaption class="text-base-content/55 mt-2 text-center text-xs italic">这次绕圆的是末端位置，朝向仍然固定；它和开头末端不动的自运动是两种任务。</figcaption>
</figure>

UR5e、iiwa 和 Panda 的直线长 1.12 m、圆半径 40 cm；Lite 6 分别为 68 cm、26 cm。六轴每一帧选取最靠近上一解的闭式候选；iiwa 固定 $\psi=-0.5\,\mathrm{rad}$ 再选分支；Panda 固定的是 **$q_3=0$**，对其余六轴做 DLS。这里固定第三关节，开头的固定末端动画才扫描第七关节。

最近邻选解使用实际的限位内关节坐标：

$$
q_k=\operatorname*{arg\,min}_{q\in\mathcal C_k}\|q-q_{k-1}\|_2.
$$

$\mathcal C_k$ 是当前帧返回的合法候选集。我还检查相邻帧、视频首尾接缝，以及从末帧重新求解首目标后是否回到起始构型；不能通过模 $2\pi$ 隐藏跳变。这仍是局部选解策略，不能保证任意目标路径都有连续解。实现见 [path_data.py](https://github.com/JiangShangJiu/ik/blob/main/scripts/showcase/path_data.py)。

<details>
<summary>分型号看直线与圆周视频</summary>

**UR5e · 直线**

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="UR5e 固定姿态直线跟踪"><source src="/assets/videos/inverse-kinematics/path_line_ur5e.mp4" type="video/mp4" /></video>
  </div>
</figure>

**UR5e · 末端圆周**

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="UR5e 固定姿态末端圆周"><source src="/assets/videos/inverse-kinematics/path_circle_ur5e.mp4" type="video/mp4" /></video>
  </div>
</figure>

**Lite 6 · 直线**

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="Lite 6 固定姿态直线跟踪"><source src="/assets/videos/inverse-kinematics/path_line_lite6.mp4" type="video/mp4" /></video>
  </div>
</figure>

**Lite 6 · 末端圆周**

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="Lite 6 固定姿态末端圆周"><source src="/assets/videos/inverse-kinematics/path_circle_lite6.mp4" type="video/mp4" /></video>
  </div>
</figure>

**iiwa 14 · 直线**

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="iiwa 固定姿态直线跟踪"><source src="/assets/videos/inverse-kinematics/path_line_iiwa14.mp4" type="video/mp4" /></video>
  </div>
</figure>

**iiwa 14 · 末端圆周**

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="iiwa 固定姿态末端圆周"><source src="/assets/videos/inverse-kinematics/path_circle_iiwa14.mp4" type="video/mp4" /></video>
  </div>
</figure>

**Panda · 直线**

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="Panda 固定 q3 的直线跟踪"><source src="/assets/videos/inverse-kinematics/path_line_panda.mp4" type="video/mp4" /></video>
  </div>
</figure>

**Panda · 末端圆周**

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted playsinline preload="none" title="Panda 固定 q3 的末端圆周"><source src="/assets/videos/inverse-kinematics/path_circle_panda.mp4" type="video/mp4" /></video>
  </div>
</figure>

</details>

<details>
<summary>路径采样、误差与关节插值对照</summary>

直线目标为 $p(s)=(1-s)p_0+sp_1$，视频以余弦进度往返；圆周目标为 $p(\theta)=c+r(h\cos\theta+v\sin\theta)$，沿整圈前进，两者的目标旋转都保持不变。八条路径各 240 帧、20 fps、12 秒，共 1920 帧，全部通过 10⁻⁵ m / 10⁻⁵ rad 位姿容差与限位检查。相邻帧及循环接缝的关节变化范数均小于 0.1 rad，详见 [path_metrics.json](/assets/data/inverse-kinematics/path_metrics.json)。

![Panda 的路径对照与四台机械臂的相邻关节变化](/assets/img/projects/inverse-kinematics/portfolio_tracking.webp)

这张图来自另一组各 48 点的位置／姿态插值短路径。上图的绿色曲线是 Panda 逐点 IK，橙色曲线只在相同端点的关节角之间插值：关节空间里的直线不会自然变成末端的直线。四台的关节插值路径到目标直线的最大偏离依次为 4.03、3.60、13.94、6.94 mm（UR5e、Lite 6、iiwa、Panda）。

下图记录四台的相邻关节变化，最大范数依次为 0.0112、0.0127、0.0320、0.0115 rad；Panda 短路径的最小限位裕度为 0.494 rad。这些数值属于 [portfolio_metrics.json](/assets/data/inverse-kinematics/portfolio_metrics.json) 的独立实验。路径没有分配真实时间，关节步长不能直接当作速度或加速度，也不保证采样点之间的所有约束。

</details>

## 3. 肘部能绕圆，为什么这一例不能转满一圈

做 iiwa 自运动时，我发现一个容易混淆的地方：每个臂型角都有解，并不意味着这些解能连成一圈。开头选用的 6°–156° 只是稳定展示段，于是我沿着当前解支继续求，看看它在哪里碰到限位。

<figure class="video-embed my-6">
  <div class="video-embed__frame" style="aspect-ratio: 8 / 5; padding-bottom: 0;">
    <video controls muted loop playsinline preload="metadata" poster="/assets/img/projects/inverse-kinematics/swivel_iiwa14_first.webp" title="iiwa 当前解支延拓至限位附近">
      <source src="/assets/videos/inverse-kinematics/swivel_iiwa14.mp4" type="video/mp4" />
    </video>
  </div>
  <figcaption class="text-base-content/55 mt-2 text-center text-xs italic">前 9 秒继续求解，后 3 秒停在最后合法采样构型。重播会跳切回起点，不表示连续回程。</figcaption>
</figure>

原因要同时看几何和限位。本目标的肘圆水平投影围住肩轴，沿同一个连续肩分支走完一圈时，J1 需要累计转过 360°，而模型只允许约 ±170°、总共 340° 的行程。我还把相邻臂型角的合法候选连成图，检查是否存在绕过当前分支限制的连接链；本次扫描中没有找到。

![当前 iiwa 目标的候选连接与肘圆水平投影](/assets/img/projects/inverse-kinematics/swivel_feasibility.webp)

这个结论只针对**当前目标、模型限位和扫描设置**。局部的一维冗余不等于全局 360° 的周期自由度，换一个目标，肘圆与肩轴的关系也可能变化。

<details>
<summary>扫描设置、最后合法采样点与替代解</summary>

扫描步长为 0.25°，0°–360° 含两端共 1441 个角度，每角有 2–8 个合法候选，总计 9414 条合法采样记录，其中包含重复几何端点，不能称为 9414 个互异解。

连接图保留每角返回的所有合法候选，分别以 0.15 / 0.30 rad 的实际关节距离建立相邻连接，两种阈值都未找到贯穿整圈的连接链。肘圆投影围住且不穿肩轴，采样点到肩轴的最小水平距离约为 8.81 mm；这为本目标提供了 J1 绕转的几何解释，有限采样本身不是普遍不可能定理。

沿视频中的解支，本次最后合法采样点为 172.75°；下一角 173° 的同支候选超出 J7 下限约 0.00272 rad。该角最近的合法替代候选与末个合法构型相差约 4.43 rad，无法直接拼成连续下一帧；这不是全域换支最小跳幅。每一帧都重新求解，说明性重播不属于连续循环验收，见 [swivel_metrics.json](/assets/data/inverse-kinematics/swivel_metrics.json)。

</details>

## 4. 几何初值之外，我还检查了数值修正

我保留了 Panda 同一次求解中的初值和最终构型，想看清楚几何步骤到底帮了什么忙。下面这例初值的位置误差约为 175.8 mm，经过 5 次迭代降到约 9.31 × 10⁻⁸ m；最终构型距离关节限位还留有约 44.4° 的最小裕度。

![Panda 同一候选的几何初值与 DLS 修正后构型](/assets/img/projects/inverse-kinematics/portfolio_polish.webp)

数值求解使用世界坐标系下的位置差和相对旋转向量：

$$
e(q)=\begin{bmatrix}p_d-p(q)\\ \operatorname{Log}\!\left(R_dR(q)^T\right)^\vee\end{bmatrix},\qquad
\Delta q=J^T\left(JJ^T+\lambda^2I\right)^{-1}e.
$$

$J=[J_v;J_\omega]$，位置与姿态分别按米、弧度验收。固定某个关节时，我删除对应雅可比列，只更新其余关节。DLS 的阻尼能压低小奇异值方向的增益，实际求解还会限幅、检查关节限位，并只接受误差平方下降的候选步；接受后减小阻尼，拒绝后增大阻尼重试。

<details>
<summary>闭式候选、几何种子和近奇异实验的完整记录</summary>

![UR5e 同一目标下的八个合法闭式候选](/assets/img/projects/inverse-kinematics/portfolio_branches.webp)

这里是几何初值补充实验，不同于前面的精选八构型图集。四台分别返回 UR5e 8 个、Lite 6 8 个、iiwa 16 个采样构型、Panda 7 个采样构型。Panda 的 `max_solutions=8` 是工程预算，不是理论上限；返回的七个中，有两个的最小限位裕度不足 10⁻³ rad，整体最小值为 0 rad。位姿算对了，还需要关心解是否贴近限位。

![近伸直目标下三种数值更新的误差与步长](/assets/img/projects/inverse-kinematics/portfolio_singularity.webp)

我另选了 iiwa 的一个近伸直目标：腕心距离为两段臂长之和的约 99.67%，肘圆半径约为 33.3 mm。伪逆对保留的非零奇异值 $\sigma$ 的增益为 $1/\sigma$，DLS 将它改为 $\sigma/(\sigma^2+\lambda^2)$。阻尼抑制过大的更新，也会改变收敛速度，不能保证任意初值都成功。

图中是独立的 80 步教学实验，固定 λ = 0.05，没有启用生产求解器的步长限幅和误差下降验收。实际 `solve_numerical` 的伪逆、DLS 分别在 7、8 次迭代内通过 10⁻⁶ m / 10⁻⁶ rad 验收；转置法 80 次迭代后位置误差约为 8.37 mm。伪逆在这一例同样收敛，不能把图解释为“伪逆遇到奇异就会失败”。这些实验与真实几何种子见 [portfolio_metrics.json](/assets/data/inverse-kinematics/portfolio_metrics.json)。

本次旋转权重为 1，可在代码中配置。雅可比的奇异值依赖平移、旋转行的尺度，混合残差不具有统一的物理单位。

</details>

## 5. 我怎样判断这些演示算是做对了

每张构型和每一帧动画都会重新跑正运动学，检查位置、朝向与关节限位；连续运动再检查相邻步长和循环接缝。测试还覆盖 Panda 的 MDH／MuJoCo 正运动学互校、雅可比有限差分、闭式回代和不可达目标。我把目标、关节角和误差留下来，是为了让图里的动作能够从代码重新算出来。

这些结果属于模型中的运动学实验，**还没有做完整碰撞检测、时间参数化、动力学或实机执行**。固定种子挑选的目标也不能代替工作空间成功率测试；七轴的有限采样不能代表全部逆解。限位规避、姿态偏好和可操作度次级目标已经实现，但阻尼下的 $N=I-J^\#J$ 只是近似零空间投影，次级步仍需经过主任务误差检查。

<details>
<summary>复现实验与打包主页素材</summary>

在项目仓库根目录运行，要求 Python 3.10+。模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，已有克隆时跳过 clone，切换到相同 commit 即可。

```bash
python -m pip install -e '.[showcase,test]' -c requirements-reproduce.txt
export MUJOCO_MENAGERIE=/path/to/mujoco_menagerie
git clone https://github.com/google-deepmind/mujoco_menagerie.git "$MUJOCO_MENAGERIE"
git -C "$MUJOCO_MENAGERIE" checkout bf756430b615819654b640f321c71ba5c3ebeef8
export MUJOCO_GL=egl

python -m scripts.showcase.build
python -m pytest -q
```

完整视频构建需要 ffmpeg、中文字体与离屏渲染环境。总入口把原始渲染写入本地 `build/showcase/`，再更新页面引用的发布资源。只打包现有素材时运行：

```bash
python -m scripts.showcase.export_homepage
```

输出为本地 `build/ik-homepage.zip`，不提交，也不会自动修改或部署个人站点。依赖、字体和复制到 Astro 的方法见 [复现说明](https://github.com/JiangShangJiu/ik/blob/main/docs/REPRODUCING.md)。五份 JSON 合计保留各阶段的源码与输入溯源，`portfolio_metrics.json` 另记录模型 XML 哈希和依赖版本；数值复现不要求时间戳、耗时、绝对路径和渲染字节一致。

</details>

延伸阅读：[逆运动学原理](/posts/逆运动学原理/) · [机械臂运动学](/posts/机械臂运动学/) · [项目源码](https://github.com/JiangShangJiu/ik)
