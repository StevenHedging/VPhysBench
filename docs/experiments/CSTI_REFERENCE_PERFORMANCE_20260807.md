# CSTI精确参考后端性能记录（2026-08-07）

## 目的与边界

本记录先保留未发布`exact_prefix_edt`原型的有限支持域裁剪前后资源基线，再记录最终
`exact_full_tube_edt`协议的校准和复测结果。历史全域oracle与历史默认后端都保持
逐prefix精确3D EDT；默认后端只省略soft membership在数学上必为0的空间voxel。

## 最大规格推导

以下规格对应历史逐prefix原型。prediction与same-Case GT使用共同存在的物理时间，
生成profile的最长常用输出为5秒；当时的碰撞配置在16 FPS、`960 × 540`原生评估画布
上采样，因此共同时间轴包含81帧。代表性GT为
`collision_r2_large_steel_medium_steel_small_steel_v01313`：其reference有1728帧、
213.443129 FPS、末帧物理时间8.091148秒，足以覆盖完整5秒prediction。

所以最大单主体体素规格为：

```text
T × H × W = 81 × 540 × 960 = 41,990,400
```

它大于同一5秒窗口下单摆的`81 × 832 × 480 = 32,348,160`体素。平抛虽使用
32 FPS，但当前reference仅约0.4秒；其它scene的时空体积也更小。

## 运行环境

```text
OS: Linux 6.8.0-71-generic x86_64, glibc 2.35
CPU: Intel Xeon Platinum 8558P, 180 logical CPUs visible
Python: 3.10.20
NumPy: 1.26.4
SciPy: 1.15.3
OpenCV: 4.11.0
```

容器没有GNU`/usr/bin/time`，因此峰值RSS由benchmark进程通过
`resource.getrusage(RUSAGE_SELF).ru_maxrss`直接记录；墙钟时间由
`time.perf_counter`围绕每个主体的CSTI调用记录。实际命令为：

```bash
PYTHONPATH=src /root/Steven/.venvs/wan22-pair-text/bin/python \
  scripts/benchmark_csti.py \
  --frames 81 \
  --height 540 \
  --width 960 \
  --objects 1 \
  --output /tmp/csti-max-benchmark-cropped.json
```

## 全域oracle基线

```json
{
  "algorithm": "exact_prefix_edt",
  "frames": 81,
  "height": 540,
  "objects": 1,
  "peak_rss_kib": 2956456,
  "scores": [
    0.98325401490544
  ],
  "wall_time_s": 351.1570487569843,
  "width": 960
}
```

运行以退出码0完成，没有OOM。峰值RSS约2.82 GiB，单主体墙钟时间约5分51秒。

## 默认精确有限支持域后端

默认后端对每个prefix取GT/pred联合前景的空间包围盒，并按
`ceil(tolerance_radius / spacing_axis)`向外扩展。裁剪外的两侧soft occupancy都严格为0，
所以intersection、union和因果时间前缀均不变。全域oracle继续由
`cumulative_soft_tube_iou_reference`保留，回归测试要求逐prefix绝对误差不超过
`1e-15`。

同一命令、同一输入的优化后结果为：

```json
{
  "algorithm": "exact_prefix_edt",
  "frames": 81,
  "height": 540,
  "objects": 1,
  "peak_rss_kib": 987980,
  "scores": [
    0.9832540149054403
  ],
  "wall_time_s": 67.08435176801868,
  "width": 960
}
```

相对全域oracle，墙钟提速5.23倍、峰值RSS降低2.99倍，分数差
`3.33e-16`。运行以退出码0完成，没有OOM。

多个主体由当前Case路径顺序评分，因此峰值内存不会按主体数叠加，但耗时近似线性增加。
direct CSTI Task有330个collision job，共726个GT主体；若每个主体都达到上述最大规格，
单进程保守上界约13.53小时。实际视频时长、前景包围盒通常更小，但CSTI仍是昂贵的精确
参考维度，规划正式运行时必须单独预留CPU时间；并发时还需叠加scene observer本身的
GPU/内存需求。未来快速后端必须逐prefix与全域oracle做数值一致性测试，不得静默改变
归一化间距、空Tube策略或GT主体分母。

## postcondition full-Tube 种子基准（2026-08-08）

上文是未发布v11的历史逐prefix实现。迭代校准的种子实现保持精确3D EDT和有限支持域
裁剪，但先剔除1个初始采样点，再以一次完整postcondition Tube Soft IoU作为正式分数；
25%/50%/75%/100%四个因果prefix仅作为诊断。空间和时间分别使用归一化轴容差
`0.005`与物理时间容差`0.05 s`。

同一81×540×960单主体输入的结果为：

```json
{
  "algorithm": "exact_full_tube_edt",
  "diagnostic_points": [4],
  "frames": 81,
  "height": 540,
  "initial_frames_excluded": 1,
  "objects": 1,
  "peak_rss_kib": 673536,
  "scored_frames": 80,
  "scores": [0.9611347245151881],
  "wall_time_s": 2.906011641025543,
  "width": 960
}
```

相对历史有限支持域逐prefix实现，单主体墙钟时间从67.084秒降至2.906秒（约23.1倍），
峰值RSS从987980 KiB降至673536 KiB（约1.47倍）。两项分数不能直接比较：新值排除了
初始样本并使用full-Tube聚合，旧值是逐prefix算术均值。最终协议参数与复测成本将在
Wan22网格校准完成后另行记录。

## 已固化v11配置与最终基准（2026-08-08）

对Wan22诊断run完成8/12/16/24 FPS网格、空间/时间容差和初始舍弃点数的联合扫描后，
`scene_default_v11`固化为：

```text
analysis_fps                = 24
spatial_tolerance_fraction  = 0.004204482076268572
temporal_tolerance_s        = 0.025
initial_frames_excluded     = 3
terminal_frame_policy       = exclude_off_grid_endpoint
score_aggregation           = full_tube
diagnostic_prefix_fractions = [0.25, 0.5, 0.75, 1.0]
```

媒体层可能为覆盖精确物理时长附加一个短于规则帧间隔的末端采样点；正式EDT在舍弃前三个
规则采样点之前，先且只先排除该off-grid末端点。其余非规则时间轴仍是契约错误。

最终24 FPS协议的5秒上限是121帧；把同一基准命令的`--frames`改为121后得到：

```json
{
  "algorithm": "exact_full_tube_edt",
  "diagnostic_points": [4],
  "frames": 121,
  "height": 540,
  "initial_frames_excluded": 3,
  "objects": 1,
  "peak_rss_kib": 946292,
  "sampling_fps": 24.0,
  "scored_frames": 118,
  "scores": [0.9647362722681895],
  "spatial_tolerance_fraction": 0.004204482076268572,
  "temporal_tolerance_s": 0.025,
  "wall_time_s": 4.531039699999383,
  "width": 960
}
```

### Wan22校准结果

校准输入为`run/wan22_pair_text_cross_attention_2184_v2`中的76个五场景finetune评测
job。最终普通评估入口的8卡复跑完成71个Case，5个因既有GT观察失败而保持
`unavailable`，无新增evaluator error。71个可评Case的CSTI场景宏平均为
`0.08242250320865943`，Case中位数为`0.020671947264249944`；旧种子协议在完全相同的
71个Case上为`0.32100262019258774`。分场景结果为：

| scene | evaluated/expected | CSTI mean |
| --- | ---: | ---: |
| `collision_1d` | 20/20 | 0.09503406472227227 |
| `inclined_plane_slide` | 14/15 | 0.11197207009405300 |
| `parabolic_motion` | 15/15 | 0.007751745504093772 |
| `pendulum` | 17/20 | 0.15527539313206162 |
| `uniform_circular_motion` | 5/6 | 0.04207924259081651 |

共同53个Case上的硬二值Tube-IoU下限候选（`r=0.001, tau=0.025 s, k=3`）场景宏平均为
`0.09783802747741362`。最终较宽空间容差在同一共同集上为
`0.10252295263835236`，仍处于预先固定的`+0.005`等价带内，因此选择对亚像素抖动更
稳定的较宽容差。`r=0.001`已小于所有原生分析轴的单像素归一化间距，`0.025 s`也小于
24 FPS的单帧间隔；继续缩小二者不会降低实际二值Tube重叠形成的硬下限，若强行压低只能
改变指标定义。

相同二值Tube的soft occupancy逐voxel完全相等，所以`S(x,x)=1.0`为精确恒等式，不依赖
上述容差取值；单元测试覆盖`k=1/2/3`，共享评估器集成测试也验证Case自比为1.0。

完整网格、refinement和最终普通入口结果分别保存在：

```text
run/wan22_pair_text_cross_attention_2184_v2/diagnostics/csti_calibration_20260808/
run/wan22_pair_text_cross_attention_2184_v2/diagnostics/csti_scene_default_v11_calibrated_20260808/
```

该历史run没有`frozen/assets.lock.json`，所以这些产物明确标为
`diagnostic_non_canonical`，用于参数校准而不是发布Leaderboard分数。输入
`plan.json`、`frozen/cases.jsonl`和`predictions.jsonl`的SHA256均随结果保存。
