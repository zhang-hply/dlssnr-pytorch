# Distillation training and validation flowchart prompts

- Source commit: `d6d4e8d08c1fb61682b1189e9d36bdb4ab056771`
- Mode: built-in `image_gen`, following the imagegen skill.
- Training image: `distillation-training-d6d4e8d-flowchart.png`
- Validation image: `distillation-validation-d6d4e8d-flowchart.png`
- Numbered continuation terminals in the training image denote the epoch and image loops.

## Training image: final selected prompt

```text
Create a professional detailed Chinese training flowchart, portrait white poster, blue process boxes, amber diamonds, green final output, crisp Chinese text. Title "训练主流程：裁剪模型恢复蒸馏", subtitle "图1/2 · dlssnr-pytorch · commit d6d4e8d". Parameter strip "20 epochs · batch 1 · 累计4张 · AdamW · lr 1e-5 → 1e-6".
Use these exact nodes and connections:
A 校验权重、离线教师缓存、固定裁剪集合（同尺寸，保留 MACs ≤80%）
→B 构建学生、AdamW、余弦调度（所有保留权重使用 FP32 主参数）
→C diamond restored is None？
C 是→D 验证裁剪基线，best = baseline，保存 epoch 0
C 否→E 恢复主权重与训练状态（优化器、调度器、epoch、历史、RNG）
D/E merge→F 完整前向与反向预检查（清空梯度，此处不更新权重）
→G diamond 还有 epoch 且 bad_epochs <5？
G 否→P green 重载 best.safetensors，再次验证（比较初始基线，写入最终报告）
G 是→H 开始 epoch（seed + epoch 打乱图片，zero_grad）
→I 读取原图与教师 target（原分辨率整帧训练）
→J 量化前向并计算蒸馏损失（L = L1 + 0.05 × (Lx + Ly)）
→K (loss / group_size).backward()
→L ONE BLUE BOX: 条件更新：仅累计满4张或本轮最后一张时执行 / 梯度范数上限1.0 → AdamW.step → zero_grad / global_step += 1；否则保留累计梯度
→M diamond 本轮还有图片？
M 是→ small continuation TERMINAL "② 读取下一张：返回 I"
M 否→N 调用图2：每轮验证与保存
N→small continuation TERMINAL "① 下一轮：返回 G 检查 epoch 与早停"
IMPORTANT: use textual continuation terminals to denote loops. DO NOT draw long return arrows, DO NOT connect N to P. There are no long outer or inner loop lines. Both continuation terminals end without outgoing arrows. Add small entry labels "①" next to G and "②" next to I to match them. This is intentional standard flowchart off-page connector notation.
Footer exact:
教师目标离线生成；训练时不运行教师
FP32 主权重：前向量化，反向代理梯度
Lx / Ly：水平 / 垂直像素差分的 L1 误差
最后不足4张按实际组大小平均；从完整 epoch 恢复
No extra connections, no watermark.
```

## Validation image: final selected prompt

```text
Use case: infographic-diagram
Create one clean, detailed Chinese validation/checkpoint flowchart as a raster image. Source: dlssnr-pytorch commit d6d4e8d08c1fb61682b1189e9d36bdb4ab056771.
Portrait technical poster matching a professional software flowchart: white background, large crisp Simplified Chinese sans-serif text, blue rounded process boxes, amber decision diamonds, green checkpoint/output boxes. Clear orthogonal arrows, no overlapping connectors/text, no decorative imagery or watermark.
Title: "每轮验证、最佳模型与检查点"
Subtitle: "图 2/2 · dlssnr-pytorch · commit d6d4e8d"

Draw this exact top-to-bottom control flow:
A "输入：当前学生与验证缓存"
A -> B "物化当前量化权重" / "student.export_state()"
B -> C "加载到普通 NRNet 推理模型" / "inference_mode，无梯度"
C -> D "读取一张验证原图与教师 target"
D -> E "原始推理路径生成预测" / "累计平方误差和像素数量" / "另计算 uint8 PSNR / SSIM"
E -> F diamond "还有验证图片？"
F YES "是" loops outside to D.
F NO "否" -> G "汇总整体验证指标" / "MSE = 总平方误差 / 总像素通道数" / "PSNR = 10 log10(1 / MSE)"
G -> H diamond "val MSE < best MSE？"
H YES "是" -> I "更新 best 与 best_epoch" / "bad_epochs = 0"
H NO "否" -> J "bad_epochs += 1"
I and J merge to K "完成本轮记录与学习率调度" / "history.append → scheduler.step()" / "更新 report.md / report.json"
K -> L diamond "本轮 improved？"
L YES "是" -> M green output "保存最佳结果" / "best.pt" / "best.safetensors + best.json"
L NO "否" bypasses M.
Both routes merge to N green output "总是保存 last.pt" / "主权重、优化器、调度器、epoch、RNG"
N -> O "返回图1：下一轮 / 提前停止检查"

Put these readable footer notes in a separated small box:
"最佳模型按整体浮点 MSE 严格下降选择"
"浮点 PSNR 的峰值为1；uint8 PSNR 的峰值为255"
"scheduler.step 每个完整 epoch 执行一次"
"初始基线和最终重载只复用上方推理评分步骤"
"若训练从未改善验证 MSE，最佳结果保持 epoch 0"
Critical accuracy: validation never performs optimizer.step or backward. The checkpoint save occurs AFTER scheduler.step. Only improved epochs overwrite best artifacts; every completed epoch saves last.pt. Do not confuse mean per-image PSNR with pooled PSNR. Early stopping itself is tested in the training flowchart after this returns. Keep all branch labels clear and typography highly legible.
```
