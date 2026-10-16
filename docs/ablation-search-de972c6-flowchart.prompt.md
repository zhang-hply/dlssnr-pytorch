# Flowchart prompt for commit de972c6

- Source commit: `de972c69169a97f66b8ae57cded1b2aaaa2d1a5e`
- Generation mode: built-in `image_gen` (imagegen skill)
- Intended image: `ablation-search-de972c6-flowchart.png`

```text
Use case: infographic-diagram
Create one high-resolution Chinese software flowchart as a clean raster image for the dlssnr-pytorch repository. It must document exactly commit de972c69169a97f66b8ae57cded1b2aaaa2d1a5e, "Add block pruning combination search and reports". Do not depict later distillation or detailed logging features.
Style: clean technical documentation, white background, dark readable Simplified Chinese sans-serif text, blue process boxes, amber decision diamonds, green result boxes, crisp arrowheads, generous whitespace. Portrait poster approximately 1536 x 2048, large legible labels. No decoration, unrelated icons, logos, watermark, or extraneous text.
Title verbatim: "Block 裁剪组合搜索流程"
Subtitle verbatim: "dlssnr-pytorch · commit de972c6"
Parameter strip verbatim: "保留 MACs ≤80%   |   最多 400 组   |   束宽 2 · 扩展 6   |   Top-10"

Use a readable central top-to-bottom flow, with a clearly routed loop and two distinct YES/NO branches. Process labels below must be rendered faithfully:
1. "读取与校验输入" / "单图或同尺寸文件夹 · 可选参考图"
2. "构建计划与核算 MACs" / "59 个候选：36 Swin + 15 Split + 8 ViT"
3. "完整模型基准推理" / "缓存每张图片的完整输出"
4. "逐块评估并排序" / "59 次单块跳过，按 MSE / 节省 MACs 排序"
5. "生成并实测保底组合" / "累加删除，直到保留 MACs ≤80%"
6. "初始化束搜索" / "从空删除集合开始"
7. "扩展候选并去重" / "每条路径添加前 6 个未删块"
8. "评估组合或复用缓存" / "新组合先检查 400 组预算" / "block → layer indices → net.skip" / "多图累计 MSE，再换算 PSNR"
9. Diamond: "剩余 MACs ≤80%？"
YES branch labeled "是" leads to green box "加入可行结果池" / "该组合停止扩展".
NO branch labeled "否" leads to process box "按 MSE / 已节省 MACs" / "保留最佳 2 条路径".
These two branches converge into a SECOND diamond: "预算仍可用且有路径？"
The second diamond YES branch labeled "是，继续" loops outside the boxes back to step 7. Its NO branch labeled "否，结束" goes down to the final steps.
10. "可行结果按 MSE 排序" / "取 Top-10，补算 SSIM"
11. Document/output shape: "输出 Markdown + JSON" / "最佳组合 · block / layer · MACs" / "画质指标 · 全部记录 · 停止原因"

A small dashed side note connected to step 8: "每个新组合完成后增量保存"
Bottom notes verbatim:
"400 组仅计唯一组合；基准与 SSIM 重跑另计"
"参考图指标不参与搜索排名"
"中断或异常：保存已完成结果，重置 net.skip"
"启发式搜索，不保证全局最优；MACs 不等于实际延迟"

Accuracy constraints: the seed is an actual evaluated fallback, but the beam starts from an empty deletion set. Cache hits do not consume a new evaluation. Only newly inferred combinations count toward 400. Feasible combinations are terminals, not further pruned. New combinations are truly evaluated, never assigned the sum of single-block errors. Budget exhaustion before a new evaluation also leads to final Top-10/report processing. The whole chart must be readable with no overlapping text or ambiguous arrows.
```

