# Kimodo-SOMA-RP 后训练探索

> 金山青训营结题项目

本项目围绕 NVIDIA 开源的 Kimodo-SOMA-RP-v1.1 展开，探索在官方后训练代码和原始商业训练数据未开放的条件下，建立可复现的后训练方法，以增强模型生成攻击、跳跃、浮空和击倒等游戏动作的能力。

项目完成了从数据制备、Prompt 标注、模型训练、Checkpoint 推理到定量评测和 Blender 可视化的原型闭环。实验表明，高质量游戏动作数据能够让模型学习目标领域动作，同时也需要进一步解决小规模数据全参数微调导致的原始能力下降问题。

## 内容导览

- [`index.html`](./index.html)：交互式结题展示入口，包含 Blender 插件演示和 13 个结题章节。
- [`full-report.html`](./full-report.html)：逐章完整报告。
- [`Kimodo-SOMA-RP-report.pdf`](./Kimodo-SOMA-RP-report.pdf)：48 页最终版 PDF 报告。
- [`source-report.md`](./source-report.md)：Markdown 报告原文。
- [`assets/final`](./assets/final)：实验图片、动图与 Blender 插件演示视频。

## 本地查看

1. 克隆或下载本仓库，并保留原有文件夹结构。
2. 使用 Chrome、Edge 或 Safari 等现代浏览器打开 `index.html`。
3. 页面右上角可进入完整原文或查看最终 PDF，实验 Case 支持点击放大。

本展示为离线版本，不依赖互联网资源。

## 操作提示

- 使用 `↑` / `↓` 方向键或 `PageUp` / `PageDown` 可在相邻章节间切换。
