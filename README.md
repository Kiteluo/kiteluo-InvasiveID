# 基于YOLOv5与MobileNetV3的入侵昆虫识别系统

## 项目概述

一个完整的入侵昆虫识别系统，结合了YOLOv5s作为detector和MobileNetV3作为idfier。本项目作为本科毕设尚且可以。MobileNetV3的训练参考了DOI:10.13733/j.jcam.issn.2095 5553.2024.07.033这篇文章

## 快速开始

1. 克隆本仓库
2. 安装依赖：`pip install -r requirements.txt`
3. 运行演示程序：`python demo.py`

注意：运行前需要根据本地环境修改相关配置参数。



## 数据集

本项目使用的13类入侵昆虫数据集已在[另一个仓库](https://github.com/Kiteluo/datasets)中开源。数据集主要来源于iNaturalist平台的研究级图像。

数据集特点：
- 包含13类入侵昆虫
- 研究级图像质量
- 适用于初步昆虫识别研究

## 配置与使用

1. 修改`demo.py`中的配置参数：
   - 模型路径
   - 输入/输出路径
   - 其他运行时参数
2. 注意事项：
   - 需要自行训练模型或使用预训练权重
