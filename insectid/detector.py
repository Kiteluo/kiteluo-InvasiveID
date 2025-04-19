import sys
import torch
import numpy as np
import cv2
from pathlib import Path

# 从YOLOv5源码导入必要模块
sys.path.append(str(Path(__file__).parent.parent))  # 添加根路径
from yolov5s.models.experimental import attempt_load
from yolov5s.utils.general import (
    non_max_suppression,
    check_img_size,
    xyxy2xywh
)
from yolov5s.utils.augmentations import letterbox
def scale_coords(img1_shape, coords, img0_shape, ratio=None, pad=None):
    """
    坐标缩放函数（兼容旧版YOLOv5）
    :param img1_shape: 预处理后的图像尺寸 (H, W)
    :param coords: 预测框坐标 (tensor[N, 4])
    :param img0_shape: 原始图像尺寸 (H, W)
    :param ratio: 缩放比例 (w_ratio, h_ratio)
    :param pad: 填充量 (dw, dh)
    :return: 缩放后的坐标 (tensor[N, 4])
    """
    if ratio is None:  # 从img1_shape计算比例
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 实际是旧版逻辑
        wh_padding = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:  # 使用letterbox返回的参数
        gain = ratio[0], ratio[1]
        wh_padding = pad[0], pad[1]
    
    coords[:, [0, 2]] -= wh_padding[0]  # x padding
    coords[:, [1, 3]] -= wh_padding[1]  # y padding
    coords[:, :4] /= torch.tensor([gain[0], gain[1], gain[0], gain[1]]).to(coords.device)
    
    # 确保坐标在图像范围内
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])  # x
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])  # y
    return coords
class YOLOv5Detector:
    def __init__(self, 
                 weights_path='YOLOv5的权重路径', 
                 img_size=640, 
                 conf_thres=0.25,  # 默认参数对齐官方
                 iou_thres=0.45,
                 device=None):
        """
        初始化YOLOv5检测器
        :param weights_path: 模型权重路径
        :param img_size: 输入图像尺寸（必须为32的倍数）
        :param conf_thres: 置信度阈值
        :param iou_thres: NMS的IoU阈值
        """


       # 设备选择逻辑更新
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载模型
        self.model = attempt_load(weights_path, device=self.device)
        self.model.eval()  # 强制eval模式
        self.stride = int(self.model.stride.max())  # 模型步长
        self.img_size = check_img_size(img_size, s=self.stride)  # 验证尺寸有效性
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # 获取类别名称
        
        # 推理参数
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Warmup模型（避免首次推理延迟）
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))

    def preprocess(self, img):
        """
        图像预处理（返回处理后的张量及缩放参数）
        :param img: 输入图像（numpy数组，BGR格式）
        :return: 
            img_tensor: 预处理后的张量 (1, 3, H, W)
            ratio: 缩放比例 (w_ratio, h_ratio)
            (dw, dh): 两侧填充量
        """
        # Letterbox缩放（保持长宽比）
        img_resized, ratio, (dw, dh) = letterbox(img, self.img_size, stride=self.stride, auto=True)
        
        # 转换通道顺序 BGR → RGB
        img_rgb = img_resized[:, :, ::-1].transpose(2, 0, 1)  # HWC → CHW
        img_rgb = np.ascontiguousarray(img_rgb)
        
        # 转为Tensor并归一化
        img_tensor = torch.from_numpy(img_rgb).to(self.device)
        img_tensor = img_tensor.float() / 255.0  # 归一化到[0,1]
        return img_tensor.unsqueeze(0), ratio, (dw, dh)

    def detect(self, img):
        """
        执行目标检测
        :param img: 输入图像（支持文件路径或numpy数组）
        :return: List[Dict] 检测结果列表
        """
        # 输入处理
        if isinstance(img, str):
            img = cv2.imread(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError("输入必须是文件路径或numpy数组")
        
        orig_h, orig_w = img.shape[:2]  # 原始尺寸

        # 预处理（获取处理后的张量和缩放参数）
        img_tensor, ratio, (dw, dh) = self.preprocess(img)

        # 推理
        with torch.no_grad():
            pred = self.model(img_tensor, augment=False)[0]  # 关闭数据增强

        # NMS处理
        pred = non_max_suppression(pred, 
                                  conf_thres=self.conf_thres,
                                  iou_thres=self.iou_thres,
                                  agnostic=False)[0]  # 非跨类别NMS

        detections = []
        if pred is not None and len(pred):
            # 坐标缩放（关键修正！）
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], (orig_h, orig_w, 3), ratio, (dw, dh))
            
            for *xyxy, conf, cls_id in pred:
                detections.append({
                    'bbox': [int(x) for x in xyxy],
                    'conf': float(conf),
                    'cls_id': int(cls_id),
                    'cls_name': self.names[int(cls_id)]  # 添加类别名称
                })

        return detections

    @staticmethod
    def plot_results(image, detections, color=(255,0,0), thickness=2):
        """可视化检测结果（实现与原版一致）"""
        img_draw = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # 绘制矩形框
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
            # 构建标签文本
            label = f"{det['cls_name']} {det['conf']:.2f}"
            # 计算文本位置
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)
            # 绘制文本背景
            cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            # 绘制文本
            cv2.putText(img_draw, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), thickness)
        return img_draw

"""
下面的调试是单独检测YOLOv5的

if __name__ == '__main__':
    # 示例用法
    detector = YOLOv5Detector(conf_thres=0.25)
    test_img = cv2.imread("F:\\kiteluo-InvasiveID\\pipeline\\test.jpg")
    
    # 执行检测
    results = detector.detect(test_img)
    print(f"检测到 {len(results)} 个目标:")
    for i, det in enumerate(results):
        print(f"目标{i+1}: {det['cls_name']} 位置={det['bbox']}, 置信度={det['conf']:.2f}")
    
    # 可视化（直接显示BGR格式）
    vis_img = detector.plot_results(test_img, results)
    cv2.namedWindow("Detection Results", cv2.WINDOW_NORMAL)  # 允许调整窗口大小
    cv2.imshow("Detection Results", vis_img)

    cv2.waitKey(0)
"""