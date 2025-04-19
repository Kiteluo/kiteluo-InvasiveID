# demo.py
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from insectid.detector import YOLOv5Detector
from insectid.idfinder import MobileNetClassifier

class PestAnalysisPipeline:
    def __init__(self, 
                 detector_weights: str,
                 classifier_weights: str,
                 detection_conf: float = 0.3,
                 classify_conf: float = 0.5,
                 device: str = 'auto',
                 insid_map_path: str = 'insectid/kiteluo_insid_map.txt',
                 font_path: str = 'PingFang Regular.ttf'):
        """
        :param detector_weights: YOLOv5检测模型路径
        :param classifier_weights: MobileNet分类模型路径
        :param detection_conf: 检测置信度阈值
        :param classify_conf: 分类结果置信度阈值
        :param device: 运行设备 (auto/cpu/cuda)
        :param insid_map_path: 物种信息映射文件路径
        :param font_path: 中文字体文件路径
        """
        # 设备统一处理
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化检测器和分类器
        self.detector = YOLOv5Detector(
            weights_path=detector_weights, 
            conf_thres=detection_conf,
            device=self.device
        )
        self.classifier = MobileNetClassifier(
            model_weights=classifier_weights,
            device=self.device
        )
        
        # 加载映射表
        self.insid_map = self._load_insid_map(insid_map_path)
        
        # 加载字体文件（关键修复）
        self.font = None
        font_path = Path(font_path)
        if font_path.exists():
            try:
                self.font = ImageFont.truetype(str(font_path), 20)
                print(f"✅ 字体加载成功: {font_path}")
            except Exception as e:
                print(f"‼️ 字体加载失败: {str(e)}")
                self.font = ImageFont.load_default()
        else:
            print(f"⚠️ 字体文件不存在: {font_path}")
            self.font = ImageFont.load_default()
        
        # 初始化阈值参数（关键修复）
        self.classify_conf = classify_conf  # 存储原始配置值
        self.classify_conf_thres = classify_conf
        
        # 打印配置信息
        print("\n=== 配置信息 ===")
        print(f"检测置信度阈值: {detection_conf}")
        print(f"分类置信度阈值: {classify_conf}")
        print(f"设备: {self.device}")
        print("===============\n")

    def _load_insid_map(self, path: str) -> dict:
        """解析物种信息映射文件"""
        insid_map = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ',' in line:
                    chinese_part, sci_part = line.split(',', 1)
                    sci_name = sci_part.split('#')[0].strip().replace(' ', '_')
                    insid_map[sci_name] = {
                        'chinese': chinese_part.strip(),
                        'scientific': sci_part.split('#')[0].strip()
                    }
        return insid_map

    def _merge_boxes(self, detections, iou_threshold=0.5):
        """改进的NMS合并方法"""
        if not detections:
            return []
        if len(detections) == 1:
            return detections  # 单框直接保留

        # 坐标转换
        boxes = [
            [det['bbox'][0], det['bbox'][1], 
             det['bbox'][2]-det['bbox'][0], 
             det['bbox'][3]-det['bbox'][1]] 
            for det in detections
        ]
        scores = [det['conf'] for det in detections]

        # 执行NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, 
            score_threshold=0.01, 
            nms_threshold=iou_threshold
        )
        return [detections[i] for i in indices] if indices else []

    def _crop_and_resize(self, img, bbox):
        """优化裁剪逻辑"""
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        
        # 保持长宽比缩放
        h, w = crop.shape[:2]
        scale = 224 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 居中填充
        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        dx, dy = (224 - new_w) // 2, (224 - new_h) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized
        return canvas

    def process_image(self, img_path: str, output_dir: str = None):
        """核心处理流程"""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")

        # 检测阶段
        raw_detections = self.detector.detect(img)
        print(f"[检测] 原始框数量: {len(raw_detections)}")

        # 合并阶段
        detections = self._merge_boxes(raw_detections)
        print(f"[NMS] 有效框数量: {len(detections)}")

        results = []
        for idx, det in enumerate(detections):
            crop_img = self._crop_and_resize(img, det['bbox'])
            if crop_img is None:
                continue

            try:
                # 分类阶段
                cls_result = self.classifier.classify(crop_img)
                print(f"[分类] 结果: {cls_result['class_name']} 置信度: {cls_result['confidence']:.2f}")

                # 置信度过滤
                if cls_result['confidence'] < self.classify_conf_thres:
                    print(f"🚫 过滤低置信度结果 ({cls_result['confidence']:.2f} < {self.classify_conf_thres})")
                    continue

                # 映射校验
                map_info = self.insid_map.get(
                    cls_result['class_name'],
                    {'chinese': '未知物种', 'scientific': 'Unknown'}
                )
                results.append({
                    'bbox': det['bbox'],
                    'conf': cls_result['confidence'],
                    'chinese': map_info['chinese'],
                    'scientific': map_info['scientific']
                })

            except Exception as e:
                print(f"‼️ 分类失败: {str(e)}")
                continue

        # 可视化与保存
        vis_img = self._draw_chinese(img, results)
        if output_dir:
            output_path = Path(output_dir) / f"result_{Path(img_path).name}"
            cv2.imwrite(str(output_path), vis_img)
            print(f"✅ 结果保存: {output_path}")

        return vis_img, results

    def _draw_chinese(self, image, results):
        """兼容性可视化方法"""
        try:
            if not results:
                return image

            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            for result in results:
                x1, y1, x2, y2 = result['bbox']
                label = f"{result['chinese']} {result['scientific']} {result['conf']:.2f}"

                # 计算文本尺寸
                if hasattr(self.font, 'getbbox'):
                    bbox = self.font.getbbox(label)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    text_width, text_height = self.font.getsize(label)

                # 绘制文本背景
                draw.rectangle(
                    [x1, y1-text_height-5, x1+text_width+10, y1],
                    fill=(0, 150, 0)
                )
                
                # 绘制文本
                draw.text(
                    (x1+5, y1-text_height-2),
                    label,
                    font=self.font,
                    fill=(255, 255, 255)
                )

                # 绘制边框
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"‼️ 可视化错误: {str(e)}")
            return image

if __name__ == "__main__":
    # ================= 配置参数 =================
    CONFIG = {
        "input_path": "改写为你的图片路径",
        "output_dir": "改写为识别后的目标路径",
        "detector_weights": "YOLOv5的权重路径",
        "classifier_weights": "MobileNetV3的权重路径",
        "detection_conf": 0.3,
        "classify_conf": 0.5,
        "font_path": "字体路径"  # 绝对路径
    }

    # ================= 初始化流水线 =================
    pipeline = PestAnalysisPipeline(
        detector_weights=CONFIG["detector_weights"],
        classifier_weights=CONFIG["classifier_weights"],
        detection_conf=CONFIG["detection_conf"],
        classify_conf=CONFIG["classify_conf"],
        font_path=CONFIG["font_path"]
    )

    # ================= 处理输入 =================
    input_path = Path(CONFIG["input_path"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        print(f"处理文件: {input_path}")
        pipeline.process_image(str(input_path), str(output_dir))
    elif input_path.is_dir():
        print(f"批量处理目录: {input_path}")
        for img_file in input_path.glob('*.[jJ][pP][gG]'):  # 支持大小写
            print(f"处理: {img_file.name}")
            pipeline.process_image(str(img_file), str(output_dir))

    print(f"✅ 处理完成！结果目录: {output_dir.resolve()}")