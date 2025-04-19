import torch
import cv2
import os
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch import nn
from collections import OrderedDict
import sys

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

# ---- 模型配置 ----
model_cfg = dict(
    backbone=dict(
        arch='large',      
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01)
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=13,
        in_channels=960,
        mid_channels=[512],                    #根据数据集来进行修改 大样本可修改回1024
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
    )
)

# ---- 完整模型实现 ----
class MobileNetV3Complete(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Backbone（来自原始实现）
        from MobileNetV3.configs.backbones.mobilenet_v3 import MobileNetV3  # 确保路径正确
        backbone_cfg = cfg['backbone'].copy()
        self.backbone = MobileNetV3(**backbone_cfg)
        
        # Neck
        self.neck = self._build_neck(cfg['neck'])
        
        # Head
        self.head = self._build_head(cfg['head'])
        
        # 初始化
        self._init_weights()

    def _build_neck(self, cfg):
        return nn.AdaptiveAvgPool2d(1)

    def _build_head(self, cfg):
        layers = []
        in_dim = cfg['in_channels']
        
        # 构建中间层
        for idx, mid_dim in enumerate(cfg['mid_channels']):
            layer = nn.Sequential(OrderedDict([
                (f'fc', nn.Linear(in_dim, mid_dim)),
                ('act', self._get_activation(cfg['act_cfg'])),
                ('dropout', nn.Dropout(p=cfg['dropout_rate']))
            ]))
            layers.append(layer)
            in_dim = mid_dim
        
        # 输出层
        output_layer = nn.Sequential(OrderedDict([
            (f'fc', nn.Linear(in_dim, cfg['num_classes']))
        ]))
        layers.append(output_layer)
        
        return nn.ModuleList(layers)

    def _get_activation(self, cfg):
        if cfg['type'] == 'HSwish':
            return nn.Hardswish()
        return nn.ReLU()

    def _init_weights(self):
        # Head初始化
        for layer in self.head:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone前向
        features = self.backbone(x)[-1]  # 取最后一个特征图
        
        # Neck前向
        pooled = self.neck(features)
        flattened = pooled.view(pooled.size(0), -1)
        
        # Head前向
        for layer in self.head:
            flattened = layer(flattened)
        return flattened

# ---- 分类器封装 ----
class MobileNetClassifier:
    def __init__(self, model_weights: str, device='auto'):
        self.device = self._get_device(device)
        self.model = MobileNetV3Complete(model_cfg)
        self._load_weights(model_weights)
        self.model.to(self.device)
        
        # 半精度初始化（关键修改点1）
        if self.device.type == 'cuda':
            self.model.half()  # 转换模型权重到半精度

        self.model.eval()
        
        # 预处理流程修正（关键修改点2）
        self.preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.astype(np.uint8)),  
            transforms.ToTensor(),
            # 移除冗余的Resize和CenterCrop（原256→224缩放）
            transforms.Normalize(
                mean=[123.675/255, 116.28/255, 103.53/255],
                std=[58.395/255, 57.12/255, 57.375/255]
            )
        ])
        # 注意：需确保此处归一化参数与训练时完全一致！
        
        # 类别映射
        self.class_names = [ 'Bemisia_tabaci', 'Brontispa_longissima', 'Corythucha_ciliata',
            'Cydia_pomonella', 'Hyphantria_cunea', 'Leptinotarsa_decemlineata',
            'Liriomyza_sativae', 'Lissorhoptrus_oryzophilus', 'Opogona_sacchari',
            'Phenacoccus_solenopsis', 'Rhynchophorus_ferrugineus',
            'Solenopsis_invicta', 'Spodoptera_frugiperda']  # 保持原有类别列表 

    def _get_device(self, device_type):
        if device_type == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_type)

    def _load_weights(self, weight_path: str):
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 键名转换
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # 处理backbone前缀
            if key.startswith('layer'):
                new_key = f'backbone.{key}'
            # 处理head前缀
            elif key.startswith('head.layers'):
                parts = key.split('.')
                layer_idx = parts[2]
                param_type = parts[-1]
                new_key = f'head.{layer_idx}.fc.{param_type}'
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # 调试信息（已注释）
        # print("=== 转换后的前5个键 ===")
        # print(list(new_state_dict.keys())[:5])
        # print("=== 模型期待的前5个键 ===")
        # print(list(self.model.state_dict().keys())[:5])
        
        # 加载权重
        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            # print("✅ 权重加载成功")  # 调试信息（已注释）
        except RuntimeError as e:
            self._handle_load_error(e)

    def _handle_load_error(self, e):
        error_msg = str(e)
        missing = []
        unexpected = []
        
        if 'Missing key(s)' in error_msg:
            missing_part = error_msg.split('Missing key(s):')[1].split('Unexpected key(s):')[0].strip()
            missing = [x.strip().strip("'") for x in missing_part.split(',')]
            
        if 'Unexpected key(s)' in error_msg:
            unexpected_part = error_msg.split('Unexpected key(s):')[1].strip()
            unexpected = [x.strip().strip("'") for x in unexpected_part.split(',')]
        
        # 调试信息（已注释）
        # print(f"❌ 缺失键数量: {len(missing)}, 示例: {missing[:3]}")
        # print(f"❌ 多余键数量: {len(unexpected)}, 示例: {unexpected[:3]}")
        raise RuntimeError("权重加载失败，请检查模型结构和权重文件的兼容性") from e

    def classify(self, image: np.ndarray) -> dict:
        # 输入验证增强

        if image is None:
            raise ValueError("输入图像为空")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.shape[-1] != 3:
            raise ValueError("输入图像必须为三通道BGR格式")
        
         # 预处理（添加半精度处理）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(rgb_image).unsqueeze(0)
        
        # 半精度转换（关键修改点3）
        if self.device.type == 'cuda':
            tensor = tensor.half()  # 转换为FP16
        tensor = tensor.to(self.device)
        # 推理
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        # 解析结果
        conf, class_id = torch.max(probs, dim=1)
        return {
            'class_id': class_id.item(),
            'class_name': self.class_names[class_id.item()],
            'confidence': conf.item()
        }

# ---- 使用示例 ----
if __name__ == "__main__":
    classifier = MobileNetClassifier(
        model_weights="MobileNetV3的权重路径",
        device='auto'
    )
    test_path = f"测试图像路径"  # 使用原始字符串
    if not os.path.exists(test_path):
      raise FileNotFoundError(f"图像路径不存在: {test_path}")
    test_img = cv2.imread(test_path)
    result = classifier.classify(test_img)
    
    print(f"""
    分类结果:
    - 类别ID: {result['class_id']}
    - 类别名称: {result['class_name']}
    - 置信度: {result['confidence']:.4f}
    """)