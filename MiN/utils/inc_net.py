import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear
# Import autocast để tắt nó trong quá trình tính toán ma trận chính xác cao
from torch.cuda.amp import autocast 

class BaseIncNet(nn.Module):
    def __init__(self, args: dict):
        super(BaseIncNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.feature_dim = self.backbone.out_dim
        self.fc = None

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    @staticmethod
    def generate_fc(in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        hyper_features = self.backbone(x)
        logits = self.fc(hyper_features)['logits']
        return {
            'features': hyper_features,
            'logits': logits
        }


class RandomBuffer(torch.nn.Linear):
    """
    Lớp mở rộng đặc trưng ngẫu nhiên (Random Projection).
    """
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [QUAN TRỌNG] Sử dụng float32 để đảm bảo độ chính xác khi tính RLS
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Ép kiểu input X về cùng kiểu với weight (float32)
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.W)


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        
        # Các tham số cho Analytic Learning (RLS)
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim 

        # Random Buffer
        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # Khởi tạo ma trận trọng số và ma trận hiệp phương sai cho RLS
        # Dùng float32 để tránh lỗi singular matrix khi tính nghịch đảo
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight) # Trọng số của Analytic Classifier

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R) # Ma trận hiệp phương sai đảo (Inverse Covariance Matrix)

        # Normal FC: Dùng để train Gradient Descent cho Noise Generator
        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def update_fc(self, nb_classes):
        """
        Cập nhật lớp Normal FC (cho việc training Noise).
        Lớp Analytic FC (self.weight) sẽ tự động mở rộng trong hàm fit().
        """
        self.cur_task += 1
        self.known_class += nb_classes
        
        # Tạo mới Normal FC cho task hiện tại
        if self.cur_task > 0:
            # Task sau: Không dùng Bias để tránh bias vào lớp mới quá nhiều
            new_fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            # Task đầu: Có bias
            # [FIX LỖI TẠI ĐÂY]: Đổi fc thành new_fc
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            
        if self.normal_fc is not None:
            # Sequential Init: Copy trọng số cũ
            old_nb_output = self.normal_fc.out_features
            with torch.no_grad():
                # Copy phần cũ
                new_fc.weight[:old_nb_output] = self.normal_fc.weight.data
                # Init phần mới về 0
                nn.init.constant_(new_fc.weight[old_nb_output:], 0.)
            
            del self.normal_fc
            self.normal_fc = new_fc
        else:
            # Task đầu tiên
            nn.init.constant_(new_fc.weight, 0.)
            if new_fc.bias is not None:
                nn.init.constant_(new_fc.bias, 0.)
            self.normal_fc = new_fc

    # =========================================================================
    # [MAGMAX & NOISE CONTROL SECTION]
    # =========================================================================
    
    def update_noise(self):
        """
        Gọi khi bắt đầu Task mới.
        Kích hoạt chế độ Sequential Initialization trong PiNoise.
        """
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()

    
    def unfreeze_noise(self):
        """Gọi cho Task > 0: Chỉ unfreeze Noise thưa"""
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_incremental()

    def init_unfreeze(self):
        for j in range(len(self.backbone.noise_maker)):
            self.backbone.noise_maker[j].unfreeze_task_0()
            
            # Giữ LayerNorm trainable ở Task 0 để ổn định base
            if hasattr(self.backbone.blocks[j], 'norm1'):
                for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            if hasattr(self.backbone.blocks[j], 'norm2'):
                for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
                
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters(): p.requires_grad = True
    # =========================================================================
    # [ANALYTIC LEARNING (RLS) SECTION]
    # =========================================================================

    def forward_fc(self, features):
        """Forward qua Analytic Classifier"""
        # Đảm bảo features cùng kiểu với trọng số RLS (float32)
        features = features.to(self.weight.dtype) 
        return features @ self.weight
    
           

    def forward(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [SỬA]: Đảm bảo đặc trưng đồng nhất kiểu dữ liệu trước khi vào Buffer
        hyper_features = hyper_features.to(self.weight.dtype)
        
        # Buffer trả về ReLU(X @ W), forward_fc thực hiện X @ Weight
        logits = self.forward_fc(self.buffer(hyper_features))
        
        return {'logits': logits}
    def extract_feature(self, x):
        """Chỉ trích xuất đặc trưng từ Backbone"""
        return self.backbone(x)

    def forward_normal_fc(self, x, new_forward: bool = False):
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [SỬA]: Buffer thường chứa trọng số FP32, ép hyper_features lên FP32 
        # để phép nhân trong Buffer diễn ra chính xác trước khi đưa vào Classifier
        hyper_features = self.buffer(hyper_features.to(self.buffer.weight.dtype))
        
        # Sau đó ép về kiểu của normal_fc (thường là Half nếu dùng autocast)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype)
        
        logits = self.normal_fc(hyper_features)['logits']
        return {"logits": logits}
    def collect_projections(self, mode='threshold', val=0.95):
        """
        Duyệt qua các lớp PiNoise và tính toán ma trận chiếu.
        """
        print(f"--> [IncNet] Collecting Projections (Mode: {mode}, Val: {val})...")
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].compute_projection_matrix(mode=mode, val=val)
    def apply_gpm_to_grads(self, scale=1.0):
        """
        Thực hiện chiếu trực giao gradient cho mu và sigma.
        """
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].apply_gradient_projection(scale=scale)
    def forward_with_ib(self, x):
        """
        [FIXED] Forward với IB, thêm logic lấy [CLS] token cho ViT.
        """
        kl_losses = []
        
        # 1. Embeddings
        x = self.backbone.patch_embed(x)
        if hasattr(self.backbone, '_pos_embed'):
            x = self.backbone._pos_embed(x)
        else:
            if self.backbone.pos_embed is not None:
                x = x + self.backbone.pos_embed
            x = self.backbone.pos_drop(x)

        # 2. Blocks + Noise
        for i, block in enumerate(self.backbone.blocks):
            x = block(x) 
            if hasattr(self.backbone, 'noise_maker'):
                x, kl = self.backbone.noise_maker[i](x, return_kl=True)
                kl_losses.append(kl)
        
        # 3. Norm
        if hasattr(self.backbone, 'norm'):
            x = self.backbone.norm(x)

        # [CRITICAL FIX]: Lấy [CLS] Token (Index 0)
        # Nếu output là [Batch, 197, Dim] thì chỉ lấy [Batch, Dim]
        if x.dim() == 3:
            x = x[:, 0]

        # 4. Classifier
        x = self.buffer(x.to(self.buffer.weight.dtype))
        x = x.to(self.normal_fc.weight.dtype)
        logits = self.normal_fc(x)['logits']
        
        total_kl = torch.sum(torch.stack(kl_losses)) if kl_losses else torch.tensor(0.0, device=self.device)
        
        return logits, total_kl

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor, chunk_size=2048) -> None:
        with autocast(enabled=False):
            X = X.float().to(self.device)
            Y = Y.float().to(self.device)
            num_targets = Y.shape[1]
            
            if self.weight.shape[1] == 0:
                # [FIXED] Tính dummy feature cũng phải chuẩn [CLS] token
                dummy_feat = self.backbone(X[0:2]).float()
                dummy_feat = self.buffer(dummy_feat)
                feat_dim = dummy_feat.shape[1]
                self.weight = torch.zeros((feat_dim, num_targets), device=self.device, dtype=torch.float32)
            elif num_targets > self.weight.shape[1]:
                increment = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment), device=self.device, dtype=torch.float32)
                self.weight = torch.cat((self.weight, tail), dim=1)

            N = X.shape[0]
            feat_dim = self.weight.shape[0]
            A = torch.zeros((feat_dim, feat_dim), device=self.device, dtype=torch.float32)
            B = torch.zeros((feat_dim, num_targets), device=self.device, dtype=torch.float32)
            
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                x_batch = X[start:end] 
                y_batch = Y[start:end] 
                
                features = self.backbone(x_batch).float()
                features = self.buffer(features)
                
                A += features.T @ features
                B += features.T @ y_batch
                del features, x_batch, y_batch 

            I = torch.eye(feat_dim, device=self.device, dtype=torch.float32)
            A += self.gamma * I 

            try:
                W_solution = torch.linalg.solve(A, B)
            except RuntimeError:
                W_solution = torch.linalg.pinv(A) @ B
            
            self.weight = W_solution
            del A, B, I, X, Y
            torch.cuda.empty_cache()
    # =========================================================================
    # [ANALYTIC LEARNING (OPTIMIZED FIT)]
    # =========================================================================

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor, chunk_size=2048) -> None:
        """
        Tối ưu hóa Analytic Learning (Chunking RLS):
        1. Sử dụng Accumulation (Cộng dồn) để tránh OOM khi tính X^T * X trên dataset lớn.
        2. Sử dụng torch.linalg.solve thay vì torch.inverse (Nhanh hơn & Ổn định hơn).
        """
        # [QUAN TRỌNG] Tắt Mixed Precision để đảm bảo độ chính xác ma trận
        with autocast(enabled=False):
            
            # 1. Chuẩn bị dữ liệu (Float32)
            X = X.float().to(self.device)
            Y = Y.float().to(self.device)
            
            # 2. Mở rộng Classifier nếu có class mới
            num_targets = Y.shape[1]
            
            # Nếu chưa có weight (lần đầu fit), khởi tạo
            if self.weight.shape[1] == 0:
                # Tạm tính feature dim sau khi qua buffer
                dummy_feat = self.backbone(X[0:2]).float()
                dummy_feat = self.buffer(dummy_feat)
                feat_dim = dummy_feat.shape[1]
                
                self.weight = torch.zeros((feat_dim, num_targets), device=self.device, dtype=torch.float32)
                
            elif num_targets > self.weight.shape[1]:
                # Mở rộng weight cũ (Padding 0 cho class mới)
                increment = num_targets - self.weight.shape[1]
                tail = torch.zeros((self.weight.shape[0], increment), device=self.device, dtype=torch.float32)
                self.weight = torch.cat((self.weight, tail), dim=1)

            # 3. Tính toán Ma trận A (Autocorrelation) và B (Cross-correlation)
            # A = X^T * X + lambda * I
            # B = X^T * Y
            
            N = X.shape[0]
            feat_dim = self.weight.shape[0]
            
            A = torch.zeros((feat_dim, feat_dim), device=self.device, dtype=torch.float32)
            B = torch.zeros((feat_dim, num_targets), device=self.device, dtype=torch.float32)
            
            # Kỹ thuật Chunking: Duyệt qua từng batch nhỏ
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                
                # Lấy raw images
                x_batch = X[start:end] 
                y_batch = Y[start:end] 
                
                # Extract features qua backbone + buffer
                features = self.backbone(x_batch).float()
                features = self.buffer(features) # Qua Random Projection
                
                # Cộng dồn
                A += features.T @ features
                B += features.T @ y_batch
                
                del features, x_batch, y_batch 

            # 4. Áp dụng Regularization (Ridge)
            I = torch.eye(feat_dim, device=self.device, dtype=torch.float32)
            A += self.gamma * I 

            # 5. Giải hệ phương trình tuyến tính A * W = B
            try:
                # linalg.solve tự động chọn thuật toán (Cholesky/LU) tối ưu
                W_solution = torch.linalg.solve(A, B)
            except RuntimeError:
                # Fallback dùng Pseudo-Inverse nếu ma trận suy biến
                W_solution = torch.linalg.pinv(A) @ B
            
            # 6. Cập nhật Weight
            self.weight = W_solution
            
            del A, B, I, X, Y
            torch.cuda.empty_cache()