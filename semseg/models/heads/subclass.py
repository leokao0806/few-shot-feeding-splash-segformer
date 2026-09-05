import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F


class MeanFieldInference(nn.Module):
    def __init__(self, embed_dim, total_subclass, kernel_size, loop, Potts=True):
        super().__init__()
        self.kernel_size  = kernel_size
        self.total_subclass = total_subclass
        self.loop      = loop
        self.Potts     = Potts
        self.w2 = nn.Parameter(torch.ones(1)) # weight for pairwise potientials
        self.mu = nn.Parameter(torch.ones((total_subclass,total_subclass))) # compatibility
        if self.Potts:
            self.mu.requires_grad= False
            for i in range(total_subclass):
                self.mu[i,i] = 0
        else:
            self.w2.requires_grad= False

        self.query = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.key  = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)

    def form_local(self,x):
        batch_size, embed_dim, H, W = x.size()
        return nn.functional.unfold(x,self.kernel_size,stride=1,padding=self.kernel_size//2).reshape(batch_size,embed_dim,self.kernel_size,self.kernel_size,H,W)

    def forward(self, u_potential, embed_feature): # batch_size, embed_dim, H, W
        batch_size, embed_dim, image_height, image_width = embed_feature.size()
        V      = F.softmax(u_potential,dim=1)
        if self.loop > 0:
            Q = self.query(embed_feature).unsqueeze(2).unsqueeze(2)
            K = self.form_local(self.key(embed_feature))
            A = torch.reshape(torch.sum(Q*K,dim=1,keepdim=True)/embed_dim**0.5,(batch_size,-1,image_height,image_width))
            A[:,self.kernel_size**2//2,:,:]=torch.tensor(float('-inf')) #
            pairwise_weight = F.softmax(A,dim=1).view(batch_size,1,self.kernel_size,self.kernel_size,image_height,image_width)
            for _ in range(self.loop):
                pairwise_potential = torch.sum(self.form_local(V)*pairwise_weight,dim=(2,3))
                V = F.softmax(u_potential - torch.einsum('bshw,sq->bqhw',pairwise_potential,F.relu(self.w2)*F.relu(self.mu)),dim=1)

        return V

class SubclassBlock0(nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, loop):
        super().__init__()
        self.total_subclass = total_subclass+(total_subclass % 2)
        self.conv   = nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.loop = loop
        if loop > 0:
            self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)

    def forward(self,embedding_feature_map, subclass_weight=None, output_subclass=False): # B,C,H,W subclass_weight 2,subclass
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
            subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
            subclass = F.softmax(out,dim=1)
        if subclass_weight is None:
            result = torch.sum(torch.reshape(subclass,(subclass.shape[0],2,self.total_subclass//2,subclass.shape[-2],subclass.shape[-1])),dim=2)
            if output_subclass:
                return result, subclass
            else:
                return result
        else:
            result = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
            if output_subclass:
                return result, subclass
            else:
                return result

class SubclassBlock1(nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.conv   = nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.conv_w = nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0)
        self.loop = loop
        if loop > 0:
            self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)

    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
            subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
            subclass = F.softmax(out,dim=1)
        if subclass_weight is None:
            positive_w= F.sigmoid(self.conv_w(torch.cat([F.adaptive_max_pool2d(subclass,(1,1)),F.adaptive_avg_pool2d(subclass,(1,1))],dim=1)))
            positive = torch.sum(positive_w*subclass,dim=1,keepdim=True)
            negative = 1-positive
            if output_subclass:
                return torch.cat([negative,positive],dim=1), subclass
            else:
                return torch.cat([negative,positive],dim=1)
        else:
            result = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
            if output_subclass:
                return result, subclass
            else:
                return result

class SubclassBlock2(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.conv_w = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0)
        self.conv_w2 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.conv_w3 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)
    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(subclass,(1,1)),F.adaptive_avg_pool2d(subclass,(1,1))],dim=1)
          positive_w= F.sigmoid(self.conv_w3(global_pool)*self.conv_w2(global_pool)+self.conv_w(global_pool))
          positive = torch.sum(positive_w*subclass,dim=1,keepdim=True)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          result = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          if output_subclass:
            return result, subclass
          else:
            return result

class SubclassBlock3(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.conv_w1 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0)
        self.conv_w2 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)
    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(subclass,(1,1)),F.adaptive_avg_pool2d(subclass,(1,1))],dim=1)
          positive_w= F.sigmoid(self.conv_w1(global_pool)*(self.conv_w2(global_pool)+1))
          positive = torch.sum(positive_w*subclass,dim=1,keepdim=True)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          result = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          if output_subclass:
            return result, subclass
          else:
            return result

# class SubclassBlock2L2(torch.nn.Module):
#     def __init__(self, in_channels, total_subclass, kernel_size, loop):
#         super().__init__()
#         self.total_subclass = total_subclass
#         self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
#         self.conv_w = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0)
#         self.conv_w2 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
#         self.conv_w3 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
#         self.varc  = torch.nn.Parameter(torch.zeros(1))
#         self.loop = loop
#         if loop > 0:
#           self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)
#     def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
#         out   = self.conv(embedding_feature_map) # B,S,H,W
#         if self.loop > 0:
#           subclass = self.mean_field_inference(out,embedding_feature_map)
#         else:
#           subclass = F.softmax(out,dim=1)
#         subclass = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))
#         if subclass_weight is None:
#           global_pool = torch.cat([F.adaptive_max_pool2d(subclass,(1,1)),F.adaptive_avg_pool2d(subclass,(1,1))],dim=1)
#           positive_w= self.conv_w3(global_pool)*self.conv_w2(global_pool)+self.conv_w(global_pool)
#           positive = F.sigmoid(torch.sum(positive_w*subclass,dim=1,keepdim=True)+self.varc)
#           negative = 1-positive
#           if output_subclass:
#             return torch.cat([negative,positive],dim=1), subclass
#           else:
#             return torch.cat([negative,positive],dim=1)
#         else:
#           result = F.softmax(torch.einsum('bshw,cs->bchw',subclass,subclass_weight),dim=1)
#           if output_subclass:
#             return result, subclass
#           else:
#             return result

class SubclassBlock2L2(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.conv_w = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0)
        self.conv_w2 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.conv_w3 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)
    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
          # subclass = F.relu(out)
        # subclass = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))
        subclass = F.normalize(subclass, p=2, dim=1)
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(subclass,(1,1)),F.adaptive_avg_pool2d(subclass,(1,1))],dim=1)
          positive_w= self.conv_w3(global_pool)*self.conv_w2(global_pool)+self.conv_w(global_pool)
          positive = F.sigmoid(torch.sum(positive_w*subclass,dim=1,keepdim=True)+self.varc)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          X[:,0,:,:]=torch.sqrt((1-X[:,1,:,:]**2).clamp_(0,1))
          result = X  # result = F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass
          else:
            return result

class SubclassBlock2L2_Adaptive_WU(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, total_field, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.in_channels = in_channels
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.Wk    = torch.nn.Parameter(torch.empty(total_field,in_channels*2))
        torch.nn.init.xavier_uniform_(self.Wk, gain=1.5)
        self.Vb    = torch.nn.Parameter(torch.empty(total_field,total_subclass)) #fs
        torch.nn.init.xavier_uniform_(self.Vb, gain=1.5)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)

    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        # L2
        out = embedding_feature_map / torch.sqrt(torch.sum(embedding_feature_map ** 2, dim=1, keepdim=True))
        # conv
        out   = self.conv(out) # (B, 64, H, W)
        # Softmax
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
          # subclass = F.softmax(out/0.5,dim=1)
          # subclass = torch.sigmoid(out)

        # subclass = subclass / (subclass.max(dim=1, keepdim=True)[0] + 1e-6)
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(embedding_feature_map,(1,1)),F.adaptive_avg_pool2d(embedding_feature_map,(1,1))],dim=1)
          w0 = F.softmax(torch.einsum('bchw,fc->bfhw',global_pool,self.Wk)/(2*self.in_channels)**0.5,dim=1)
          V_weight = self.Vb #F.softmax(self.Vb,axis=1) #fs
          positive_w= torch.einsum('bfhw,fs->bshw',w0,V_weight)
          positive = F.sigmoid(torch.sum(positive_w*subclass,dim=1,keepdim=True)+self.varc)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          X = F.relu(X)  # Ensure non-negative values
          # X[:,0,:,:]=torch.sqrt((1-X[:,1,:,:]**2).clamp_(0,1))
          X[:,0,:,:]=torch.sqrt(1-X[:,1,:,:]**2)
          result = X  #F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass
          else:
            return result
"""
2026/07/30
"""
class SubclassBlock_Unified(nn.Module):
    def __init__(self, in_channels, total_subclass, total_field, num_prompts=1):
        super().__init__()
        self.total_subclass = total_subclass
        self.in_channels = in_channels
        self.num_prompts = num_prompts

        # 1. 基礎子類別卷積
        self.conv = nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        # 2. 環境/場景適應參數 (學長 MIL 框架：Wk=Key, Vb=Value)
        self.Wk = nn.Parameter(torch.empty(total_field, in_channels * 2))
        nn.init.xavier_uniform_(self.Wk, gain=1.5)
        self.Vb = nn.Parameter(torch.empty(total_field, total_subclass))
        nn.init.xavier_uniform_(self.Vb, gain=1.5)

        self.varc = nn.Parameter(torch.zeros(1))

        # 3. Q-prime 內部提示參數 (文字提示/Graph RAG 的學理實現)
        self.q_prime = nn.Parameter(torch.zeros(num_prompts, in_channels * 2))
        nn.init.normal_(self.q_prime, std=0.02)
        # ⬇️ 新增：老師提議的 gamma 參數，初始值設為 0.5 (代表原本的一半一半)
        self.gamma = nn.Parameter(torch.tensor([0.5]))

    def forward(self, embedding_feature_map, subclass_weight=None, q_prime=False, output_subclass=True):
        """
        整合三種物理意義路徑：
        1. subclass_weight 不為 None -> 學長 Explicit MIL (點/袋提示)
        2. q_prime 為 True -> 新 Q-prime 模式 (內建文字提示/意圖對齊)
        3. 皆無 -> 學長 Context Attention (自動環境感知)
        """
        B, C, H, W = embedding_feature_map.shape
        out = self.conv(embedding_feature_map)
        # --- 第一階段：決定 Subclass 空間的正規化邏輯 ---
        if q_prime:
            # Q-prime 模式：保留原始 Logits 強度資訊，用於新的 Loss 設計 (防止逃避、強制學習)
            subclass = F.normalize(out, p=2, dim=1)
        else:
            # MIL/Context 模式：移除 Softmax，僅保留 L2 正規化
            # 物理意義：轉化為餘弦相似度空間，描述「子類別中間態」
            subclass = F.normalize(out, p=2, dim=1)

        # --- 第二階段：根據提示來源計算權重 (Weight Generation) ---

        # [A] MIL Prompt (已優化出的最佳 Vb 組合權重)
        if subclass_weight is not None:
            # 修正：這裡的維度要是 total_subclass (如 64)，而不是 embedding 的通道數 C (如 256)
            num_subclass = subclass.shape[1]
            positive_w = subclass_weight.view(1, num_subclass, 1, 1).to(subclass.device)

        # [B] Q-prime (局部 Patch 模式)
        # 取得影像全域上下文 (Image Context)
        global_pool = torch.cat([
            F.adaptive_max_pool2d(embedding_feature_map, (1, 1)),
            F.adaptive_avg_pool2d(embedding_feature_map, (1, 1))
        ], dim=1)  # (B, 512, 1, 1)

        # [B] Q-prime (整張圖全域模式 - 拔除 Patch 限制)
        if q_prime:
            # 1. 將全域特徵壓成 Token 形式
            q_base = global_pool.view(B, 1, -1)  # (B, 1, 512)

            # 2. 確保 q_prime 正確擴展給所有 Batch，並支援多組 prompt 的狀況
            q_p_batch = self.q_prime.unsqueeze(0).expand(B, -1, -1)  # (B, num_prompts, 512)

            # 3. 在 Token 維度拼接 -> (B, 1 + num_prompts, 512)
            q_concat = torch.cat([q_base, q_p_batch], dim=1)

            # 4. 計算 Attention 與子類別映射
            w0 = F.softmax(torch.einsum('bnc,fc->bnf', q_concat, self.Wk) / (2 * self.in_channels) ** 0.5, dim=2)
            positive_w_concat = torch.einsum('bnf,fs->bns', w0, self.Vb)  # (B, 1 + num_prompts, 64)

            # 5. 抽離「舊特徵」(原圖 context，位於 dim=1 的第 0 個位置)
            w_old = positive_w_concat[:, 0:1, :]  # (B, 1, 64)

            # 6. 抽離「新特徵」(1 ~ 末尾的 prompts) 並做平均濃縮
            w_new = positive_w_concat[:, 1:, :].mean(dim=1, keepdim=True)  # (B, 1, 64)

            # 7. 合併為 1x2 結構，準備與 gamma 進行權重分配
            w_combined = torch.cat([w_old, w_new], dim=1)  # (B, 2, 64)

            # 8. 限制 gamma 範圍在 0 ~ 1 之間
            gamma_clamped = torch.clamp(self.gamma, min=0.0, max=1.0)

            # 9. 建立權重張量 [gamma (舊特徵權重), 1 - gamma (新特徵權重)]
            gamma_weight = torch.stack([gamma_clamped, 1.0 - gamma_clamped]).view(1, 2, 1)

            # 10. 加權求和並壓回 (B, 64, 1, 1) 以符合後續特徵圖計算維度
            positive_w_flat = (w_combined * gamma_weight).sum(dim=1)  # (B, 64)
            positive_w = positive_w_flat.view(B, -1, 1, 1)  # (B, 64, 1, 1)

        # [C] Context Attention (無任何提示時的自動模式)
        elif subclass_weight is None: # 加入 elif 避免變數覆蓋
            # 依影像全域特徵尋找對應的子類別組合 (w0 維度: B, 5, 1, 1)
            w0 = F.softmax(torch.einsum('bchw,fc->bfhw', global_pool, self.Wk) / (2 * self.in_channels) ** 0.5, dim=1)
            # 從領域知識庫 Vb 提取權重 (B, 64, 1, 1)
            positive_w = torch.einsum('bfhw,fs->bshw', w0, self.Vb)

        # --- 第三階段：合成最終預測與機率 ---
        # 融合提示權重與子類別圖層
        logits = torch.sum(positive_w * subclass, dim=1, keepdim=True) + self.varc
        positive = torch.sigmoid(logits)
        negative = 1 - positive
        result = torch.cat([negative, positive], dim=1)

        return (result, subclass) if output_subclass else result


"""
2026/03/27
"""
class SubclassBlock2L2_Adaptive_W_visual_prompt(nn.Module):
    def __init__(self, in_channels, total_subclass, total_field, num_prompts=1):
        super().__init__()
        self.total_subclass = total_subclass
        self.in_channels = in_channels
        self.num_prompts = num_prompts  # 新增：記錄提示數量

        # 1. 基礎子類別卷積 (產生 subclass 概率圖)
        self.conv = nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)

        # 2. 子空間定義參數 (Wk 為 Key, Vb 為 Value)
        # Wk 預期維度為 in_channels * 2 (即 512)
        self.Wk = nn.Parameter(torch.empty(total_field, in_channels * 2))
        nn.init.xavier_uniform_(self.Wk, gain=1.5)
        self.Vb = nn.Parameter(torch.empty(total_field, total_subclass))
        nn.init.xavier_uniform_(self.Vb, gain=1.5)

        self.varc = nn.Parameter(torch.zeros(1))

        # 3. 宣告獨立自由變數 q_prime
        self.q_prime = nn.Parameter(torch.zeros(num_prompts, in_channels * 2))
        nn.init.normal_(self.q_prime, std=0.02)

    # 【新增 prompt_enabled 參數，預設為 False】
    def forward(self, embedding_feature_map, output_subclass=False, prompt_enabled=False):
        B, C, H, W = embedding_feature_map.shape

        # --- 產生 subclass 機率圖 (與原版完全一致) ---
        out = self.conv(embedding_feature_map)
        subclass = F.softmax(out, dim=1)
        subclass = F.normalize(subclass, p=2, dim=1)  # 安全的 L2 正規化

        # --- 取得原始 Q (Image Context) ---
        global_pool = torch.cat([
            F.adaptive_max_pool2d(embedding_feature_map, (1, 1)),
            F.adaptive_avg_pool2d(embedding_feature_map, (1, 1))
        ], dim=1)  # (B, 512, 1, 1)

        if not prompt_enabled:
            # 直接使用 global_pool 計算 Attention，避免 Softmax(0) 稀釋問題
            w0 = F.softmax(torch.einsum('bchw,fc->bfhw', global_pool, self.Wk) / (2 * self.in_channels) ** 0.5, dim=1)
            positive_w = torch.einsum('bfhw,fs->bshw', w0, self.Vb)  # (B, 64, 1, 1)

        else:
            # 【開啟提示】
            q_base = global_pool.view(B, 1, -1)  # (B, 1, 512)
            q_p_batch = self.q_prime.repeat(B, 1, 1)  # (B, 1, 512)
            # 在 Token 維度拼接 -> (B, 2, 512)
            q_concat = torch.cat([q_base, q_p_batch], dim=1)
            # 計算 Attention: (B, 2, 5)
            w0 = F.softmax(torch.einsum('bnc,fc->bnf', q_concat, self.Wk) / (2 * self.in_channels) ** 0.5, dim=2)
            # 生成子類別權重: (B, 2, 64)
            positive_w_concat = torch.einsum('bnf,fs->bns', w0, self.Vb)
            # 融合兩個 Token 的預測，壓回 (B, 64, 1, 1) 以符合後續計算維度
            positive_w = positive_w_concat.mean(dim=1).view(B, -1, 1, 1)
        # ==========================================

        # --- 計算最終預測 (與原版完全一致) ---
        logits = torch.sum(positive_w * subclass, dim=1, keepdim=True) + self.varc
        positive = torch.sigmoid(logits)
        negative = 1 - positive
        result = torch.cat([negative, positive], dim=1)
        return (result, subclass) if output_subclass else result
"""
origin version
"""
class SubclassBlock2L2_Adaptive_W(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, total_field, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.in_channels = in_channels
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.Wk    = torch.nn.Parameter(torch.empty(total_field,in_channels*2))
        torch.nn.init.xavier_uniform_(self.Wk, gain=1.5)
        self.Vb    = torch.nn.Parameter(torch.empty(total_field,total_subclass)) #fs
        torch.nn.init.xavier_uniform_(self.Vb, gain=1.5)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)

    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # (B, 64, H, W)
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
          # subclass = F.softmax(out/0.5,dim=1)
          # subclass = torch.sigmoid(out)
        subclass = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))
        # subclass = subclass / (subclass.max(dim=1, keepdim=True)[0] + 1e-6)
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(embedding_feature_map,(1,1)),F.adaptive_avg_pool2d(embedding_feature_map,(1,1))],dim=1)
          w0 = F.softmax(torch.einsum('bchw,fc->bfhw',global_pool,self.Wk)/(2*self.in_channels)**0.5,dim=1)
          V_weight = self.Vb #F.softmax(self.Vb,axis=1) #fs
          positive_w= torch.einsum('bfhw,fs->bshw',w0,V_weight)
          positive = F.sigmoid(torch.sum(positive_w*subclass,dim=1,keepdim=True)+self.varc)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          X = F.relu(X)  # Ensure non-negative values
          # X[:,0,:,:]=torch.sqrt((1-X[:,1,:,:]**2).clamp_(0,1))
          X[:,0,:,:]=torch.sqrt(1-X[:,1,:,:]**2)
          result = X  #F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass
          else:
            return result



class SubclassBlock2L2_Adaptive_WA(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, total_field, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.in_channels = in_channels
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.Wk    = torch.nn.Parameter(torch.empty(total_field,in_channels*2))
        torch.nn.init.xavier_uniform_(self.Wk, gain=1.5)
        self.Vb    = torch.nn.Parameter(torch.empty(total_field,total_subclass)) #fs
        torch.nn.init.xavier_uniform_(self.Vb, gain=1.5)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.total_field = total_field
        self.activte_field = 1
        # self.field_mask = torch.nn.Parameter(torch.zeros(total_field)) # mask for field
        self.register_buffer('field_mask', torch.zeros(total_field))  # Not a parameter
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)

    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
        subclass = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))+1e-6
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(embedding_feature_map,(1,1)),F.adaptive_avg_pool2d(embedding_feature_map,(1,1))],dim=1)
          # w0 = F.softmax(torch.einsum('bchw,fc->bfhw',global_pool,self.Wk)/(2*self.in_channels)**0.5,dim=1)
          with torch.no_grad():
            new_mask = torch.zeros_like(self.field_mask)
            new_mask[:self.activte_field] = 1
            self.field_mask.copy_(new_mask)
          mask = self.field_mask
          wk = self.Wk * mask.view(self.total_field,1)
          w0 = F.softmax(torch.einsum('bchw,fc->bfhw',global_pool,wk)/(2*self.in_channels)**0.5,dim=1)

          # V_weight = self.Vb #F.softmax(self.Vb,axis=1) #fs
          V_weight = self.Vb * mask.view(self.total_field,1) #F.softmax(self.Vb,axis=1) #fs
          positive_w= torch.einsum('bfhw,fs->bshw',w0,V_weight)
          logits = torch.sum(positive_w * subclass, dim=1, keepdim=True) + self.varc
          positive = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1 - 1e-4)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          # X[:,0,:,:]=torch.sqrt((1-X[:,1,:,:]**2).clamp_(0,1))
          X[:,0,:,:]=torch.sqrt(1-X[:,1,:,:]**2)
          result = X  #F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass
          else:
            return result

class SubclassBlock2L2_Adaptive_W2(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, total_field, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.in_channels = in_channels
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.Wk    = torch.nn.Parameter(torch.empty(total_field,in_channels*2))
        torch.nn.init.xavier_uniform_(self.Wk, gain=1.1)
        self.Vb    = torch.nn.Parameter(torch.empty(total_field,total_subclass)) #fs
        torch.nn.init.xavier_uniform_(self.Vb, gain=1.1)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)

    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
        subclass = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))
        # subclass = F.normalize(subclass, p=2, dim=1)
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(embedding_feature_map,(1,1)),F.adaptive_avg_pool2d(embedding_feature_map,(1,1))],dim=1)
          w0 = F.softmax(torch.einsum('bchw,fc->bfhw',global_pool,self.Wk)/(2*self.in_channels)**0.5,dim=1)
          V_weight = self.Vb #F.softmax(self.Vb,axis=1) #fs
          positive_w= torch.einsum('bfhw,fs->bshw',w0,V_weight)
          positive_w = positive_w/torch.sqrt(torch.sum(positive_w**2,dim=1,keepdim=True))
          # positive_w = F.normalize(positive_w, p=2, dim=1)
          positive = torch.sum(positive_w*subclass,dim=1,keepdim=True)
          # positive = torch.sum(positive_w*subclass,dim=1,keepdim=True).clamp_(0,1)
          # positive = F.sigmoid(torch.sum(positive_w*subclass,dim=1,keepdim=True)+self.varc)
          negative = torch.sqrt((1-positive**2))
          # negative = torch.sqrt((1-positive**2).clamp_(0,1))
          # negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          # X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight).clamp_(0,1)
          # X[:,0,:,:]=torch.sqrt((1-X[:,1,:,:]**2).clamp_(0,1))
          X[:,0,:,:]=torch.sqrt(1-X[:,1,:,:]**2)
          result = X  #F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass
          else:
            return result

class SubclassBlock2L2_Adaptive_W3(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, total_field, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.in_channels = in_channels
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.Wk    = torch.nn.Parameter(torch.empty(total_field,in_channels*2))
        torch.nn.init.xavier_uniform_(self.Wk, gain=1.1)
        self.Vb    = torch.nn.Parameter(torch.empty(total_field,total_subclass)) #fs
        torch.nn.init.xavier_uniform_(self.Vb, gain=1.1)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)

    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
        subclass = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))
        # subclass = F.normalize(subclass, p=2, dim=1)
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(embedding_feature_map,(1,1)),F.adaptive_avg_pool2d(embedding_feature_map,(1,1))],dim=1)
          w0 = F.softmax(torch.einsum('bchw,fc->bfhw',global_pool,self.Wk)/(2*self.in_channels)**0.5,dim=1)
          V_weight = self.Vb #F.softmax(self.Vb,axis=1) #fs
          positive_w= torch.einsum('bfhw,fs->bshw',w0,V_weight)
          positive_w = positive_w/torch.sqrt(torch.sum(positive_w**2,dim=1,keepdim=True))
          # positive_w = F.normalize(positive_w, p=2, dim=1)
          positive = torch.sum(positive_w*subclass,dim=1,keepdim=True)
          positive = F.relu(positive)
          # positive = torch.sum(positive_w*subclass,dim=1,keepdim=True).clamp_(0,1)
          # positive = F.sigmoid(torch.sum(positive_w*subclass,dim=1,keepdim=True)+self.varc)
          negative = torch.sqrt((1-positive**2))
          # negative = torch.sqrt((1-positive**2).clamp_(0,1))
          # negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          # X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight).clamp_(0,1)
          # X[:,0,:,:]=torch.sqrt((1-X[:,1,:,:]**2).clamp_(0,1))
          X[:,0,:,:]=torch.sqrt(1-X[:,1,:,:]**2)
          result = X  #F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass
          else:
            return result

class SubclassBlock2L2_C_T(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.conv_w = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0)
        self.conv_w2 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.conv_w3 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.conv_fuse = torch.nn.Conv2d(2, 1, kernel_size=1, padding=0,bias=False)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)
    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
          # subclass = F.relu(out)
        # subclass = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))
        subclass = F.normalize(subclass, p=2, dim=1)
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(subclass,(1,1)),F.adaptive_avg_pool2d(subclass,(1,1))],dim=1)
          positive_w= self.conv_w3(global_pool)*self.conv_w2(global_pool)+self.conv_w(global_pool)
          #===
          subclass_prob = positive_w * subclass
          subclass_argmax = subclass.argmax(dim=1, keepdim=True).float() # (b,1,h,w)
          subclass_argmax_unit = F.normalize(subclass_argmax, p=2, dim=1)  # torch.Size([b, 1, h, w])
          subclass_prob_sum = torch.sum(subclass_prob,dim=1,keepdim=True)  # torch.Size([b, 1, h, w])
          positive = F.sigmoid(self.conv_fuse(torch.cat([subclass_prob_sum,subclass_argmax_unit],dim=1))+self.varc)
          #===
          # positive = F.sigmoid(torch.sum(positive_w*subclass,dim=1,keepdim=True)+self.varc)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          X[:,0,:,:]=torch.sqrt((1-X[:,1,:,:]**2).clamp_(0,1))
          result = X  # result = F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass
          else:
            return result

class SubclassBlock2L2_Adaptive_W_T(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, total_field, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.in_channels = in_channels
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.Wk    = torch.nn.Parameter(torch.empty(total_field,in_channels*2))
        torch.nn.init.xavier_uniform_(self.Wk, gain=0.1)
        self.Vb    = torch.nn.Parameter(torch.empty(total_field,total_subclass)) #fs
        torch.nn.init.xavier_uniform_(self.Vb, gain=0.1)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)

    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          ori_subclass = F.softmax(out,dim=1)
        subclass = ori_subclass/torch.sqrt(torch.sum(ori_subclass**2,dim=1,keepdim=True))
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(embedding_feature_map,(1,1)),F.adaptive_avg_pool2d(embedding_feature_map,(1,1))],dim=1)
          w0 = F.softmax(torch.einsum('bchw,fc->bfhw',global_pool,self.Wk)/(2*self.in_channels)**0.5,dim=1)
          V_weight = self.Vb #F.softmax(self.Vb,axis=1) #fs
          # batch_subclass = F.adaptive_avg_pool2d(out, (1,1)) #bs11 #out, ori_subclass
          batch_subclass = F.adaptive_max_pool2d(ori_subclass, (1,1)) #bs11 #out, ori_subclass
          subclass_field = torch.einsum('bshw,fs->bsf',batch_subclass,V_weight)
          positive_w= torch.einsum('bfhw,bsf->bshw',w0,subclass_field)
          # positive_w= torch.einsum('bfhw,fs->bshw',w0,V_weight)
          positive = F.sigmoid(torch.sum(positive_w*subclass,dim=1,keepdim=True)+self.varc)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass,subclass_weight)
          X[:,0,:,:]=torch.sqrt(1-X[:,1,:,:]**2)
          result = X  #F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass
          else:
            return result

class CombinedBlock(torch.nn.Module):
    def __init__(self, total_subclass, num_field=32):
      super().__init__()
      self.field_embedding = torch.nn.Embedding(total_subclass, num_field)
      self.linear = torch.nn.Linear(num_field, 3)
      self.cross_attn1 = torch.nn.MultiheadAttention(embed_dim=total_subclass, num_heads=2, batch_first=True)
      self.linear1 = torch.nn.Linear(total_subclass, total_subclass)
      self.cross_attn2 = torch.nn.MultiheadAttention(embed_dim=total_subclass, num_heads=2, batch_first=True)
      self.linear2 = torch.nn.Linear(total_subclass, total_subclass)
      self.linear3 = torch.nn.Linear(3, 1)
      self.activation = F.gelu

    def forward(self, semantic_description): # B,C,H,W
      field_embed = self.activation(self.linear(self.field_embedding.weight)).transpose_(0,1)
      field_embed = F.layer_norm(field_embed, field_embed.size()[1:])
      semantic_to_field, semantic_to_field_w = self.cross_attn1(query=semantic_description, key=field_embed, value=field_embed)
      semantic_to_field = self.activation(self.linear1(semantic_to_field))
      field_to_semantic, field_to_semantic_w = self.cross_attn2(query=field_embed, key=semantic_to_field, value=semantic_to_field)
      field_semantic = self.activation(self.linear2(field_to_semantic))
      field_semantic = F.layer_norm(field_semantic, field_semantic.size()[1:])
      field_semantic_w = self.linear3(field_semantic.transpose_(0,1))

      return field_semantic_w

class SubclassBlock2L2_F(torch.nn.Module):
    def __init__(self, in_channels, total_subclass, kernel_size, loop):
        super().__init__()
        self.total_subclass = total_subclass
        self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
        self.conv_w = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0)
        self.conv_w2 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.conv_w3 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
        self.varc  = torch.nn.Parameter(torch.zeros(1))
        self.combine = CombinedBlock(total_subclass)
        self.loop = loop
        if loop > 0:
          self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)
    def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
        out   = self.conv(embedding_feature_map) # B,S,H,W
        if self.loop > 0:
          subclass = self.mean_field_inference(out,embedding_feature_map)
        else:
          subclass = F.softmax(out,dim=1)
        
        subclass_unit = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))
        if subclass_weight is None:
          global_pool = torch.cat([F.adaptive_max_pool2d(subclass_unit,(1,1)),F.adaptive_avg_pool2d(subclass_unit,(1,1))],dim=1)
          global_pool_L1 = torch.cat([F.adaptive_max_pool2d(subclass,(1,1)),F.adaptive_avg_pool2d(subclass,(1,1))],dim=1)

          positive_w= self.conv_w3(global_pool)*self.conv_w2(global_pool)+self.conv_w(global_pool)
          positive_sb1= self.conv_w3(global_pool_L1)*self.conv_w2(global_pool_L1)+self.conv_w(global_pool_L1)
          positive_sb3= self.conv_w3(global_pool_L1)*(self.conv_w2(global_pool_L1)+1)

          semantic_query_v = torch.cat([positive_w,positive_sb1,positive_sb3],dim=0).squeeze(-1).squeeze(-1)
          # semantic_query_w = F.sigmoid(semantic_query_v)
          # semantic_query = torch.cat([semantic_query_v,semantic_query_w],dim=0)
          field_semantic_w = self.combine(semantic_query_v)
          field_semantic_w = field_semantic_w.unsqueeze(0).unsqueeze(-1)

          positive = F.sigmoid(torch.sum(field_semantic_w*subclass_unit,dim=1,keepdim=True)+self.varc)
          negative = 1-positive
          if output_subclass:
            return torch.cat([negative,positive],dim=1), subclass_unit
          else:
            return torch.cat([negative,positive],dim=1)
        else:
          X   = torch.einsum('bshw,cs->bchw',subclass_unit,subclass_weight)
          X[:,0,:,:]=torch.sqrt(1-X[:,1,:,:]**2)
          result = F.softmax(X,dim=1)
          if output_subclass:
            return result, subclass_unit
          else:
            return result

# class CombinedBlock(torch.nn.Module):
#     def __init__(self, total_subclass, num_field=32):
#       super().__init__()
#       self.field_embedding = torch.nn.Embedding(total_subclass, num_field)
#       self.linear = torch.nn.Linear(num_field, 3)
#       self.cross_attn1 = torch.nn.MultiheadAttention(embed_dim=total_subclass, num_heads=2, batch_first=True)
#       self.linear1 = torch.nn.Linear(total_subclass, total_subclass)
#       self.cross_attn2 = torch.nn.MultiheadAttention(embed_dim=total_subclass, num_heads=2, batch_first=True)
#       self.linear2 = torch.nn.Linear(total_subclass, total_subclass)
#       self.linear3 = torch.nn.Linear(3, 2)
#       self.activation = F.gelu

#     def forward(self, semantic_description):
#       field_embed = self.activation(self.linear(self.field_embedding.weight)).transpose_(0,1)
#       field_embed = F.layer_norm(field_embed, field_embed.size()[1:])
#       semantic_to_field, semantic_to_field_w = self.cross_attn1(query=semantic_description, key=field_embed, value=field_embed)
#       semantic_to_field = self.activation(self.linear1(semantic_to_field))
#       field_to_semantic, field_to_semantic_w = self.cross_attn2(query=field_embed, key=semantic_to_field, value=semantic_to_field)
#       field_semantic = self.activation(self.linear2(field_to_semantic))
#       field_semantic = F.layer_norm(field_semantic, field_semantic.size()[1:])
#       field_semantic_w = self.linear3(field_semantic.transpose_(0,1))
#       # positive_w = field_semantic_w[:,1]
#       # negative_w = field_semantic_w[:,0]
#       return field_semantic_w

# class SubclassBlock2L2_F(torch.nn.Module):
#     def __init__(self, in_channels, total_subclass, kernel_size, loop):
#         super().__init__()
#         self.total_subclass = total_subclass
#         self.conv   = torch.nn.Conv2d(in_channels, total_subclass, kernel_size=1, padding=0)
#         self.conv_w = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0)
#         self.conv_w2 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
#         self.conv_w3 = torch.nn.Conv2d(2*total_subclass, total_subclass, kernel_size=1, padding=0,bias=False)
#         self.varc  = torch.nn.Parameter(torch.zeros(1))
#         self.combine = CombinedBlock(total_subclass)        
#         self.loop = loop
#         if loop > 0:
#           self.mean_field_inference = MeanFieldInference(in_channels, total_subclass, kernel_size, loop)
#     def forward(self,embedding_feature_map,subclass_weight=None, output_subclass=False): # B,C,H,W
#         out   = self.conv(embedding_feature_map) # B,S,H,W
#         if self.loop > 0:
#           subclass = self.mean_field_inference(out,embedding_feature_map)
#         else:
#           subclass = F.softmax(out,dim=1)
        
#         subclass_unit = subclass/torch.sqrt(torch.sum(subclass**2,dim=1,keepdim=True))
#         if subclass_weight is None:
#           global_pool = torch.cat([F.adaptive_max_pool2d(subclass_unit,(1,1)),F.adaptive_avg_pool2d(subclass_unit,(1,1))],dim=1)
#           global_pool_L1 = torch.cat([F.adaptive_max_pool2d(subclass,(1,1)),F.adaptive_avg_pool2d(subclass,(1,1))],dim=1)

#           positive_w= self.conv_w3(global_pool)*self.conv_w2(global_pool)+self.conv_w(global_pool)
#           positive_sb1= self.conv_w3(global_pool_L1)*self.conv_w2(global_pool_L1)+self.conv_w(global_pool_L1)
#           positive_sb3= self.conv_w3(global_pool_L1)*(self.conv_w2(global_pool_L1)+1)
#           semantic_query_v = torch.cat([positive_w,positive_sb1,positive_sb3],dim=0).squeeze(-1).squeeze(-1)
#           # semantic_query_w = F.sigmoid(semantic_query_v)
#           # semantic_query = torch.cat([semantic_query_v,semantic_query_w],dim=0)
#           field_semantic_w = self.combine(semantic_query_v)
#           positive_w = field_semantic_w[:,1]
#           negative_w = field_semantic_w[:,0]
#           positive_w = positive_w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#           negative_w = negative_w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#           positive = F.sigmoid(torch.sum(positive_w*subclass_unit,dim=1,keepdim=True)+self.varc)
#           negative = F.sigmoid(torch.sum(negative_w*subclass_unit,dim=1,keepdim=True)+self.varc)

#           # positive = F.sigmoid(torch.sum(field_semantic_w*subclass_unit,dim=1,keepdim=True)+self.varc)
#           # negative = 1-positive
#           if output_subclass:
#             return torch.cat([negative,positive],dim=1), subclass_unit
#           else:
#             return torch.cat([negative,positive],dim=1)
#         else:
#           X   = torch.einsum('bshw,cs->bchw',subclass_unit,subclass_weight)
#           X[:,0,:,:]=torch.sqrt(1-X[:,1,:,:]**2)
#           result = F.softmax(X,dim=1)
#           if output_subclass:
#             return result, subclass_unit
#           else:
#             return result

# Copy from: semseg/models/heads/segformer.py
class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))



"""
2026/04/07
"""
class SubclassSegFormerHead_Unified(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 2, subclass: int = 64,
                 total_field: int = 5, num_prompts: int = 1) -> None:
        super().__init__()

        # 1. 多尺度特徵投影層 (MLP)
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i + 1}", MLP(dim, embed_dim))

        # 2. 特徵融合與預處理
        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout2d(0.2)

        # 3. 綁定整合了 MIL 與 Q-prime 邏輯的子類別嵌塊
        self.total_field = total_field
        self.subclass_block = SubclassBlock_Unified(
            embed_dim, subclass, total_field, num_prompts=num_prompts
        )

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor],
                subclass_weight=None, q_prime=False, output_subclass=True) -> Tensor:
        """
        參數邏輯對齊：
        - subclass_weight: 若傳入，則走學長 MIL 點/袋提示路徑。
        - q_prime: 若為 True，則走文字提示路徑 (不使用 L2 Normalize)。
        - 兩者皆無: 走自動環境感知路徑。
        """
        # --- 階段 1：特徵融合 (與 SegFormer 結構一致) ---
        B, _, H, W = features[0].shape

        # 使用 getattr 獲取投影層，並將各尺度特徵對齊至 H/4, W/4
        outs = []
        for i, feature in enumerate(features):
            cf_layer = getattr(self, f"linear_c{i + 1}")
            cf = cf_layer(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            if i > 0:  # 尺度大於 1 的特徵需要上採樣
                cf = F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(cf)

        # 融合四層特徵 (B, 1024, H, W) -> (B, 256, H, W)
        seg_features = self.linear_fuse(torch.cat(outs[::-1], dim=1))

        # --- 階段 2：進入子類別提示機制 ---
        # 將融合後的特徵經過 Dropout 後，傳入 subclass_block
        # 這裡會根據 subclass_weight 與 q_prime 的狀態自動切換物理路徑
        seg, subclass = self.subclass_block(
            self.dropout(seg_features),
            subclass_weight=subclass_weight,
            q_prime=q_prime,
            output_subclass=output_subclass
        )

        return seg, subclass

    def get_fusion_map(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """
        提取融合後的特徵圖 (通常用於儲存 5 個場域的特徵，供事後分析或 Graph RAG 使用)
        """
        B, _, H, W = features[0].shape
        outs = []
        for i, feature in enumerate(features):
            cf_layer = getattr(self, f"linear_c{i + 1}")
            cf = cf_layer(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            if i > 0:
                cf = F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(cf)
        return self.linear_fuse(torch.cat(outs[::-1], dim=1))

"""
2026/03/27
"""
class SubclassSegFormerHead_visual_prompt(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19, subclass: int = 10,
                 total_field: int = 5, num_prompts: int = 1) -> None:
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i + 1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout2d(0.2)

        # 綁定帶有 q_prime 的新版 Block
        self.subclass_block = SubclassBlock2L2_Adaptive_W_visual_prompt(embed_dim, subclass, total_field, num_prompts=num_prompts)

    # 【關鍵修改 1】：接收 prompt_enabled 參數
    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor], output_subclass=True,
                prompt_enabled=False) -> Tensor:
        B, _, H, W = features[0].shape

        # 使用 getattr 替代 eval，更穩定
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]
        for i, feature in enumerate(features[1:]):
            cf_layer = getattr(self, f"linear_c{i + 2}")
            cf = cf_layer(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))

        # 【關鍵修改 2】：將 prompt_enabled 往下傳給 subclass_block
        seg, subclass = self.subclass_block(self.dropout(seg), output_subclass=output_subclass,
                                            prompt_enabled=prompt_enabled)
        return seg, subclass

    # 用來把 5 個場域特徵存成 .pt 檔
    def get_fusion_map(self, features):
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]
        for i, feature in enumerate(features[1:]):
            cf_layer = getattr(self, f"linear_c{i + 2}")
            cf = cf_layer(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))
        return self.linear_fuse(torch.cat(outs[::-1], dim=1))


"""
origin
"""
class SubclassSegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19, subclass: int = 10, kernel_size: int = 3, loop: int = 3) -> None:
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        # self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.2)
        # self.subclass_block = SubclassBlock2L2(embed_dim, subclass, kernel_size, loop)
        self.total_field = 5
        self.subclass_block = SubclassBlock2L2_Adaptive_W(embed_dim, subclass, kernel_size, self.total_field, loop)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor], subclass_weight=None, output_subclass=True) -> Tensor:
        # H/4 , W/4
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            # Upsampling (B, 256, H, W)
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))
        # Fusion at last dim (B, 1024, H, W) > conv (B, 256, H, W)
        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        # seg = self.linear_pred(self.dropout(seg))
        if subclass_weight is None:
            # embedding_feature_map(B, 256, H, W) --> seg (B, 2, H, W) , subclass (B, 64, H, W)
            seg, subclass = self.subclass_block(self.dropout(seg), output_subclass=output_subclass)
        else:
            print(subclass_weight)
            seg, subclass = self.subclass_block(self.dropout(seg), subclass_weight, output_subclass)
        return seg, subclass