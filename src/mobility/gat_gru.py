import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class MobilityDetector(nn.Module):
    def __init__(self, in_dim=16, hidden=32, gru_hidden=64):
        super().__init__()
        
        # لایه MLP برای استخراج ویژگی‌های اولیه
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 24), 
            nn.LeakyReLU(),
            nn.Linear(24, hidden)
        )
        
        # شبکه‌های توجه گراف (GAT) طبق مشخصات مقاله
        # لایه اول با 4 سر توجه (Multi-head attention)
        self.gat1 = GATConv(hidden, hidden, heads=4, concat=False)
        # لایه دوم با 1 سر توجه برای تجمیع ویژگی‌ها
        self.gat2 = GATConv(hidden, hidden, heads=1, concat=False)
        
        # بخش Encoder-Decoder با استفاده از شبکه‌های GRU
        self.encoder = nn.GRU(hidden, gru_hidden, num_layers=2, batch_first=True, dropout=0.5)
        self.decoder = nn.GRU(gru_hidden, gru_hidden, num_layers=2, batch_first=True, dropout=0.5)
        
        # لایه خروجی برای پیش‌بینی مختصات (x, y)
        self.out = nn.Linear(gru_hidden, 2)

    def forward(self, x_seq, edge_index_seq, future_steps=5):
        """
        x_seq: دنباله ویژگی‌های خودروها در زمان‌های گذشته
        edge_index_seq: ماتریس مجاورت گراف در زمان‌های گذشته
        future_steps: تعداد قدم‌هایی که در آینده پیش‌بینی می‌کنیم
        """
        feats = []
        # پردازش ویژگی‌های مکانی-زمانی هر گام تاریخی
        for t in range(x_seq.size(1)):
            e = self.mlp(x_seq[:, t])
            e = torch.relu(self.gat1(e, edge_index_seq[t]))
            e = self.gat2(e, edge_index_seq[t])
            feats.append(e)
            
        feats = torch.stack(feats, dim=1)
        
        # استخراج ویژگی‌های پنهان توالی زمانی با انکودر
        _, h = self.encoder(feats)
        
        # آماده‌سازی ورودی دیکودر برای پیش‌بینی آینده
        dec_in = h[-1].unsqueeze(1).repeat(1, future_steps, 1)
        out, _ = self.decoder(dec_in, h)
        
        return self.out(out)