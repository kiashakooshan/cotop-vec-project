import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # لایه‌های مشترک (Shared Layers) برای استخراج ویژگی‌های محیط
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # شبکه Actor: خروجی آن احتمالات انتخاب هر اکشن است
        self.actor = nn.Linear(128, action_dim)
        
        # شبکه Critic: خروجی آن یک عدد اسکالر برای ارزش‌گذاری حالت (State Value) است
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state):
        # عبور وضعیت محیط از لایه‌های مشترک
        z = self.shared(state)
        
        # تولید توزیع احتمال برای اکشن‌ها (استفاده از Softmax برای شبکه Actor)
        action_probs = F.softmax(self.actor(z), dim=-1)
        
        # محاسبه ارزش حالت فعلی
        state_value = self.critic(z)
        
        return action_probs, state_value

    def get_action(self, state):
        """
        یک متد کمکی برای نمونه‌برداری (Sampling) اکشن از روی توزیع احتمال
        """
        action_probs, _ = self.forward(state)
        # نمونه‌گیری تصادفی وزن‌دار بر اساس احتمالات
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)