import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from carla_env import CarlaGymEnv  # 导入你第一阶段封装好的环境

# ==========================================
# 1. 初始化目录与日志设置
# ==========================================
log_dir = "./ppo_carla_tensorboard/"
model_dir = "./saved_models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ==========================================
# 2. 实例化并验证环境
# ==========================================
print("正在初始化 CARLA 环境...")
env = CarlaGymEnv()

# [关键排错步骤] 使用 SB3 自带的工具严格检查你的自定义环境接口是否符合 Gymnasium 标准
print("正在执行环境标准接口检查...")
try:
    check_env(env)
    print("-> 环境接口检查完美通过！")
except Exception as e:
    print(f"-> 环境接口存在问题，请按提示修正: {e}")
    exit(1)

# ==========================================
# 3. 实例化 PPO 算法大脑
# ==========================================
# 策略网络：MlpPolicy (因为我们是 7 维纯向量输入，不需要 CNN)
# 硬件加速：device="cuda" 确保调用你的 RTX 5060
print("正在加载 PPO 算法模型...")
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,                 # 打印训练进度日志
    tensorboard_log=log_dir,   # 绑定 TensorBoard 监控目录
    device="cuda",             # 强制使用 GPU 加速
    learning_rate=3e-4,        # 初始学习率 (PPO 的经典默认值)
    n_steps=2048,              # 每次更新前收集的步数
    batch_size=64,             # 神经网络优化的批次大小
)

# ==========================================
# 4. 启动基准 (Baseline) 训练
# ==========================================
# 初期先设定 50,000 步，主要为了验证代码能跑通，并观察奖励是否有收敛上升的趋势
total_steps = 50000
print(f"开始执行 Baseline 训练，目标步数: {total_steps} ...")

# tb_log_name 会在 tensorboard 目录下生成类似 ppo_baseline_01_1 的子文件夹
model.learn(total_timesteps=total_steps, tb_log_name="ppo_baseline_01")

# ==========================================
# 5. 保存模型与清理资源
# ==========================================
model_path = os.path.join(model_dir, "ppo_carla_baseline_50k")
model.save(model_path)
print(f"🎉 训练完成！模型已保存至: {model_path}.zip")

env.close()