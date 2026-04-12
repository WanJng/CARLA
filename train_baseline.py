import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from carla_env import CarlaGymEnv

def main():
    # 1. 定义 TensorBoard 日志和模型权重的保存路径
    log_dir = "./logs/ppo_baseline/"
    model_dir = "./models/ppo_baseline/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. 实例化封装好的纯向量 CARLA 环境
    print("正在初始化 CARLA 向量化环境...")
    # 采用 1 个单线程环境
    env = CarlaGymEnv()

    # 3. 配置并实例化 PPO 模型
    # 策略选型：MlpPolicy，完美契合 7 维状态向量，大幅降低计算开销
    print("正在构建 PPO 策略网络...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,     # PPO 的经典初始学习率
        n_steps=4096,           # 每次更新前收集的步数【修改】收集更多数据再更新一次，防止一惊一乍
        batch_size=256,          # 每次梯度下降的批次大小【修改】配合更大的 n_steps
        n_epochs=10,            # 每次收集数据后优化策略的轮数
        gamma=0.99,             # 折扣因子，0.99 适合这种偏长期的驾驶任务
        ent_coef=0.02,  # 【新增】强制加入 2% 的随机探索，防止它太早认命！
        device="cuda"
    )

    # 4. 设置回调函数：防止训练意外中断，每跑 50000 步存一次档
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix="ppo_carla_model"
    )

    # 5. 正式启动训练
    # 目标：500,000 步。
    print("开始 PPO 基准训练 (目标: 500,000 步)...")
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback,
        tb_log_name="PPO_run_1",
        progress_bar=True  # 在终端显示进度条
    )

    # 6. 训练结束，保存最终模型并安全关闭环境
    model.save(os.path.join(model_dir, "ppo_carla_final"))
    print("训练完成！模型已保存。")
    env.close()

if __name__ == "__main__":
    main()