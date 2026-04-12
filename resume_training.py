import os
from stable_baselines3 import PPO
from carla_env import CarlaGymEnv

def resume():
    # 1. 初始化环境
    env = CarlaGymEnv()

    # 2. 指定之前保存的模型路径（例如你 50 万步时的模型）
    model_path = "./models/ppo_baseline/ppo_carla_model_500000_steps.zip" # 请根据实际文件名修改
    log_dir = "./logs/ppo_baseline/"

    # 3. 加载模型并关联环境
    print(f"正在加载模型: {model_path}")
    model = PPO.load(model_path, env=env, device="cuda")

    # 4. 继续训练
    # 建议再跑 500,000 步
    total_steps_to_add = 500000

    print(f"开始续训，计划新增步数: {total_steps_to_add}")
    model.learn(
        total_timesteps=total_steps_to_add,
        tb_log_name="PPO_run_1_resumed", # TensorBoard 会创建一个新文件夹记录
        reset_num_timesteps=False,       # 【关键】设为 False，TensorBoard 的时间轴会接在 50w 步之后
        progress_bar=True
    )

    # 5. 保存最终模型
    model.save("./models/ppo_baseline/ppo_carla_final_v2")
    env.close()

if __name__ == "__main__":
    resume()