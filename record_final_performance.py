import os
from stable_baselines3 import PPO
from carla_env import CarlaGymEnv  # 确保当前目录下有之前的 carla_env.py


def record():
    # 1. 配置文件路径
    model_path = "./models/ppo_baseline/ppo_carla_final.zip"
    recording_name = "ppo_final_eval_17.log"

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    # 2. 加载训练好的模型
    print(f"正在加载模型: {model_path} ...")
    model = PPO.load(model_path)

    # 3. 初始化环境
    print("正在初始化 CARLA 向量化环境...")
    env = CarlaGymEnv()

    # 4. 重置环境并开始录制
    obs, info = env.reset()
    env.start_recording(recording_name)
    print(f"开始运行并录制，回放文件将被保存在 CARLA 服务端目录下: {recording_name}")

    # 5. 运行一个完整回合 (使用 deterministic=True 确保输出动作是确定性的，以评估真实表现)
    for step in range(env.max_steps):
        action, _states = model.predict(obs, deterministic=True)
        print(f"第 {step} 步 | 预测动作: 方向盘 {action[0]:.2f}, 油门/刹车 {action[1]:.2f}")
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            reason = "碰撞或出界" if terminated else "达到最大步数"
            print(f"回合结束 (原因: {reason})，共运行了 {step + 1} 步。")
            break

    # 6. 停止录制并清理环境
    env.stop_recording()
    env.close()
    print("录制脚本执行完毕！")


if __name__ == "__main__":
    record()