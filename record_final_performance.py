import os
import numpy as np
from stable_baselines3 import PPO
from carla_env import CarlaGymEnv  # 确保当前目录下有之前的 carla_env.py


def record():
    # 1. 配置文件路径
    model_path = "./models/ppo_baseline/ppo_carla_model_500000_steps.zip"
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

    # 5. 运行一个完整回合
    for step in range(env.max_steps):
        # 使用 deterministic=True 确保输出动作是确定性的
        action, _states = model.predict(obs, deterministic=True)

        # 打印实时状态
        print(f"第 {step} 步 | 预测动作: 方向盘 {action[0]:.2f}, 油门/刹车 {action[1]:.2f}")

        obs, reward, terminated, truncated, info = env.step(action)

        # 6. 判断并打印细致的停止原因
        if terminated or truncated:
            reason = "未知原因"

            if truncated:
                reason = f"达到最大步数上限 ({env.max_steps} 步)"
            elif terminated:
                # 根据 carla_env.py 中的逻辑推断具体原因
                if env.has_collided:
                    reason = "检测到碰撞 (Collision)"
                elif abs(obs[2]) >= 1.0:  # lat_dev = signed_lat_dev / 2.0，超过 1.0 即为偏差 > 2m
                    reason = "车辆严重偏离车道中心线 (Out of Lane)"
                elif obs[7] > 0.5 and obs[8] < 0.2 and (obs[0] * 30.0) > 2.5:
                    # tl_state > 0.5 (红灯) 且 tl_dist < 0.2 (近路口) 且 v > 2.5 m/s
                    reason = "违反交通规则：闯红灯"
                elif env.stuck_steps > 30:
                    reason = "车辆长时间卡死或无故静止 (Stuck)"
                else:
                    reason = "其他环境定义的终止条件"

            print("-" * 30)
            print(f">>> 回合结束 <<<")
            print(f"最终原因: {reason}")
            print(f"运行步数: {step + 1}")
            print("-" * 30)
            break

    # 7. 停止录制并清理环境
    env.stop_recording()
    env.close()
    print("录制脚本执行完毕！")


if __name__ == "__main__":
    record()