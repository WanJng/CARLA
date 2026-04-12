import time
import psutil
from carla_env import CarlaGymEnv


def main():
    print("开始初始化 CARLA 向量化环境...")
    env = CarlaGymEnv()
    obs, info = env.reset()

    total_steps = 10000
    start_time = time.time()

    print(f"环境初始化成功！开始执行 {total_steps} 步随机压力测试...")

    for step in range(1, total_steps + 1):
        # 随机采样动作：方向盘[-1, 1], 油门/刹车[-1, 1]
        action = env.action_space.sample()

        # 步进环境
        obs, reward, terminated, truncated, info = env.step(action)

        # 打印进度与观测向量
        if step % 100 == 0:
            memory_usage = psutil.virtual_memory().percent
            print(f"进度: {step}/{total_steps} 步 | 内存占用: {memory_usage}%")
            print(f"当前状态向量: {obs}")

        # 模拟回合结束（碰撞、超时或主动截断）
        # 这里为了测试 reset 逻辑，我们强制每 500 步重置一次
        if terminated or truncated or step % 500 == 0:
            print(f"--- 触发环境重置 (Step: {step}) ---")
            obs, info = env.reset()

    env.close()
    elapsed_time = time.time() - start_time
    print(f"测试完成！总耗时: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()