import carla
import time


def main():
    recording_name = "ppo_final_eval_3.log"
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    print(f"正在加载回放文件: {recording_name}")

    # 1. 先“试播”一下，目的是让引擎把录像里的车在地图上生成出来
    client.replay_file(recording_name, 0, 0, 0)

    # 给引擎 1 秒钟把车辆刷出来
    time.sleep(1.0)

    # 2. 去世界里找我们要跟踪的车
    world = client.get_world()
    vehicles = world.get_actors().filter('vehicle.*')

    if not vehicles:
        print("❌ 找不到车辆！请确认录制文件是否有效。")
        return

    # 在环境代码中，自车是特斯拉，NPC是奥迪。我们专门找出特斯拉的 ID
    ego_vehicle = None
    for v in vehicles:
        if 'tesla' in v.type_id:
            ego_vehicle = v
            break

    # 兜底：如果没找到特斯拉，就选列表里的第一辆车
    if ego_vehicle is None:
        ego_vehicle = vehicles[0]

    print(f"✅ 锁定自车 (ID: {ego_vehicle.id}, 型号: {ego_vehicle.type_id})")
    print("🚀 正在启用引擎原生平滑追踪并重新开始播放...")

    # 3. 核心魔法：第四个参数传入车辆 ID！
    # 引擎接管视角，从 0 秒重新播放，完美丝滑不闪烁
    client.replay_file(recording_name, 0, 0, ego_vehicle.id)

    print("=======================================")
    print("🎬 原生视角回放已启动，请切回 CARLA 窗口观看！")
    print("按 Ctrl+C 随时结束本脚本（不影响播放）")
    print("=======================================")

    try:
        # 保持脚本活着就行，不用再手动算坐标了
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n脚本退出。")


if __name__ == "__main__":
    main()