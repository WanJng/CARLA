import carla
import time


def main():
    recording_name = "ppo_final_eval.log"
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    print(f"发送回放指令: {recording_name}")
    # 播放回放，不使用自带的视角绑定（我们下面自己写）
    client.replay_file(recording_name, 0, 0, 0)

    world = client.get_world()

    # 给 CARLA 引擎 2 秒钟的时间加载回放中的车辆实体
    print("等待实体加载...")
    time.sleep(2.0)

    # 扫描当前世界里的所有车辆
    vehicles = world.get_actors().filter('vehicle.*')

    if not vehicles:
        print("❌ 警告：回放已经开始，但地图上找不到任何车辆！请查看下方的【原因二】。")
        return

    # 随便取第一辆车（通常就是我们的自车 Ego）
    target_vehicle = vehicles[0]
    print(f"✅ 找到车辆 (ID: {target_vehicle.id}，型号: {target_vehicle.type_id})，正在自动跟随视角...")

    spectator = world.get_spectator()

    try:
        print("🎥 正在跟随车辆，请切回 CARLA 窗口观看！（按 Ctrl+C 退出脚本）")
        # 写一个循环，让上帝视角永远跟在车后方 6 米，上方 3 米的位置
        while True:
            transform = target_vehicle.get_transform()

            # 计算车身正后方的坐标
            forward_vector = transform.get_forward_vector()
            cam_location = transform.location - carla.Location(
                x=forward_vector.x * 6.0,
                y=forward_vector.y * 6.0,
                z=-3.0  # 稍微抬高
            )

            cam_transform = carla.Transform(cam_location, transform.rotation)
            cam_transform.rotation.pitch = -15.0  # 视角往下倾斜一点看车

            spectator.set_transform(cam_transform)
            time.sleep(0.03)  # 约 30 FPS 的刷新率

    except KeyboardInterrupt:
        print("\n已退出视角跟随。")


if __name__ == "__main__":
    main()