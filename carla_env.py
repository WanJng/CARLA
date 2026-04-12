import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import math

class CarlaGymEnv(gym.Env):
    """
    纯向量化的 CARLA 强化学习环境
    专为低显存、高并发训练设计
    """

    def __init__(self):
        super(CarlaGymEnv, self).__init__()

        # 1. 连接 CARLA 服务端 (默认本地 2000 端口)
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # 获取地图所有的默认出生点
        self.spawn_points = self.world.get_map().get_spawn_points()

        # 内部状态变量
        self.ego_vehicle = None

        # 2. 定义动作空间 (Action Space)
        # action[0]: 转向 (Steer)，范围 [-1.0, 1.0]
        # action[1]: 油门/刹车 (Throttle/Brake)，范围 [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 3. 定义观测空间 (Observation Space)
        # 7维纯物理向量，为了神经网络收敛，全部归一化到 [-1.0, 1.0] 或 [0.0, 1.0]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def step(self, action):
        """
        环境步进：执行动作 -> 推进物理 -> 返回新状态
        """
        # 1. 解析动作并应用到自车
        steer = float(action[0])
        throttle_brake = float(action[1])

        # 核心技巧：分离合并的油门/刹车指令
        throttle = throttle_brake if throttle_brake > 0 else 0.0
        brake = -throttle_brake if throttle_brake < 0 else 0.0

        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        self.ego_vehicle.apply_control(control)

        # 2. 等待 CARLA 服务端进行一次物理 Tick
        self.world.wait_for_tick()

        # 3. 获取下一帧观测 (由第三步详细实现)
        obs = self._get_observation()

        # 4. 计算 Reward (第二阶段实现，当前先给 0)
        reward = 0.0

        # 5. 判定回合是否结束 (撞车/超时等，当前先给 False)
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        回合重置：清理现场 -> 重新生成 -> 返回初始状态
        """
        super().reset(seed=seed)

        # 1. 彻底清理上一回合遗留的实体（防止显存泄漏！）
        self._clear_actors()

        # 2. 生成自车 (以 Tesla Model 3 为例)
        ego_bp = self.blueprint_library.find('vehicle.tesla.model3')

        # 随机选择一个出生点
        spawn_point = np.random.choice(self.spawn_points)
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_point)

        # 异常处理：如果由于碰撞等原因生成失败，抛出异常
        if self.ego_vehicle is None:
            raise RuntimeError("自车生成失败，可能是出生点被占用！")

        # 3. 让物理引擎空转几帧，确保车辆稳定落地
        for _ in range(10):
            self.world.wait_for_tick()

        # 4. 获取初始观测值
        obs = self._get_observation()
        info = {}

        return obs, info

    def _get_observation(self):
        """
        提取并归一化 7 维纯物理向量
        1. 自车速度 [0, 1]
        2. 自车加速度 [-1, 1]
        3. 横向偏差 [-1, 1]
        4. 航向角偏差 [-1, 1]
        5. 前车相对距离 [0, 1]
        6. 前车相对速度 [-1, 1]
        7. 前车制动信号 [0, 1]
        """
        if self.ego_vehicle is None:
            return np.zeros(7, dtype=np.float32)

        # 获取自车状态
        ego_trans = self.ego_vehicle.get_transform()
        ego_loc = ego_trans.location
        ego_vel = self.ego_vehicle.get_velocity()
        ego_accel = self.ego_vehicle.get_acceleration()

        # 1. 自车速度归一化 (假设最高限速 30m/s)
        v_ego_raw = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
        v_ego = np.clip(v_ego_raw / 30.0, 0.0, 1.0)

        # 2. 自车纵向加速度归一化 (假设最大加速度为 8m/s^2)
        ego_fwd = ego_trans.get_forward_vector()
        a_ego_raw = (ego_accel.x * ego_fwd.x) + (ego_accel.y * ego_fwd.y)  # 点乘求前向加速度
        a_ego = np.clip(a_ego_raw / 8.0, -1.0, 1.0)

        # 获取当前位置的地图 Waypoint，用于计算偏差
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(ego_loc)

        # 3. 横向偏差归一化 (假设最大允许偏差 2.0m)
        # 用自车位置与waypoint位置的距离近似
        lat_dev_raw = ego_loc.distance(waypoint.transform.location)
        lat_dev = np.clip(lat_dev_raw / 2.0, 0.0, 1.0)  # 简化为绝对偏差

        # 4. 航向角偏差归一化 (角度差值缩放到 [-1, 1])
        ego_yaw = ego_trans.rotation.yaw
        wp_yaw = waypoint.transform.rotation.yaw
        yaw_diff = (ego_yaw - wp_yaw + 180) % 360 - 180  # 将角度差限制在 [-180, 180] 之间
        heading_dev = np.clip(yaw_diff / 90.0, -1.0, 1.0)  # 假设超过 90 度就直接截断

        # --- 寻找前车并计算相对状态 ---
        target_npc = self._find_target_lead_vehicle(ego_trans)

        if target_npc:
            # 存在前车
            npc_loc = target_npc.get_location()
            npc_vel = target_npc.get_velocity()

            # 5. 前车相对距离归一化 (最大探测距离设为 50m)
            dist_raw = ego_loc.distance(npc_loc)
            delta_x = np.clip(dist_raw / 50.0, 0.0, 1.0)

            # 6. 前车相对速度归一化 (NPC速度在自车前向的投影 - 自车速度)
            v_npc_fwd = math.sqrt(npc_vel.x ** 2 + npc_vel.y ** 2) * math.cos(
                math.radians(target_npc.get_transform().rotation.yaw - ego_yaw))
            delta_v_raw = v_npc_fwd - v_ego_raw
            delta_v = np.clip(delta_v_raw / 30.0, -1.0, 1.0)

            # 7. 前车制动信号
            brake_signal = 1.0 if target_npc.get_light_state() & carla.VehicleLightState.Brake else 0.0
        else:
            # 前方无车，赋予安全默认值
            delta_x = 1.0  # 距离最远 (安全)
            delta_v = 0.0  # 相对速度为0
            brake_signal = 0.0

        # 组合成 7 维向量
        obs = np.array([v_ego, a_ego, lat_dev, heading_dev, delta_x, delta_v, brake_signal], dtype=np.float32)
        return obs

    def _find_target_lead_vehicle(self, ego_trans, max_distance=50.0, lane_width_threshold=1.5):
        """
        利用向量点乘筛选同车道正前方的最近车辆
        """
        ego_loc = ego_trans.location
        ego_fwd = ego_trans.get_forward_vector()
        ego_right = ego_trans.get_right_vector()

        vehicles = self.world.get_actors().filter('vehicle.*')

        closest_dist = float('inf')
        target_vehicle = None

        for npc in vehicles:
            if npc.id == self.ego_vehicle.id:
                continue

            npc_loc = npc.get_location()

            # 相对位移向量
            vec_rel_x = npc_loc.x - ego_loc.x
            vec_rel_y = npc_loc.y - ego_loc.y

            # 投影到自车前向向量 (纵向距离)
            dx = vec_rel_x * ego_fwd.x + vec_rel_y * ego_fwd.y

            # 投影到自车右向向量 (横向距离)
            dy = vec_rel_x * ego_right.x + vec_rel_y * ego_right.y

            # 筛选条件：在前方 (dx > 0) 且在探测范围内，且在同一车道内 (|dy| < threshold)
            if 0 < dx < max_distance and abs(dy) < lane_width_threshold:
                if dx < closest_dist:
                    closest_dist = dx
                    target_vehicle = npc

        return target_vehicle

    def _clear_actors(self):
        """
        安全销毁场景中的实体
        """
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def close(self):
        """
        关闭环境时的清理工作
        """
        self._clear_actors()