import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import math
import random
import time

class CarlaGymEnv(gym.Env):
    """
    纯向量化的 CARLA 强化学习环境
    专为低显存、高并发训练设计
    """

    def __init__(self):
        super(CarlaGymEnv, self).__init__()

        # 1. 连接 CARLA 服务端
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        if not self.world.get_map().name.endswith('Town01'):
            print("正在加载 Town01 地图...")
            self.world = self.client.load_world('Town01')
        self._clear_all_zombies()
        # 2. 开启同步模式与固定时间步长
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # 3. 实体追踪器与传感器状态
        self.ego_vehicle = None
        self.npc_vehicle = None
        self.collision_sensor = None
        self.has_collided = False

        # 【新增】回合步数控制
        self.max_steps = 2000  # 约 100 秒物理时间
        self.current_step = 0

        # 【优化：新增】记录上一帧的方向盘动作，用于平滑性惩罚
        self.prev_steer = 0.0

        # 4. 定义动作空间与状态空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def start_recording(self, filename):
        """【新增】开启 CARLA 官方录制器，记录所有 Actor 轨迹"""
        # 文件会保存在 CARLA 服务端所在的路径下 (例如 /carla/dist/carla/PythonAPI/examples/)
        print(f"开始录制到文件: {filename}")
        self.client.start_recorder(filename)

    def stop_recording(self):
        """【新增】停止录制"""
        print("停止录制。")
        self.client.stop_recorder()

    def step(self, action):
        """核心步进函数：打通强化学习的 MDP 闭环"""
        # 【新增】步数累加
        self.current_step += 1

        # 1. 解析解耦后的动作
        steer = float(action[0])
        throttle_brake = float(action[1])

        throttle = throttle_brake if throttle_brake > 0 else 0.0
        brake = -throttle_brake if throttle_brake < 0 else 0.0

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego_vehicle.apply_control(control)

        # 2. 使用同步推进
        self.world.tick()

        # 3. 获取新状态
        obs = self._get_observation()

        # 4. 计算奖励与回合终止标志
        reward, terminated = self._get_reward()

        # 【优化修改】大幅降低方向盘变化的惩罚权重，从 -1.0 降到 -0.1
        # 让它敢于做微调，同时又能防画龙
        steer_change_penalty = -0.1 * abs(steer - self.prev_steer)
        reward += steer_change_penalty
        self.prev_steer = steer

        # 【新增】截断判定：达到最大步数上限则截断回合
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True

        info = {}

        return obs, reward, terminated, truncated, info

    def _get_reward(self):
        reward = 0.0
        terminated = False

        velocity = self.ego_vehicle.get_velocity()
        v_current = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_yaw = ego_transform.rotation.yaw

        current_waypoint = self.map.get_waypoint(ego_location)

        # 获取预瞄点 (仅用于驱动进度计算，不用于姿态惩罚)
        lookahead_distance = np.clip(v_current * 0.5, 3.0, 10.0)
        next_waypoints = current_waypoint.next(lookahead_distance)
        target_waypoint = next_waypoints[0] if next_waypoints else current_waypoint

        # 计算带符号的横向偏差和绝对距离
        current_wp_right = current_waypoint.transform.get_right_vector()
        vector_to_current_wp = ego_location - current_waypoint.transform.location
        signed_lat_dev = vector_to_current_wp.x * current_wp_right.x + vector_to_current_wp.y * current_wp_right.y
        distance_to_center = abs(signed_lat_dev)

        # ---------------- 1. 致命错误 ----------------
        if self.has_collided:
            return -200.0, True
        if distance_to_center > 2.0:
            return -200.0, True

        # ---------------- 2. 僵死熔断 ----------------
        if getattr(self, 'current_step', 0) > 50:
            if getattr(self, 'stuck_steps', None) is None:
                self.stuck_steps = 0
            if v_current < 1.0:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0
            if self.stuck_steps > 30:
                return -20.0, True
        else:
            self.stuck_steps = 0

        # ---------------- 3. 核心驱动力 ----------------
        wp_forward = target_waypoint.transform.get_forward_vector()
        v_progress = velocity.x * wp_forward.x + velocity.y * wp_forward.y
        target_speed = 10.0

        if v_progress > 0.5:
            base_progress_reward = 3.0 * min(v_progress / target_speed, 1.0)
            safety_factor = 1.0 - min(distance_to_center / 2.0, 1.0)
            reward += base_progress_reward * safety_factor

            # 【核心修复 3】：航向角惩罚必须和【当前】车道线对齐！
            # 这样它在转弯前只会乖乖走在自己车道中心，而不会过早“切弯”
            current_wp_yaw = current_waypoint.transform.rotation.yaw
            current_yaw_diff = abs((ego_yaw - current_wp_yaw + 180) % 360 - 180)
            r_heading = 0.5 * (1.0 - (min(current_yaw_diff / 15.0, 1.0)) ** 2)
            reward += r_heading

            # 强迫居中的线性拉力
            r_lane = 1.0 * (1.0 - min(distance_to_center / 1.0, 1.0))
            reward += r_lane

        elif v_progress < -0.5:
            reward -= 2.0
        else:
            reward -= 0.1

        # ---------------- 4. 舒适性与动作惩罚 (替换这部分) ----------------

        # 1. 保留原本的跳变惩罚（防抽搐）
        steer = self.ego_vehicle.get_control().steer
        # 注意：这里需要在 step 函数中把当前的 steer 传进来，或者直接获取控制状态

        # 2. 【核心新增】绝对转向角度平方惩罚 (L2 Steering Penalty)
        # 强迫网络：非必要不打大方向！
        # steer^2 使得微调(0.1)几乎无惩罚，但打死(1.0)会有极大的惩罚
        steer_magnitude_penalty = -0.5 * (steer ** 2)
        reward += steer_magnitude_penalty

        # 3. 【核心新增】高速大转向惩罚 (Speed-Steering Coupling)
        # 解决直角弯前，速度还很快就猛打方向盘导致切弯撞墙的问题
        if v_current > 5.0 and abs(steer) > 0.5:
            # 车速大于5m/s（约18km/h）时，如果方向盘打超过一半，给予额外重罚
            reward -= 1.0

        return reward, terminated

    def reset(self, seed=None, options=None):
        """环境重置：搭建新手村场景并挂载痛觉神经"""
        super().reset(seed=seed)
        self._clear_actors()
        self.has_collided = False

        # 【新增】重置当前步数与防抖记录
        self.current_step = 0
        self.stuck_steps = 0
        self.prev_steer = 0.0  # 【优化：重置】重置上一帧动作

        # 1. 随机选择生成点并生成自车 (Ego)
        ego_bp = random.choice(self.blueprint_library.filter('vehicle.tesla.model3'))
        spawn_points = self.map.get_spawn_points()

        ego_spawn_point = random.choice(spawn_points)
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, ego_spawn_point)

        if self.ego_vehicle is None:
            return self.reset()

        # 2. 给自车挂载碰撞传感器
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.try_spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # 3. 计算自车正前方 20 米的坐标，生成 NPC
        npc_bp = random.choice(self.blueprint_library.filter('vehicle.audi.a2'))
        ego_transform = self.ego_vehicle.get_transform()
        forward_vector = ego_transform.get_forward_vector()

        npc_spawn_transform = carla.Transform(
            carla.Location(
                x=ego_transform.location.x + forward_vector.x * 20.0,
                y=ego_transform.location.y + forward_vector.y * 20.0,
                z=ego_transform.location.z + 0.5
            ),
            ego_transform.rotation
        )

        self.npc_vehicle = self.world.try_spawn_actor(npc_bp, npc_spawn_transform)

        if self.npc_vehicle is None:
            return self.reset()

        # 4. 赋予 NPC 初始行为：匀速车
        self.npc_vehicle.set_autopilot(False)
        npc_control = carla.VehicleControl(throttle=0.4, steer=0.0, brake=0.0)
        self.npc_vehicle.apply_control(npc_control)

        # 5. 同步模式下等待物理引擎稳定
        for _ in range(5):
            self.world.tick()

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

        ego_trans = self.ego_vehicle.get_transform()
        ego_loc = ego_trans.location
        ego_vel = self.ego_vehicle.get_velocity()
        ego_accel = self.ego_vehicle.get_acceleration()
        ego_yaw = ego_trans.rotation.yaw

        # 1. 自车速度
        v_ego_raw = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
        v_ego = np.clip(v_ego_raw / 30.0, 0.0, 1.0)

        # 2. 自车加速度
        ego_fwd = ego_trans.get_forward_vector()
        a_ego_raw = (ego_accel.x * ego_fwd.x) + (ego_accel.y * ego_fwd.y)
        a_ego = np.clip(a_ego_raw / 8.0, -1.0, 1.0)

        carla_map = self.world.get_map()
        current_waypoint = carla_map.get_waypoint(ego_loc)

        # 【核心修复 1】：计算带有正负号的横向偏差 (Signed Cross-track Error)
        # 这样模型才能知道自己是偏左了还是偏右了！
        wp_right = current_waypoint.transform.get_right_vector()
        vec_to_ego = ego_loc - current_waypoint.transform.location
        signed_lat_dev = vec_to_ego.x * wp_right.x + vec_to_ego.y * wp_right.y
        lat_dev = np.clip(signed_lat_dev / 2.0, -1.0, 1.0)  # 偏右为正，偏左为负

        # 【核心修复 2】：使用纯追踪几何角 (Pure Pursuit Angle) 代替简单的 Yaw 差
        lookahead_distance = np.clip(v_ego_raw * 0.5, 3.0, 10.0)
        next_waypoints = current_waypoint.next(lookahead_distance)
        target_waypoint = next_waypoints[0] if next_waypoints else current_waypoint

        # 计算车身当前位置指向预瞄点的向量角度
        vec_to_target = target_waypoint.transform.location - ego_loc
        target_yaw = math.degrees(math.atan2(vec_to_target.y, vec_to_target.x))
        # 这个角度差告诉模型：“前面的路在你的左前方还是右前方”
        angle_diff = (target_yaw - ego_yaw + 180) % 360 - 180
        heading_dev = np.clip(angle_diff / 90.0, -1.0, 1.0)

        # --- 寻找前车并计算相对状态 (保持原样) ---
        target_npc = self._find_target_lead_vehicle(ego_trans)
        if target_npc:
            npc_loc = target_npc.get_location()
            npc_vel = target_npc.get_velocity()
            dist_raw = ego_loc.distance(npc_loc)
            delta_x = np.clip(dist_raw / 50.0, 0.0, 1.0)
            v_npc_fwd = math.sqrt(npc_vel.x ** 2 + npc_vel.y ** 2) * math.cos(
                math.radians(target_npc.get_transform().rotation.yaw - ego_yaw))
            delta_v_raw = v_npc_fwd - v_ego_raw
            delta_v = np.clip(delta_v_raw / 30.0, -1.0, 1.0)
            brake_signal = 1.0 if target_npc.get_light_state() & carla.VehicleLightState.Brake else 0.0
        else:
            delta_x = 1.0
            delta_v = 0.0
            brake_signal = 0.0

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
        """严谨的资源回收机制，防止显存泄漏"""
        # 新增：优先销毁传感器
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

        if self.npc_vehicle is not None:
            self.npc_vehicle.destroy()
            self.npc_vehicle = None

    def _clear_all_zombies(self):
        """无差别全局大扫除"""
        print("正在清理世界中的残留实体...")
        # 销毁所有残留车辆
        vehicles = self.world.get_actors().filter('*vehicle*')
        for v in vehicles:
            v.destroy()
        # 销毁所有残留传感器
        sensors = self.world.get_actors().filter('*sensor*')
        for s in sensors:
            s.destroy()

    def _on_collision(self, event):
        """碰撞传感器回调函数"""
        self.has_collided = True

    def close(self):
        """
        关闭环境时的清理工作
        """
        self._clear_actors()

        # 退出时恢复异步模式，防止 CARLA 服务端假死
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception as e:
            print(f"关闭同步模式时发生异常: {e}")