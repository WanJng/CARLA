import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import math
import random
import time


class CarlaGymEnv(gym.Env):
    """
    高级 CARLA 强化学习环境 (支持交规、交通流、行人与平顺控制)
    """

    def __init__(self):
        super(CarlaGymEnv, self).__init__()

        # 1. 连接 CARLA 服务端与交通管理器 (Traffic Manager)
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        if not self.world.get_map().name.endswith('Town01'):
            print("正在加载 Town01 地图...")
            self.world = self.client.load_world('Town01')

        # 设置 Traffic Manager
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.global_percentage_speed_difference(10.0)  # 让 NPC 开慢点，模拟城市路况

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
        self.collision_sensor = None
        self.has_collided = False
        self.npc_list = []  # 记录生成的 NPC 车辆和行人，方便回收

        # 回合步数与平顺性记录
        self.max_steps = 2000
        self.current_step = 0
        self.stuck_steps = 0

        self.prev_steer = 0.0
        self.prev_throttle_brake = 0.0

        # 4. 定义动作空间与【全新 10 维状态空间】
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Obs: [v_ego, a_ego, lat_dev, heading_dev, hazard_dist, hazard_v, hazard_brake, tl_state, tl_dist, hazard_type]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)


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

        # 3. 获取新状态 (传入 obs 以便后续提取交规信息)
        obs = self._get_observation()

        # 4. 计算奖励与回合终止标志
        reward, terminated = self._get_reward(obs)

        # ---------------- 【必要优化：L2 二次方平顺性惩罚】 ----------------
        # 方向盘防抖 (严惩出弯甩尾和暴力反打，权重调高至 -2.5)
        steer_change_penalty = -1 * ((steer - self.prev_steer) ** 2)
        reward += steer_change_penalty
        self.prev_steer = steer

        # 踏板防抖 (严惩满油门瞬间接满刹车)
        # 权重设为 -0.1，允许平滑收放油门，但严惩 1.0 直接切到 -1.0
        throttle_change_penalty = -0.1 * abs(throttle_brake - self.prev_throttle_brake)
        reward += throttle_change_penalty
        self.prev_throttle_brake = throttle_brake
        # -----------------------------------------------------------------

        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _get_reward(self, obs):  # 注意这里增加了 obs 参数
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

        # --- 【必要新增】：从 Obs 中提取交规与障碍物信息 ---
        hazard_dist_norm = obs[4]
        tl_state = obs[7]
        tl_dist_norm = obs[8]

        is_red_light = (tl_state > 0.5)
        is_near_intersection = (tl_dist_norm < 0.2)  # 距离路口约 10 米内
        hazard_is_close = (hazard_dist_norm < 0.2)  # 距离障碍物约 10 米内

        # ---------------- 1. 致命错误 ----------------
        if self.has_collided:
            return -200.0, True
        if distance_to_center > 2.0:
            return -200.0, True

        # 【必要新增】：闯红灯致死逻辑
        if is_red_light and is_near_intersection and v_current > 2.5:
            # 红灯跟前还保持高车速，视为闯红灯，直接重罚并结束
            return -200.0, True

        # ---------------- 2. 僵死熔断与智慧停车 ----------------
        if v_current < 0.5:
            if is_red_light or hazard_is_close:
                # 【必要新增】：合法停车（等红灯/避让行人），给正向奖励并重置僵死状态
                reward += 1.0
                self.stuck_steps = 0
            else:
                # 原有的无故怠速惩罚逻辑保持不变
                if getattr(self, 'current_step', 0) > 50:
                    if getattr(self, 'stuck_steps', None) is None:
                        self.stuck_steps = 0
                    self.stuck_steps += 1
                    reward -= 0.1
                    if self.stuck_steps > 30:
                        return -20.0, True
        else:
            self.stuck_steps = 0

        # ---------------- 3. 核心驱动力 (原封不动) ----------------
        wp_forward = target_waypoint.transform.get_forward_vector()
        v_progress = velocity.x * wp_forward.x + velocity.y * wp_forward.y
        target_speed = 10.0

        if v_progress > 0.5:
            base_progress_reward = 3.0 * min(v_progress / target_speed, 1.0)
            safety_factor = 1.0 - min(distance_to_center / 2.0, 1.0)
            reward += base_progress_reward * safety_factor

            # 原封不动的极其成功的姿态对齐奖励
            current_wp_yaw = current_waypoint.transform.rotation.yaw
            current_yaw_diff = abs((ego_yaw - current_wp_yaw + 180) % 360 - 180)
            r_heading = 0.5 * (1.0 - (min(current_yaw_diff / 15.0, 1.0)) ** 2)
            reward += r_heading

            r_lane = 1.0 * (1.0 - min(distance_to_center / 1.0, 1.0))
            reward += r_lane

            # 【必要新增】：严格的超速惩罚 (治好地板油)
            if v_current > target_speed:
                speeding_penalty = -0.5 * (v_current - target_speed)
                reward += speeding_penalty

        elif v_progress < -0.5:
            reward -= 2.0
        # 无故怠速的扣分在前面处理过了

        # ---------------- 4. 绝对控制幅度惩罚 (原封不动) ----------------
        steer = self.ego_vehicle.get_control().steer

        steer_magnitude_penalty = -0.5 * (steer ** 2)
        reward += steer_magnitude_penalty

        if v_current > 5.0 and abs(steer) > 0.5:
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
        self.prev_throttle_brake = 0.0  # 【重置】

        # 1. 随机选择生成点并生成自车 (Ego)
        ego_bp = random.choice(self.blueprint_library.filter('vehicle.tesla.model3'))
        spawn_points = self.map.get_spawn_points()

        ego_spawn_point = random.choice(spawn_points)
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, ego_spawn_point)

        if self.ego_vehicle is None:
            return self.reset()

        self._spawn_background_traffic()

        # 2. 给自车挂载碰撞传感器
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.try_spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # 5. 同步模式下等待物理引擎稳定
        for _ in range(5):
            self.world.tick()

        obs = self._get_observation()
        info = {}

        return obs, info

    def _get_observation(self):
        """提取 10 维交规与环境感知向量"""
        if self.ego_vehicle is None:
            return np.zeros(10, dtype=np.float32)

        ego_trans = self.ego_vehicle.get_transform()
        ego_loc = ego_trans.location
        ego_vel = self.ego_vehicle.get_velocity()
        ego_yaw = ego_trans.rotation.yaw

        v_ego_raw = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
        v_ego = np.clip(v_ego_raw / 30.0, 0.0, 1.0)

        ego_accel = self.ego_vehicle.get_acceleration()
        ego_fwd = ego_trans.get_forward_vector()
        a_ego_raw = (ego_accel.x * ego_fwd.x) + (ego_accel.y * ego_fwd.y)
        a_ego = np.clip(a_ego_raw / 8.0, -1.0, 1.0)

        carla_map = self.world.get_map()
        current_waypoint = carla_map.get_waypoint(ego_loc)

        # 3. 带符号横向偏差
        wp_right = current_waypoint.transform.get_right_vector()
        vec_to_ego = ego_loc - current_waypoint.transform.location
        signed_lat_dev = vec_to_ego.x * wp_right.x + vec_to_ego.y * wp_right.y
        lat_dev = np.clip(signed_lat_dev / 2.0, -1.0, 1.0)

        # 4. Pure Pursuit 预瞄角
        lookahead_distance = np.clip(v_ego_raw * 0.5, 3.0, 10.0)
        next_waypoints = current_waypoint.next(lookahead_distance)
        target_waypoint = next_waypoints[0] if next_waypoints else current_waypoint
        vec_to_target = target_waypoint.transform.location - ego_loc
        target_yaw = math.degrees(math.atan2(vec_to_target.y, vec_to_target.x))
        angle_diff = (target_yaw - ego_yaw + 180) % 360 - 180
        heading_dev = np.clip(angle_diff / 90.0, -1.0, 1.0)

        # ---------------- 【全新感知：交通灯】 ----------------
        tl_state = 0.0
        tl_dist = 1.0
        if self.ego_vehicle.is_at_traffic_light():
            tl = self.ego_vehicle.get_traffic_light()
            if tl and tl.get_state() in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                tl_state = 1.0  # 前方是红灯或黄灯
                # 计算与红绿灯的距离
                dist_raw = ego_loc.distance(tl.get_location())
                tl_dist = np.clip(dist_raw / 50.0, 0.0, 1.0)

        # ---------------- 【全新感知：全类型危险物 (车/人)】 ----------------
        target_hazard, hazard_type_val = self._find_nearest_hazard(ego_trans)
        if target_hazard:
            hz_loc = target_hazard.get_location()
            hz_vel = target_hazard.get_velocity()
            dist_raw = ego_loc.distance(hz_loc)
            hazard_dist = np.clip(dist_raw / 50.0, 0.0, 1.0)

            v_hz_fwd = math.sqrt(hz_vel.x ** 2 + hz_vel.y ** 2) * math.cos(
                math.radians(target_hazard.get_transform().rotation.yaw - ego_yaw))
            delta_v_raw = v_hz_fwd - v_ego_raw
            hazard_v = np.clip(delta_v_raw / 30.0, -1.0, 1.0)

            # 判断是否踩刹车 (行人默认不发刹车信号，设为0)
            if hasattr(target_hazard, 'get_light_state') and (
                    target_hazard.get_light_state() & carla.VehicleLightState.Brake):
                hazard_brake = 1.0
            else:
                hazard_brake = 0.0
        else:
            hazard_dist = 1.0
            hazard_v = 0.0
            hazard_brake = 0.0
            hazard_type_val = 0.0  # 0=无危险, 0.5=车辆, 1.0=行人

        # 组合 10 维向量
        obs = np.array([v_ego, a_ego, lat_dev, heading_dev, hazard_dist, hazard_v, hazard_brake, tl_state, tl_dist,
                        hazard_type_val], dtype=np.float32)
        return obs

    def _find_nearest_hazard(self, ego_trans, max_distance=50.0):
        """寻找前方最近的危险源 (不仅找车，还要找人)"""
        ego_loc = ego_trans.location
        ego_fwd = ego_trans.get_forward_vector()
        ego_right = ego_trans.get_right_vector()

        closest_dist = float('inf')
        target_hazard = None
        hazard_type = 0.0  # 0.5 车辆, 1.0 行人

        # 合并遍历车辆和行人
        all_hazards = self.world.get_actors().filter('*vehicle*')
        walkers = self.world.get_actors().filter('*walker*')

        hazards_to_check = list(all_hazards) + list(walkers)

        for hz in hazards_to_check:
            if hz.id == self.ego_vehicle.id:
                continue

            hz_loc = hz.get_location()
            vec_rel_x = hz_loc.x - ego_loc.x
            vec_rel_y = hz_loc.y - ego_loc.y

            dx = vec_rel_x * ego_fwd.x + vec_rel_y * ego_fwd.y
            dy = vec_rel_x * ego_right.x + vec_rel_y * ego_right.y

            # 筛选：在前方，在探测距离内，且在同一车道/斑马线宽度内 (|dy| < 2.5米 容忍度设宽点以保护行人)
            if 0 < dx < max_distance and abs(dy) < 2.5:
                if dx < closest_dist:
                    closest_dist = dx
                    target_hazard = hz
                    hazard_type = 1.0 if 'walker' in hz.type_id else 0.5

        return target_hazard, hazard_type

    def _spawn_background_traffic(self, num_vehicles=15, num_walkers=10):
        """利用 Traffic Manager 生成服从交规的交通流"""
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        # 1. 生成 NPC 车辆
        v_bps = self.blueprint_library.filter('vehicle.*')
        for i in range(min(num_vehicles, len(spawn_points))):
            bp = random.choice(v_bps)
            npc = self.world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                npc.set_autopilot(True, self.tm.get_port())
                self.npc_list.append(npc)

        # 2. 生成行人 (随机散布在人行道上)
        w_bps = self.blueprint_library.filter('walker.pedestrian.*')
        for _ in range(num_walkers):
            spawn_loc = self.world.get_random_location_from_navigation()
            if spawn_loc is not None:
                bp = random.choice(w_bps)
                walker = self.world.try_spawn_actor(bp, carla.Transform(spawn_loc))
                if walker is not None:
                    self.npc_list.append(walker)

    def _clear_actors(self):
        """严格的资源回收机制"""
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

        # 销毁通过 TM 生成的 NPC
        for npc in self.npc_list:
            if npc.is_alive:
                npc.destroy()
        self.npc_list.clear()

    def _clear_all_zombies(self):
        """无差别全局大扫除"""
        print("正在清理世界中的残留实体...")
        for actor_type in ['*vehicle*', '*sensor*', '*walker*']:
            actors = self.world.get_actors().filter(actor_type)
            for a in actors:
                a.destroy()

    def _on_collision(self, event):
        self.has_collided = True

    def close(self):
        self._clear_actors()
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception as e:
            print(f"关闭同步模式时发生异常: {e}")