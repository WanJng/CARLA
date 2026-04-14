import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import math
import random


class CarlaGymEnv(gym.Env):
    """
    高级 CARLA 强化学习环境 (完美修复版：支持动态行人、提前预警红灯、弯道轨迹避障)
    """

    def __init__(self):
        super(CarlaGymEnv, self).__init__()

        # 1. 连接 CARLA 服务端与交通管理器
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        if not self.world.get_map().name.endswith('Town01'):
            print("正在加载 Town01 地图...")
            self.world = self.client.load_world('Town01')

        # 设置 Traffic Manager
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.global_percentage_speed_difference(10.0)

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
        self.npc_list = []

        self.max_steps = 2000
        self.current_step = 0
        self.stuck_steps = 0

        self.prev_steer = 0.0
        self.prev_throttle_brake = 0.0

        # 4. 定义动作空间与 10 维状态空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Obs: [v_ego, a_ego, lat_dev, heading_dev, hazard_dist, hazard_v, hazard_brake, tl_state, tl_dist, hazard_type]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        steer = float(action[0])
        throttle_brake = float(action[1])

        throttle = throttle_brake if throttle_brake > 0 else 0.0
        brake = -throttle_brake if throttle_brake < 0 else 0.0

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego_vehicle.apply_control(control)

        self.world.tick()

        obs = self._get_observation()
        reward, terminated = self._get_reward(obs)

        # 平顺性惩罚
        steer_change_penalty = -2.5 * ((steer - self.prev_steer) ** 2)
        reward += steer_change_penalty
        self.prev_steer = steer

        throttle_change_penalty = -0.1 * abs(throttle_brake - self.prev_throttle_brake)
        reward += throttle_change_penalty
        self.prev_throttle_brake = throttle_brake

        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _get_reward(self, obs):
        reward = 0.0
        terminated = False

        velocity = self.ego_vehicle.get_velocity()
        v_current = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_yaw = ego_transform.rotation.yaw

        current_waypoint = self.map.get_waypoint(ego_location)
        lookahead_distance = np.clip(v_current * 0.5, 3.0, 10.0)
        next_waypoints = current_waypoint.next(lookahead_distance)
        target_waypoint = next_waypoints[0] if next_waypoints else current_waypoint

        current_wp_right = current_waypoint.transform.get_right_vector()
        vector_to_current_wp = ego_location - current_waypoint.transform.location
        signed_lat_dev = vector_to_current_wp.x * current_wp_right.x + vector_to_current_wp.y * current_wp_right.y
        distance_to_center = abs(signed_lat_dev)

        hazard_dist_norm = obs[4]
        tl_state = obs[7]
        tl_dist_norm = obs[8]

        is_red_light = (tl_state > 0.5)
        is_near_intersection = (tl_dist_norm < 0.2)
        hazard_is_close = (hazard_dist_norm < 0.25)

        # 1. 致命错误
        if self.has_collided or distance_to_center > 2.0:
            return -200.0, True

        if is_red_light and is_near_intersection and v_current > 2.5:
            return -200.0, True

        # 2. 僵死熔断与智慧停车
        if v_current < 0.5:
            if is_red_light or hazard_is_close:
                self.stuck_steps = 0
            else:
                if self.current_step > 50:
                    self.stuck_steps += 1
                    reward -= 0.1
                    if self.stuck_steps > 100:
                        return -20.0, True
        else:
            self.stuck_steps = 0

        # 3. 核心驱动力
        wp_forward = target_waypoint.transform.get_forward_vector()
        v_progress = velocity.x * wp_forward.x + velocity.y * wp_forward.y
        target_speed = 10.0

        if v_progress > 0.5:
            base_progress_reward = 3.0 * min(v_progress / target_speed, 1.0)
            safety_factor = 1.0 - min(distance_to_center / 2.0, 1.0)
            reward += base_progress_reward * safety_factor

            current_wp_yaw = current_waypoint.transform.rotation.yaw
            current_yaw_diff = abs((ego_yaw - current_wp_yaw + 180) % 360 - 180)
            r_heading = 0.5 * (1.0 - (min(current_yaw_diff / 15.0, 1.0)) ** 2)
            reward += r_heading

            r_lane = 1.0 * (1.0 - min(distance_to_center / 1.0, 1.0))
            reward += r_lane

            if v_current > target_speed:
                reward -= 1.0 * (v_current - target_speed)

        elif v_progress < -0.5:
            reward -= 2.0

        # 4. 绝对控制幅度惩罚
        steer = self.ego_vehicle.get_control().steer
        reward -= 0.5 * (steer ** 2)

        if v_current > 5.0 and abs(steer) > 0.5:
            reward -= 1.0

        return reward, terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._clear_actors()
        self.has_collided = False
        self.current_step = 0
        self.stuck_steps = 0
        self.prev_steer = 0.0
        self.prev_throttle_brake = 0.0

        ego_bp = random.choice(self.blueprint_library.filter('vehicle.tesla.model3'))
        spawn_points = self.map.get_spawn_points()
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, random.choice(spawn_points))

        if self.ego_vehicle is None:
            return self.reset()

        self._spawn_background_traffic()

        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.try_spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        for _ in range(10):
            self.world.tick()

        return self._get_observation(), {}

    def _get_observation(self):
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

        current_waypoint = self.map.get_waypoint(ego_loc)
        wp_right = current_waypoint.transform.get_right_vector()
        vec_to_ego = ego_loc - current_waypoint.transform.location
        signed_lat_dev = vec_to_ego.x * wp_right.x + vec_to_ego.y * wp_right.y
        lat_dev = np.clip(signed_lat_dev / 2.0, -1.0, 1.0)

        lookahead_distance = np.clip(v_ego_raw * 0.5, 3.0, 10.0)
        next_waypoints = current_waypoint.next(lookahead_distance)
        target_waypoint = next_waypoints[0] if next_waypoints else current_waypoint
        vec_to_target = target_waypoint.transform.location - ego_loc
        target_yaw = math.degrees(math.atan2(vec_to_target.y, vec_to_target.x))
        angle_diff = (target_yaw - ego_yaw + 180) % 360 - 180
        heading_dev = np.clip(angle_diff / 90.0, -1.0, 1.0)

        # ---------------- 【修复2：远距离交通灯检测】 ----------------
        tl_state, tl_dist = self._get_upcoming_traffic_light(ego_trans)

        # ---------------- 【修复3：轨迹预瞄危险检测】 ----------------
        target_hazard, hazard_type_val, hazard_dist_raw = self._find_nearest_hazard(ego_trans)

        if target_hazard:
            hazard_dist = np.clip(hazard_dist_raw / 50.0, 0.0, 1.0)
            hz_vel = target_hazard.get_velocity()
            v_hz_fwd = math.sqrt(hz_vel.x ** 2 + hz_vel.y ** 2) * math.cos(
                math.radians(target_hazard.get_transform().rotation.yaw - ego_yaw))
            hazard_v = np.clip((v_hz_fwd - v_ego_raw) / 30.0, -1.0, 1.0)

            hazard_brake = 0.0
            if hasattr(target_hazard, 'get_light_state') and (
                    target_hazard.get_light_state() & carla.VehicleLightState.Brake):
                hazard_brake = 1.0
        else:
            hazard_dist = 1.0
            hazard_v = 0.0
            hazard_brake = 0.0

        obs = np.array([v_ego, a_ego, lat_dev, heading_dev, hazard_dist, hazard_v, hazard_brake, tl_state, tl_dist,
                        hazard_type_val], dtype=np.float32)
        return obs

    def _get_upcoming_traffic_light(self, ego_trans, max_distance=40.0):
        """【修复】通过遍历和夹角计算，提前 40 米察觉前方红灯"""
        ego_loc = ego_trans.location
        ego_fwd = ego_trans.get_forward_vector()

        closest_dist = max_distance
        tl_state = 0.0
        tl_dist = 1.0

        lights = self.world.get_actors().filter('*traffic_light*')
        for light in lights:
            light_loc = light.get_transform().location
            vec_to_light = light_loc - ego_loc
            dist = math.sqrt(vec_to_light.x ** 2 + vec_to_light.y ** 2)

            if dist < closest_dist:
                # 归一化向量计算点乘，判断是否在正前方 30 度视角内
                dot = (vec_to_light.x * ego_fwd.x + vec_to_light.y * ego_fwd.y) / dist
                if dot > 0.866:  # cos(30)
                    if light.get_state() in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                        closest_dist = dist
                        tl_state = 1.0

        if tl_state == 1.0:
            tl_dist = np.clip(closest_dist / 50.0, 0.0, 1.0)

        return tl_state, tl_dist

    def _find_nearest_hazard(self, ego_trans, max_distance=50.0):
        """【高级修复】基于运动学投影的轨迹预瞄，完美克制鬼探头"""
        ego_loc = ego_trans.location
        ego_vel = self.ego_vehicle.get_velocity()
        v_ego_raw = max(math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2), 0.1)  # 避免除以0

        curr_wp = self.map.get_waypoint(ego_loc)

        # 生成未来 50 米的路径点 (每 2.5 米取一个点)
        future_wps = [curr_wp]
        for _ in range(int(max_distance / 2.5)):
            next_wps = curr_wp.next(2.5)
            if not next_wps: break
            curr_wp = next_wps[0]
            future_wps.append(curr_wp)

        closest_dist = float('inf')
        target_hazard = None
        hazard_type = 0.0

        hazards = list(self.world.get_actors().filter('*vehicle*')) + list(self.world.get_actors().filter('*walker*'))

        for hz in hazards:
            if hz.id == self.ego_vehicle.id: continue

            hz_loc = hz.get_location()
            hz_vel = hz.get_velocity()  # 【新增】获取障碍物速度向量

            # 绝对距离过滤
            if ego_loc.distance(hz_loc) > max_distance + 10.0: continue

            is_walker = 'walker' in hz.type_id
            margin = 3.5 if is_walker else 2.5

            for i, wp in enumerate(future_wps):
                path_dist = i * 2.5

                # 【核心改造：时空交汇预测】
                # 计算自车到达这个轨迹点大约需要多少时间
                time_to_reach = path_dist / v_ego_raw

                # 预测在那个时间点，障碍物会走到哪里 (简单的匀速直线预测)
                predicted_hz_x = hz_loc.x + hz_vel.x * time_to_reach
                predicted_hz_y = hz_loc.y + hz_vel.y * time_to_reach

                wp_loc = wp.transform.location
                # 计算障碍物【未来位置】与自车【未来轨迹】的距离
                predicted_dist = math.sqrt((predicted_hz_x - wp_loc.x) ** 2 + (predicted_hz_y - wp_loc.y) ** 2)

                if predicted_dist < margin:
                    if path_dist < closest_dist:
                        closest_dist = path_dist
                        target_hazard = hz
                        hazard_type = 1.0 if is_walker else 0.5
                    break

        return target_hazard, hazard_type, closest_dist

    def _spawn_background_traffic(self, num_vehicles=20, num_walkers=15):
        """【修复】为行人挂载 AI Controller 并设置行走速度"""
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        # 1. 车辆
        v_bps = self.blueprint_library.filter('vehicle.*')
        for i in range(min(num_vehicles, len(spawn_points))):
            npc = self.world.try_spawn_actor(random.choice(v_bps), spawn_points[i])
            if npc:
                npc.set_autopilot(True, self.tm.get_port())
                self.npc_list.append(npc)

        # 2. 行人与 AI
        w_bps = self.blueprint_library.filter('walker.pedestrian.*')
        controller_bp = self.blueprint_library.find('controller.ai.walker')

        walkers_to_start = []
        for _ in range(num_walkers):
            spawn_loc = self.world.get_random_location_from_navigation()
            if spawn_loc:
                walker = self.world.try_spawn_actor(random.choice(w_bps), carla.Transform(spawn_loc))
                if walker:
                    self.npc_list.append(walker)
                    # 必须挂载控制器
                    controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                    if controller:
                        self.npc_list.append(controller)
                        walkers_to_start.append(controller)

        # CARLA 物理引擎特性：行人生成后必须先跑几帧 tick 稳定物理，再启动控制器，否则会卡在地下
        for _ in range(5):
            self.world.tick()

        # 激活行人乱跑
        for controller in walkers_to_start:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.2 + random.random())  # 随机步行速度 1.2~2.2 m/s

    def _clear_actors(self):
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None
        for npc in self.npc_list:
            if npc.is_alive:
                npc.destroy()
        self.npc_list.clear()

    def _clear_all_zombies(self):
        for actor_type in ['*vehicle*', '*sensor*', '*walker*', '*controller*']:
            for a in self.world.get_actors().filter(actor_type):
                a.destroy()

    def _on_collision(self, event):
        self.has_collided = True

    def close(self):
        self._clear_actors()
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass