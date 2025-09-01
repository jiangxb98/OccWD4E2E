import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.draw import polygon

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])  # 栅格大小，即BEV的分辨率是dx = [0.5, 0.5, 20.0]  # 米/像素
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])  # 计算栅格中心到坐标原点的偏移量: 边界偏移 = 最小边界 + 分辨率/2
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])  # 栅格数量

    return dx, bx, nx

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    # 提取各轴的分辨率参数
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    # 计算各轴的起始位置（栅格中心到世界原点的偏移量）
    # 公式：起始位置 = 最小边界 + 分辨率/2
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    # 计算各轴的栅格数量（像素数量）
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


class Cost_Function(nn.Module):
    def __init__(self, cfg):
        super(Cost_Function, self).__init__()

        self.safetycost = SafetyCost(cfg)
        self.headwaycost = HeadwayCost(cfg)
        # self.lrdividercost = LR_divider(cfg)
        self.comfortcost = Comfort(cfg)
        self.progresscost = Progress(cfg)
        self.rulecost = Rule(cfg)
        self.costvolume = Cost_Volume(cfg)
        self.comfortcost_navsim = Comfort_Navsim(cfg)

    def forward(self, cost_volume, trajs, instance_occupancy, drivable_area):
        '''
        trajs: torch.Tensor (B, N, 2)
        cost_volume: torch.Tensor (B, 200, 200)
        instance_occupancy: torch.Tensor(B, 200, 200)   instance_occupied=1
        drivable_area: torch.Tensor(B, 200, 200)        driveable_surface=1
        '''
        safetycost = torch.clamp(self.safetycost(trajs, instance_occupancy), 0, 100)                 # penalize overlap with instance_occupancy
        headwaycost = torch.clamp(self.headwaycost(trajs, instance_occupancy, drivable_area), 0, 100)# penalize overlap with front instance (10m)
        # lrdividercost = torch.clamp(self.lrdividercost(trajs, lane_divider), 0, 100)               # penalize distance with lane
        # comfortcost = torch.clamp(self.comfortcost(trajs), 0, 100)                                   # penalize high accelerations (lateral, longitudinal, jerk)
        # progresscost = torch.clamp(self.progresscost(trajs), -100, 100)                              # L2 loss
        rulecost = torch.clamp(self.rulecost(trajs, drivable_area), 0, 100)                          # penalize overlap with out of drivable_area
        costvolume = torch.clamp(self.costvolume(trajs, cost_volume), 0, 100)                        # sample on costvolume

        cost_fo = safetycost + headwaycost + costvolume + rulecost
        # cost_fc = progresscost

        return cost_fo
    

    def forward_sim(self, trajs, instance_occupancy, drivable_area, sim_reward_nums=1, vel_steering=None):

        '''
        trajs: torch.Tensor (B, N, 2)
        instance_occupancy: torch.Tensor(B, 200, 200)   instance_occupied=1
        drivable_area: torch.Tensor(B, 200, 200)        driveable_surface=1
        sim_reward_nums: int
        vel_steering: B,4 (vx, vy, v_yaw, steering)
        '''
        # 是否发生碰撞
        safetycost = torch.clamp(self.safetycost(trajs, instance_occupancy), 0, 100)                 # penalize overlap with instance_occupancy

        # 是否违规
        rulecost = torch.clamp(self.rulecost(trajs, drivable_area), 0, 100)                          # penalize overlap with out of drivable_area

        # 是否前进
        progresscost = self.progresscost(trajs)                         # L2 loss

        # 前方是否有车
        headwaycost = torch.clamp(self.headwaycost(trajs, instance_occupancy, drivable_area), 0, 100)# penalize overlap with front instance (10m)
        
        # 是否舒适
        comfortcost = self.comfortcost_navsim(trajs, vel_steering[:,:2], vel_steering[:,2], vel_steering[:,3])                                   # penalize high accelerations (lateral, longitudinal, jerk)


        if sim_reward_nums == 1:
            cost_fo = safetycost + rulecost
        elif sim_reward_nums == 5:
            cost_fo = [safetycost, rulecost, progresscost, headwaycost, comfortcost]
        else:
            raise ValueError(f'sim_reward_nums must be 1 or 3, but got {sim_reward_nums}')
        

        return cost_fo


class BaseCost(nn.Module):
    def __init__(self, grid_conf):
        super(BaseCost, self).__init__()
        self.grid_conf = grid_conf
        # 生成栅格坐标系的基本参数
        dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx,requires_grad=False)
        self.bx = nn.Parameter(bx,requires_grad=False)
        
        # 计算BEV图像的一些基本参数
        _,_, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
        )

        # 车辆尺寸参数（米）
        self.W = 1.85
        self.H = 4.084

    def get_origin_points(self, lambda_=0):
        """
        获取车辆轮廓的栅格坐标点
        
        Args:
            lambda_: float - 安全距离膨胀系数（米）
        Returns:
            rc: torch.Tensor - 车辆轮廓的栅格坐标点
        """
        W = self.W
        H = self.H

        # 车辆轮廓：定义车辆四个角点的世界坐标（相对于车辆中心）
        pts = np.array([
            [-H / 2. + 0.5 - lambda_, W / 2. + lambda_],
            [H / 2. + 0.5 + lambda_, W / 2. + lambda_],
            [H / 2. + 0.5 + lambda_, -W / 2. - lambda_],
            [-H / 2. + 0.5 - lambda_, -W / 2. - lambda_],
        ])  # [lidar_y, lidar_x]

        #  将世界坐标转换为栅格坐标
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())   # [bev_w, bev_h]
        # pts[:, [0, 1]] = pts[:, [1, 0]] # [bev_h, bev_w]

        # 使用多边形填充算法获取车辆轮廓内的所有栅格点
        # rr 表示车宽， cc 表示车长
        rr , cc = polygon(pts[:,1], pts[:,0])   # [bev_h, bev_w]
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)  # [bev_h, bev_w]
        return torch.from_numpy(rc).to(device=self.bx.device) # (27,2)

    def get_points(self, trajs, lambda_=0):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        return:
        List[ torch.Tensor<int> (B, N), torch.Tensor<int> (B, N)]
        '''
        rc = self.get_origin_points(lambda_)    # [bev_h, bev_w], rr表示车宽，cc表示车长
        B, N, _ = trajs.shape         # delta_[lidar_x, lidar_y]

        # 将轨迹坐标转换为栅格偏移量
        trajs = trajs.view(B, N, 1, 2) / self.dx  # delta_[bev_h, bev_w]
        # trajs[:,:,:,:,[0,1]] = trajs[:,:,:,:,[1,0]]
        # 将偏移量加到车辆轮廓坐标上
        # 因为是lidar坐标系，所以x表示向右为正，是车宽，y表示向前为正，是车长方向
        trajs = trajs + rc  # [bev_h, bev_w]

        rr = trajs[:,:,:,0].long()
        rr = torch.clamp(rr, 0, self.bev_dimension[0] - 1)

        cc = trajs[:,:,:,1].long()
        cc = torch.clamp(cc, 0, self.bev_dimension[1] - 1)

        return rr, cc

    def compute_area(self, instance_occupancy, trajs, ego_velocity=None, _lambda=0):
        '''
        instance_occupancy: torch.Tensor<float> (B, 200, 200)
        trajs: torch.Tensor<float> (B, N, 2)
        ego_velocity: torch.Tensor<float> (B, N)
        '''
        # 将安全距离从米转换为栅格像素数
        _lambda = int(_lambda / self.dx[0])
        # 获取轨迹点对应的栅格坐标
        rr, cc = self.get_points(trajs, _lambda)    # [bev_h, bev_w]
        B, N, _ = trajs.shape

        # 如果没有提供速度，默认为1
        if ego_velocity is None:
            ego_velocity = torch.ones((B,N), device=trajs.device)

        ii = torch.arange(B)

        # 计算轨迹点在障碍物栅格图中的重叠代价
        subcost = instance_occupancy[ii[:, None, None], rr, cc].sum(dim=-1)
        subcost = subcost * ego_velocity  # 速度越快，重叠代价越大

        return subcost

    def discretize(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        '''
        B, N,  _ = trajs.shape # delta_[lidar_x, lidar_y]

        # 提取X和Y坐标
        xx, yy = trajs[:,:,0], trajs[:,:,1] # delta_[lidar_x, lidar_y]

        # discretize
        # 将坐标离散化为栅格索引
        xi = ((xx - self.bx[0]) / self.dx[0]).long()
        xi = torch.clamp(xi, 0, self.bev_dimension[0]-1)    # bev_h

        yi = ((yy - self.bx[1]) / self.dx[1]).long()
        yi = torch.clamp(yi,0, self.bev_dimension[1]-1)     # bev_w

        return xi, yi

    def evaluate(self, trajs, C):
        '''
        评估轨迹在代价图上的值
            trajs: torch.Tensor<float> (B, N, 2)   N: sample number
            C: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape

        ii = torch.arange(B)

        Syi, Sxi = self.discretize(trajs)

        CS = C[ii, Syi, Sxi]
        return CS

class Cost_Volume(BaseCost):
    def __init__(self, cfg):
        super(Cost_Volume, self).__init__(cfg)

        self.factor = 100.

    def forward(self, trajs, cost_volume):
        '''
        cost_volume: torch.Tensor<float> (B, 200, 200)
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        '''

        cost_volume = torch.clamp(cost_volume, 0, 1000)

        return self.evaluate(trajs, cost_volume) * self.factor

class Rule(BaseCost):
    def __init__(self, cfg):
        """规则合规性代价函数 - 评估轨迹是否违反交通规则"""
        super(Rule, self).__init__(cfg)

        self.factor = 5 # 整体权重系数

    def forward(self, trajs, drivable_area):
        '''
            trajs: torch.Tensor<float> (B, N, 2)   N: sample number
            drivable_area: torch.Tensor<float> (B, 200, 200)
        '''
        B, _,  _ = trajs.shape

        dangerous_area = torch.logical_not(drivable_area).float()
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(dangerous_area[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        # 计算轨迹在危险区域内的重叠代价
        subcost = self.compute_area(dangerous_area, trajs)

        return subcost * self.factor


class SafetyCost(BaseCost):
    def __init__(self, cfg):
        super(SafetyCost, self).__init__(cfg)
        self.w = nn.Parameter(torch.tensor([1.,1.]),requires_grad=False)

        self._lambda = 1. # 动态安全距离系数
        self.factor = 0.1 # 整体权重系数

    def forward(self, trajs, instance_occupancy):
        '''
        计算轨迹的安全性代价
        
        Args:
            trajs: torch.Tensor<float> (B, N, 2) - 轨迹点坐标, N为采样点数
            instance_occupancy: torch.Tensor<float> (B, 200, 200) - 障碍物占用栅格图
        Returns:
            subcost: 安全性代价，值越大表示越不安全
        '''
        B, N, _ = trajs.shape
        ego_velocity = torch.sqrt((trajs ** 2).sum(axis=-1)) / 0.5  # B,N

        # o_c(tau, t, 0)
        # # 子代价1：静态碰撞代价 - 不考虑速度的碰撞风险
        subcost1 = self.compute_area(instance_occupancy, trajs)
        # o_c(tau, t, lambda) x v(tau, t)
        # 子代价2：动态碰撞代价 - 考虑速度的碰撞风险（速度越快，安全距离越大，碰撞风险越大）
        subcost2 = self.compute_area(instance_occupancy, trajs, ego_velocity, self._lambda)

        subcost = subcost1 * self.w[0] + subcost2 * self.w[1]

        return subcost * self.factor


class HeadwayCost(BaseCost):
    """车距保持代价函数 - 评估与前车的安全距离"""
    def __init__(self, cfg):
        super(HeadwayCost, self).__init__(cfg)
        self.L = 10  # 期望保持的纵向距离10（米）
        self.factor = 1. # 整体权重系数

    def forward(self, trajs, instance_occupancy, drivable_area):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        instance_occupancy: torch.Tensor<float> (B, 200, 200)
        drivable_area: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape
        # 只在可行驶区域内考虑障碍物
        instance_occupancy_ = instance_occupancy * drivable_area  # B,H,W
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(instance_occupancy_[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        tmp_trajs = trajs.clone()
        # 注意这里的坐标系，x是横向，y是纵向（因为是nuScenes的LiDAR坐标系）
        tmp_trajs[:,:,1] = tmp_trajs[:,:,1]+self.L  # 将轨迹向前延伸L米，模拟前车位置

        subcost = self.compute_area(instance_occupancy_, tmp_trajs)

        return subcost * self.factor

class LR_divider(BaseCost):
    """车道线偏离代价函数 - 评估轨迹与车道线的距离"""
    def __init__(self, cfg):
        super(LR_divider, self).__init__(cfg)
        self.L = 1 # 与车道线保持的最小距离（米）
        self.factor = 10.

    def forward(self, trajs, lane_divider):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        lane_divider: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape
        # 将轨迹坐标离散化到栅格坐标系
        xx, yy = self.discretize(trajs) # [bev_h, bev_w]
        xy = torch.stack([xx,yy],dim=-1) # (B, N, 2)  [bev_h, bev_w]

        # lane divider
         # 计算每个轨迹点到车道线的最短距离
        res1 = []
        for i in range(B):
            # 找到车道线像素的坐标
            index = torch.nonzero(lane_divider[i]) # (n, 2)
            if len(index) != 0:
                xy_batch = xy[i].view(N, 1, 2)
                # 计算轨迹点到所有车道线像素的距离
                distance = torch.sqrt((((xy_batch - index) * reversed(self.dx))**2).sum(dim=-1)) # (N, n)
                distance,_ = distance.min(dim=-1) # (N) - 取最短距离
                # 如果距离大于阈值L，则代价为0
                index = distance > self.L
                distance = (self.L - distance) ** 2
                distance[index] = 0
            else:
                distance = torch.zeros((N),device=trajs.device)
            res1.append(distance)
        res1 = torch.stack(res1, dim=0)

        return res1 * self.factor


class Comfort(BaseCost):
    """舒适性代价函数 - 评估轨迹的乘坐舒适度"""
    def __init__(self, cfg):
        super(Comfort, self).__init__(cfg)

        self.c_lat_acc = 3 # m/s2 # 横向加速度阈值 (m/s²)
        self.c_lon_acc = 3 # m/s2 # 纵向加速度阈值 (m/s²)
        self.c_jerk = 1 # m/s3      # 加速度变化率阈值 (m/s³)

        self.factor = 0.1

    def forward(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        '''
        B, N, _ = trajs.shape
        # 计算速度（基于位移和时间间隔0.5s）
        lateral_velocity = trajs[:,:,0] / 0.5   # 横向速度
        longitudinal_velocity = trajs[:,:,1] / 0.5  # 纵向速度
        # 计算加速度
        lateral_acc = lateral_velocity / 0.5    # B,N  # 横向加速度
        longitudinal_acc = longitudinal_velocity / 0.5  # B,N  # 纵向加速度

        # 计算加加速度 jerk
        ego_velocity = torch.sqrt((trajs ** 2).sum(dim=-1)) / 0.5  # 总速度
        ego_acc = ego_velocity / 0.5  # 总加速度
        ego_jerk = ego_acc / 0.5    # B,N  # 加加速度

        subcost = torch.zeros((B, N), device=trajs.device)

        # 计算超出阈值的代价（使用clamp限制最大值）
        lateral_acc = torch.clamp(torch.abs(lateral_acc) - self.c_lat_acc, 0, 30)
        subcost += lateral_acc ** 2
        longitudinal_acc = torch.clamp(torch.abs(longitudinal_acc) - self.c_lon_acc, 0, 30)
        subcost += longitudinal_acc ** 2
        ego_jerk = torch.clamp(torch.abs(ego_jerk) - self.c_jerk, 0, 20)
        subcost += ego_jerk ** 2

        return subcost * self.factor


class Comfort_Navsim(BaseCost):
    """舒适性代价函数 - 基于两点状态评估舒适度"""
    def __init__(self, cfg):
        super(Comfort_Navsim, self).__init__(cfg)

        # 参考navsim的舒适性阈值
        self.max_lat_acc = 4.89      # [m/s²] 横向加速度阈值
        self.max_lon_acc = 2.40      # [m/s²] 纵向加速度上限
        self.min_lon_acc = -4.05     # [m/s²] 纵向加速度下限
        self.max_jerk = 8.37         # [m/s³] 总加加速度阈值
        self.max_lon_jerk = 4.13     # [m/s³] 纵向加加速度阈值
        self.max_yaw_accel = 1.93    # [rad/s²] 偏航角加速度阈值
        self.max_yaw_rate = 0.95     # [rad/s] 偏航角速度阈值

        self.dt = 0.5  # 时间间隔为0.5s
        self.dt_half = 0.25  # 时间间隔为0.25s

    def forward(self, trajs, initial_velocities, initial_yaw_rate, initial_steering_angle):
        '''
        trajs: torch.Tensor<float> (B, 1, 2) - 第0.5s时刻的[x, y]坐标
        initial_velocities: torch.Tensor<float> (B, 2) - 当前帧的[x方向速度, y方向速度]
        initial_yaw_rate: torch.Tensor<float> (B,) - 当前帧的角速度
        initial_steering_angle: torch.Tensor<float> (B,) - 当前帧的转向角
        返回: torch.Tensor<float> (B,) - 每个轨迹的舒适性得分 {0, 1}
        '''
        B, N, _ = trajs.shape
        initial_velocities = initial_velocities.repeat(1, N, 1)
        initial_yaw_rate = initial_yaw_rate.repeat(1, N, 1).squeeze(-1)
        initial_steering_angle = initial_steering_angle.repeat(1, N, 1).squeeze(-1)
        
        # 计算终点状态
        final_positions = trajs  # (B, N, 2) - [x横向, y纵向]
        final_heading = torch.atan2(final_positions[:, :, 1], final_positions[:, :, 0])  # 修正：使用x/y计算航向角  (B, N)
        
        # 计算平均速度和加速度
        avg_velocities = final_positions / self.dt  # 平均速度 [x横向, y纵向]
        accelerations = (avg_velocities - initial_velocities) / self.dt  # 加速度
        
        # 计算角速度和角加速度
        heading_change = final_heading - initial_steering_angle
        avg_yaw_rate = heading_change / self.dt  # 平均角速度
        yaw_acceleration = (avg_yaw_rate - initial_yaw_rate) / self.dt  # 角加速度
        
        # 计算加加速度（假设线性变化）
        initial_accelerations = torch.zeros_like(accelerations)  # 假设初始加速度为0
        jerks = (accelerations - initial_accelerations) / self.dt  # 加加速度
        
        # 舒适性检查
        comfort_scores = torch.ones((B, N), device=trajs.device)
        
        # 1. 纵向加速度检查（y方向，车辆前进方向）
        lon_acc_ok = (accelerations[:, :, 1] >= self.min_lon_acc) & \
                     (accelerations[:, :, 1] <= self.max_lon_acc)
        comfort_scores *= lon_acc_ok.float()
        
        # 2. 横向加速度检查（x方向，车辆左右方向）
        lat_acc_ok = torch.abs(accelerations[:, :, 0]) <= self.max_lat_acc
        comfort_scores *= lat_acc_ok.float()
        
        # 3. 总加加速度检查
        total_jerk = torch.sqrt((jerks ** 2).sum(dim=-1))
        total_jerk_ok = total_jerk <= self.max_jerk
        comfort_scores *= total_jerk_ok.float()
        
        # 4. 纵向加加速度检查（y方向）
        lon_jerk_ok = torch.abs(jerks[:, :, 1]) <= self.max_lon_jerk
        comfort_scores *= lon_jerk_ok.float()
        
        # # 5. 角加速度检查
        # yaw_accel_ok = torch.abs(yaw_acceleration) <= self.max_yaw_accel
        # comfort_scores *= yaw_accel_ok.float()
        
        # # 6. 角速度检查（使用平均角速度）
        # yaw_rate_ok = torch.abs(avg_yaw_rate) <= self.max_yaw_rate
        # comfort_scores *= yaw_rate_ok.float()
        
        return comfort_scores


    def get_detailed_metrics(self, trajs, initial_velocities, initial_yaw_rate, initial_steering_angle):
        """返回详细的舒适性指标，用于调试"""
        B, _, _ = trajs.shape
        
        final_positions = trajs.squeeze(1)  # [x横向, y纵向]
        final_heading = torch.atan2(final_positions[:, 0], final_positions[:, 1])  # 修正航向角计算
        
        avg_velocities = final_positions / self.dt
        accelerations = (avg_velocities - initial_velocities) / self.dt
        
        heading_change = final_heading - initial_steering_angle
        avg_yaw_rate = heading_change / self.dt
        yaw_acceleration = (avg_yaw_rate - initial_yaw_rate) / self.dt
        
        initial_accelerations = torch.zeros_like(accelerations)
        jerks = (accelerations - initial_accelerations) / self.dt
        total_jerk = torch.sqrt((jerks ** 2).sum(dim=-1))
        
        return {
            'lon_acc': accelerations[:, 1],      # y方向，纵向加速度
            'lat_acc': accelerations[:, 0],      # x方向，横向加速度
            'total_jerk': total_jerk,
            'lon_jerk': jerks[:, 1],             # y方向，纵向加加速度
            'lat_jerk': jerks[:, 0],             # x方向，横向加加速度
            'yaw_accel': yaw_acceleration,
            'avg_yaw_rate': avg_yaw_rate,
            'final_positions': final_positions,  # [x横向, y纵向]
            'avg_velocities': avg_velocities,    # [x横向, y纵向]
            'final_heading': final_heading
        }

class Progress(BaseCost):
    """进度代价函数 - 评估轨迹的前进进度，鼓励车辆向前行驶，同时考虑与目标点的距离。"""

    def __init__(self, cfg):
        super(Progress, self).__init__(cfg)
        self.factor = 0.5

    # def forward(self, trajs):
    #     '''
    #     trajs: torch.Tensor<float> (B, N, 2)
    #     target_points: torch.Tensor<float> (B, 2)
    #     '''
    #     # 目标点设为原点（可根据需要修改）
    #     target_points = torch.zeros_like(trajs[:, 0, :])    # B,2
    #     B, N,  _ = trajs.shape
    #     # 子代价1：纵向位移（前进为正） -subcost1就是惩罚倒车
    #     subcost1 = trajs[:,:,1]
    #     # 子代价2：与目标点的距离（如果有目标点）
    #     if target_points.sum() < 0.5:
    #         subcost2 = 0
    #     else:
    #         target_points = target_points.unsqueeze(1)
    #         subcost2 = ((trajs - target_points) ** 2).sum(dim=-1)  #
    #     # 最终代价：距离代价 - 前进奖励
    #     return (subcost2 - subcost1) * self.factor

    def forward(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        target_points: torch.Tensor<float> (B, 2)
        '''
        B, N,  _ = trajs.shape
        # 距离代价
        # 如果都小于5m就是1，如果是大于5m，那就是按照比例来算
        # 如果最大前进距离 > 5.0米，按相对比例归一化
        # max_progress = trajs[:,:,1].max()
        # if max_progress > 5.0:
        #     return trajs[:,:,1] / max_progress
        # else:
        #     return torch.ones_like(trajs[:,:,1])
        return trajs[:,:,1]