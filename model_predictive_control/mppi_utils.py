import heapq

import numpy as np
import torch

from utilities.misc_utils import DEFAULT_DTYPE


def snap_to_grid(point, grid_step):
    snapped_x = round(round(point[0] / grid_step) * grid_step, 10)
    snapped_y = round(round(point[1] / grid_step) * grid_step, 10)
    return snapped_x, snapped_y


def snap_to_grid_torch(pt, grid_step, boundaries):
    pt = pt.clone()
    pt[:, 0] = torch.round((pt[:, 0] - boundaries[0]) / grid_step)
    pt[:, 1] = torch.round(-(pt[:, 1] - boundaries[3]) / grid_step)
    return pt.to(torch.int)


def unsnap_to_grid_torch(idx, grid_step, boundaries):
    px = idx[0] * grid_step + boundaries[0]
    py = boundaries[3] - idx[1] * grid_step

    return px, py


def simple_collision(point, obstacles):
    px, py = point
    for xmin, xmax, ymin, ymax in obstacles:
        if (xmin <= px <= xmax) and (ymin <= py <= ymax):
            return True
    return False


def get_rotated_corners(x, y, w, h, theta):
    dx = w / 2
    dy = h / 2

    corners = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy)
    ]

    rotated_corners = []
    for corner in corners:
        cx = corner[0] * np.cos(theta) - corner[1] * np.sin(theta)
        cy = corner[0] * np.sin(theta) + corner[1] * np.cos(theta)
        rotated_corners.append((x + cx, y + cy))

    return rotated_corners


def check_overlap(corners1, corners2):
    # Use Separating Axis Theorem (SAT) to check for overlap
    axes = get_axes(corners1) + get_axes(corners2)

    for axis in axes:
        if not projection_overlap(corners1, corners2, axis):
            return False
    return True


def get_axes(corners):
    axes = []
    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        axis = (-edge[1], edge[0])  # Perpendicular vector
        length = np.sqrt(axis[0] ** 2 + axis[1] ** 2)
        axes.append((axis[0] / length, axis[1] / length))
    return axes


def projection_overlap(corners1, corners2, axis):
    def project(corners, axis):
        return [corner[0] * axis[0] + corner[1] * axis[1] for corner in corners]

    p1 = project(corners1, axis)
    p2 = project(corners2, axis)

    if max(p1) < min(p2) or max(p2) < min(p1):
        return False
    return True


def coll_det(point, obstacles, robot_dims=(0.33, 0.15), boundary=(-3, 1, -1.4, 0.4)):
    # (0.4, 0.27)
    # (-1.0, 1.0, -1.0, 4.0)):
    # boundary is (x_min, x_max, y_min, y_max)
    robot_x, robot_y, robot_theta = point
    robot_w, robot_h = robot_dims

    # Calculate robot bounding box corners
    robot_corners = get_rotated_corners(robot_x, robot_y, robot_w, robot_h, robot_theta)
    # print(robot_corners)

    for corner in robot_corners:
        if boundary[0] > corner[0] or boundary[1] < corner[0] or boundary[2] > corner[1] or boundary[3] < corner[1]:
            # print("Border", str(corner))
            # print("Border")
            return True

    for obs_x_min, obs_x_max, obs_y_min, obs_y_max in obstacles:
        obs_x = (obs_x_max + obs_x_min) / 2
        obs_y = (obs_y_max + obs_y_min) / 2
        obs_w = obs_x_max - obs_x_min
        obs_h = obs_y_max - obs_y_min

        # Calculate obstacle bounding box corners (assuming no rotation)
        obstacle_corners = get_rotated_corners(obs_x, obs_y, obs_w, obs_h, 0)

        # Check if there is overlap between the robot and the obstacle
        if check_overlap(robot_corners, obstacle_corners):
            # print("obstacle")
            return True

    return False


def l2_dist(a, b):
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dist_heuristic(a, b, obstacles=[], k=0):
    dist = l2_dist(a, b)
    if k == 0 or len(obstacles) < 1:
        return dist

    penalty = 0
    for obstacle in obstacles:
        distance_to_obstacle = l2_dist(a, obstacle)
        penalty += 1 / (1 + distance_to_obstacle)

    return dist + k * penalty


# print(dist_heuristic((1.5982203824585355, -0.38906994686732577),(0.0,3.0)))

########
def snap_to_grid(point, grid_step):
    snapped_x = round(round(point[0] / grid_step) * grid_step, 10)
    snapped_y = round(round(point[1] / grid_step) * grid_step, 10)
    return (snapped_x, snapped_y)


def simple_collision(point, obstacles):
    px, py = point
    for obs_x_min, obs_x_max, obs_y_min, obs_y_max in obstacles:
        ox, oy = (obs_x_max + obs_x_min) / 2, (obs_y_max + obs_y_min) / 2
        half_w, half_h = (obs_x_max - obs_x_min) / 2, (obs_y_max - obs_y_min) / 2
        if (ox - half_w <= px <= ox + half_w) and (oy - half_h <= py <= oy + half_h):
            return True
    return False


def fill_grid(goal, boundary, grid_step=0.1, obstacles=()):
    # BFS from goal
    h_val = {}
    unassigned = set()
    obs_loc = set()
    goal = snap_to_grid(goal, grid_step)
    # h_val[goal]=0
    gaits = [[grid_step, 0], [0, grid_step], [0, -grid_step], [-grid_step, 0], \
             [grid_step, grid_step], [-grid_step, grid_step], [grid_step, -grid_step], [-grid_step, -grid_step]]

    for i in np.arange(boundary[0], boundary[1] + grid_step, grid_step):
        for j in np.arange(boundary[2], boundary[3] + grid_step, grid_step):
            current = snap_to_grid((i, j), grid_step=grid_step)
            if simple_collision((i, j), obstacles):
                h_val[current] = np.inf
                obs_loc.add(current)
                # all_points.remove()
            else:
                unassigned.add(current)

    open_list = []
    heapq.heappush(open_list, (0, goal))

    while unassigned:
        if not open_list:
            break
        node = heapq.heappop(open_list)

        current = snap_to_grid(node[1], grid_step)
        if current in obs_loc or current not in unassigned:
            continue

        h_val[current] = node[0]
        unassigned.remove(current)

        for k in range(len(gaits)):
            gait_num = gaits[k]
            neighbor = (current[0] + gait_num[0], current[1] + gait_num[1])

            if k < 4:
                cost = 1
            else:
                cost = np.sqrt(2)

            if (not simple_collision((neighbor[0], neighbor[1]), obstacles)
                    and neighbor[0] not in [boundary[0], boundary[1]]
                    and neighbor[1] not in [boundary[2], boundary[3]]):
                dist_to_bound = min([
                    abs(neighbor[0] - boundary[0]),
                    abs(neighbor[0] - boundary[1]),
                    abs(neighbor[1] - boundary[2]),
                    abs(neighbor[1] - boundary[3])
                ])

                dist_to_obs = []
                for xmin, xmax, ymin, ymax in obstacles:
                    if xmin <= neighbor[0] <= xmax:
                        dist_to_obs.append(min(abs(neighbor[1] - ymin), abs(neighbor[1] - ymax)))
                    elif ymin <= neighbor[1] <= ymax:
                        dist_to_obs.append(min(abs(neighbor[0] - xmin), abs(neighbor[0] - xmax)))
                    else:
                        dist_to_obs.append(min([
                            np.sqrt((neighbor[0] - xmin) ** 2 + (neighbor[1] - ymin) ** 2),
                            np.sqrt((neighbor[0] - xmin) ** 2 + (neighbor[1] - ymax) ** 2),
                            np.sqrt((neighbor[0] - xmax) ** 2 + (neighbor[1] - ymin) ** 2),
                            np.sqrt((neighbor[0] - xmax) ** 2 + (neighbor[1] - ymax) ** 2),
                        ]))
                dist_to_obs = min(dist_to_obs) if dist_to_obs else np.inf
                dist = min(dist_to_obs, dist_to_bound)
                dist = max(dist, 0.0)
                penalty = 50 * np.exp(-0.7 * dist)
            else:
                penalty = np.inf

            cost += penalty

            heapq.heappush(open_list, (node[0] + cost, neighbor))

    return h_val


def wave_heuristic(start, goal, grid_step=0.1, obstacles=()):
    start = snap_to_grid(start, grid_step)  # now in the format (x,y)
    goal = snap_to_grid(goal, grid_step)

    if simple_collision(start, obstacles):
        return 1000

    gaits = [[grid_step, 0], [0, grid_step], [0, -grid_step], [-grid_step, 0], \
             [grid_step, grid_step], [-grid_step, grid_step], [grid_step, -grid_step], [-grid_step, -grid_step]]

    open_list = []

    # came_from = {}
    g_score = {start: 0}
    f_score = {start: dist_heuristic(start, goal)}
    heapq.heappush(open_list, (f_score[start], start))
    closed_list = set()  # only used to check if already visited using kd tree

    while open_list:
        # Get the node with the lowest f_score value
        node = heapq.heappop(open_list)
        current = node[1]
        closed_list.add(current)

        if simple_collision(current, obstacles):
            continue

        # If the goal is reached, reconstruct and return the path
        if current == goal:
            return g_score[current]

        for k in range(len(gaits)):
            gait_num = gaits[k]
            neighbor = snap_to_grid((current[0] + gait_num[0], current[1] + gait_num[1]), grid_step)

            if neighbor in closed_list:
                continue

            if k < 4:
                cost = 1
            else:
                cost = np.sqrt(2)

            tentative_g_score = g_score[current] + cost
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                # came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + dist_heuristic(neighbor[:2], goal, obstacles)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))


def wave_heuristic_dict_to_arr(h_val, boundary, obstacles, grid_step, device='cpu', dtype=DEFAULT_DTYPE):
    size_x = int((boundary[1] - boundary[0]) / grid_step) + 1
    size_y = int((boundary[3] - boundary[2]) / grid_step) + 1
    h_val_arr = np.zeros((size_x, size_y), dtype=np.float64)
    obs_val_arr = np.full((size_x, size_y), np.inf, dtype=np.float64)

    for i in np.arange(boundary[0], boundary[1] + grid_step, grid_step):
        for j in np.arange(boundary[2], boundary[3] + grid_step, grid_step):
            i, j = snap_to_grid((i, j), grid_step=grid_step)
            idx_x = round((i - boundary[0]) / grid_step)
            idx_y = round(-(j - boundary[3]) / grid_step)
            h_val_arr[idx_x, idx_y] = h_val[(i, j)]

            if (not simple_collision((i, j), obstacles)
                    and i not in [boundary[0], boundary[1]]
                    and j not in [boundary[2], boundary[3]]):
                dist_to_bound = min([
                    abs(i - boundary[0]),
                    abs(i - boundary[1]),
                    abs(j - boundary[2]),
                    abs(j - boundary[3])
                ])

                dist_to_obs = []
                for xmin, xmax, ymin, ymax in obstacles:
                    if xmin <= i <= xmax:
                        dist_to_obs.append(min(abs(j - ymin), abs(j - ymax)))
                    elif ymin <= j <= ymax:
                        dist_to_obs.append(min(abs(i - xmin), abs(i - xmax)))
                    else:
                        dist_to_obs.append(min([
                            np.sqrt((i - xmin) ** 2 + (j - ymin) ** 2),
                            np.sqrt((i - xmin) ** 2 + (j - ymax) ** 2),
                            np.sqrt((i - xmax) ** 2 + (j - ymin) ** 2),
                            np.sqrt((i - xmax) ** 2 + (j - ymax) ** 2),
                        ]))
                dist_to_obs = min(dist_to_obs) if dist_to_obs else np.inf
                dist = min(dist_to_obs, dist_to_bound) - 0.5
                dist = max(dist, 0.0)
                penalty = 10. / (dist ** 2 + 1e-6)

                obs_val_arr[idx_x, idx_y] = penalty

    # plot(h_val_arr, obs_val_arr, boundary, grid_step)

    h_val_arr = torch.from_numpy(h_val_arr).to(device).to(dtype)
    obs_val_arr = torch.from_numpy(obs_val_arr).to(device).to(dtype)

    return h_val_arr, obs_val_arr


def heuristic_dir(cost_grid, snapped_pt, num_steps=20):
    # snapped_x, snapped_y = snapped_pt.cpu().flatten().numpy().tolist()
    snapped_x, snapped_y = snapped_pt
    gaits = [
        [1, 0], [0, 1], [0, -1], [-1, 0],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ]

    open_list = [(snapped_x, snapped_y)]
    for n in range(num_steps):
        new_open_list = set()
        for x, y in open_list:
            for g in gaits:
                x_, y_ = x + g[0], y + g[1]
                if 0 <= x_ < cost_grid.shape[0] and 0 <= y_ < cost_grid.shape[1]:
                    new_open_list.add((x_, y_))
        open_list = new_open_list

    best_score, best_pt = 1e9, [0, 0]
    for x, y in open_list:
        score = cost_grid[x, y].item()
        if score < best_score:
            best_score = score
            best_pt = [x, y]

    return best_pt


def heuristic_dir2(cost_grid, snapped_pt, radius=20):
    if isinstance(snapped_pt, torch.Tensor):
        snapped_pt = snapped_pt.cpu().flatten().numpy().tolist()
    snapped_x, snapped_y = snapped_pt

    if isinstance(cost_grid, np.ndarray):
        cost_grid = torch.from_numpy(cost_grid)

    w, h = cost_grid.shape
    xmin = max(snapped_x - radius, 0)
    xmax = min(snapped_x + radius + 1, w)
    ymin = max(snapped_y - radius, 0)
    ymax = min(snapped_y + radius + 1, h)

    subgrid = cost_grid[xmin:xmax, ymin:ymax]
    flat_idx = torch.argmin(subgrid)
    best_pt = torch.unravel_index(flat_idx, subgrid.shape)
    best_pt = (best_pt[0] + xmin, best_pt[1] + ymin)

    return best_pt

def plot(h_val_arr, obs_val_arr, boundary, grid_step):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.imshow(h_val_arr.T)
    plt.colorbar()
    plt.title("dist")
    plt.show()

    obs_val_arr_cpy = obs_val_arr.copy()
    obs_val_arr_cpy2 = np.clip(obs_val_arr_cpy, 0., 500)

    x = np.linspace(boundary[1], boundary[0], h_val_arr.shape[1])
    y = np.linspace(boundary[2], boundary[3], h_val_arr.shape[0])
    X, Y = np.meshgrid(x, y)

    plt.imshow(obs_val_arr_cpy2.T)
    plt.colorbar()
    plt.title("obs")
    plt.show()
    plt.imshow((obs_val_arr_cpy2 + h_val_arr).T)
    plt.colorbar()
    plt.title("combined")
    plt.show()

    h_val_arr_cpy = h_val_arr.copy()

    cost_grid = h_val_arr_cpy
    u, v = np.zeros_like(h_val_arr_cpy), np.zeros_like(h_val_arr_cpy)
    for i in range(cost_grid.shape[0]):
        if i % 10 != 0:
            continue
        # print(i)
        for j in range(cost_grid.shape[1]):
            if j % 10 != 0:
                continue
            if cost_grid[i, j] == np.inf:
                continue

            ii, jj = unsnap_to_grid_torch((i, j), grid_step=grid_step, boundaries=boundary)
            best_pt = heuristic_dir2(cost_grid, [i, j], 1)
            best_pt_ = unsnap_to_grid_torch(best_pt, grid_step=grid_step, boundaries=boundary)
            uu, vv = best_pt_[0] - ii, best_pt_[1] - jj

            print((i, j), best_pt, (ii, jj), best_pt_, (uu, vv))

            u[i, j] = uu
            v[i, j] = vv

    plt.figure()
    skip = 10
    plt.quiver(Y[::skip, ::skip], X[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Vector Field")
    plt.show()
