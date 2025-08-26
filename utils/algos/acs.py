import torch
import math


def find_max_square(matrix: torch.Tensor, threshold: float = 1e-13):
    if matrix.ndim > 2:
        matrix = matrix[(0,) * (matrix.ndim - 2)]
    h, w = matrix.shape
    mask = (matrix > threshold).float()

    prefix_sum = torch.cumsum(torch.cumsum(mask, dim=0), dim=1)

    def get_region_sum(x1, y1, x2, y2):
        total = prefix_sum[x2, y2].clone()
        if x1 > 0:
            total -= prefix_sum[x1 - 1, y2]
        if y1 > 0:
            total -= prefix_sum[x2, y1 - 1]
        if x1 > 0 and y1 > 0:
            total += prefix_sum[x1 - 1, y1 - 1]

        return total
    
    nx_max = 0
    ny_max = 0
    for n in range(1, min(h, w) + 1):
        nx = n + ((n % 2) != (h / 2))
        ny = n + ((n % 2) != (w / 2))

        x_start = math.floor(h / 2) + math.ceil(-nx / 2)
        x_end = math.floor(h / 2) + math.ceil(nx / 2)

        y_start = math.floor(w / 2) + math.ceil(-ny / 2)
        y_end = math.floor(w / 2) + math.ceil(ny / 2)

        if x_start < 0 or x_end > h or y_start < 0 or y_end > w:
            break
        region_sum = get_region_sum(x_start, y_start, x_end - 1, y_end - 1)
        if region_sum == (x_end - x_start) * (y_end - y_start):
            nx_max = nx
            ny_max = ny
        else:
            break
    return nx_max, ny_max