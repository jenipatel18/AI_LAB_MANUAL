
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random, math, copy

def merge_tiles(tiles, n_tiles):
    rows = []
    for i in range(n_tiles):
        row = np.hstack(tiles[i*n_tiles:(i+1)*n_tiles])
        rows.append(row)
    return np.vstack(rows)

def fitness_function(tiles, n_tiles):
    cost = 0
    for i in range(n_tiles):
        for j in range(n_tiles):
            idx = i * n_tiles + j
            if j < n_tiles - 1:
                right_idx = i * n_tiles + (j + 1)
                cost += np.sum(np.abs(tiles[idx][:, -1, :] - tiles[right_idx][:, 0, :]))
            if i < n_tiles - 1:
                bottom_idx = (i + 1) * n_tiles + j
                cost += np.sum(np.abs(tiles[idx][-1, :, :] - tiles[bottom_idx][0, :, :]))
    return cost

def random_swap(state):
    i, j = random.sample(range(len(state)), 2)
    new_state = copy.deepcopy(state)
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state

def simulated_annealing(initial_state, n_tiles, T=1e5, cooling_rate=0.995, max_iter=5000):
    current_state = initial_state
    current_cost = fitness_function(current_state, n_tiles)
    best_state, best_cost = current_state, current_cost

    for step in range(max_iter):
        new_state = random_swap(current_state)
        new_cost = fitness_function(new_state, n_tiles)

        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / T):
            current_state, current_cost = new_state, new_cost
            if new_cost < best_cost:
                best_state, best_cost = new_state, new_cost

        T *= cooling_rate
        if step % 200 == 0:
            print(f"Iter {step:4d} | Temp={T:.2f} | Cost={current_cost:.2f}")

    return best_state, best_cost

if __name__ == "__main__":
    data = loadmat("scrambled_lena.mat")
    scrambled_img = None
    for key in data:
        if isinstance(data[key], np.ndarray):
            scrambled_img = data[key]
            break

    if scrambled_img is None:
        raise ValueError("Could not find valid image data in scrambled_lena.mat")

    scrambled_img = scrambled_img.astype(np.float32)
    if scrambled_img.max() > 1:
        scrambled_img /= 255.0

    n_tiles = 4
    h, w, c = scrambled_img.shape
    tile_h, tile_w = h // n_tiles, w // n_tiles

    tiles = []
    for i in range(n_tiles):
        for j in range(n_tiles):
            tiles.append(scrambled_img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w, :])

    random.shuffle(tiles)

    print("Initial scrambled cost:", fitness_function(tiles, n_tiles))
    best_state, best_cost = simulated_annealing(tiles, n_tiles)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(merge_tiles(tiles, n_tiles))
    ax[0].set_title("Initial Scrambled")
    ax[0].axis("off")

    ax[1].imshow(merge_tiles(best_state, n_tiles))
    ax[1].set_title(f"Solved (Cost={best_cost:.2f})")
    ax[1].axis("off")

    plt.show()
