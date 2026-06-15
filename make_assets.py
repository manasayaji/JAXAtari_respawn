import os
import numpy as np

# Find your local repository directory layout
base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
target_dir = os.path.join(base_dir, "src", "jaxatari", "games", "sprites", "demonattack")
os.makedirs(target_dir, exist_ok=True)

# Define rendering bounds configurations matching environment parameters
WIDTH = 160
HEIGHT = 160
PLAYER_SIZE = (6, 7)
DEMON_SIZE = (8, 12)
LASER_SIZE = (4, 1)
BOMB_SIZE = (4, 1)

# --- 1. Generate Background Array Map ---
bg_rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
bg_rgba[:, :, :3] = (0, 0, 0) # Black
bg_rgba[:, :, 3] = 255
np.save(os.path.join(target_dir, "background.npy"), bg_rgba)

# --- 2. Generate Player Sprite ---
player_mask = np.array([
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 1, 1],
], dtype=np.uint8)
player_rgba = np.zeros((*PLAYER_SIZE, 4), dtype=np.uint8)
color_p = (206, 49, 173, 255)
for r in range(player_mask.shape[0]):
    for c in range(player_mask.shape[1]):
        if player_mask[r, c]:
            player_rgba[r, c] = color_p
np.save(os.path.join(target_dir, "player.npy"), player_rgba)

# --- 3. Generate Demon Alien Sprite ---
demon_mask = np.array([
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
], dtype=np.uint8)
demon_rgba = np.zeros((*DEMON_SIZE, 4), dtype=np.uint8)
color_d = (184, 70, 162, 255)
start_col = (DEMON_SIZE[1] - 10) // 2
for r in range(demon_mask.shape[0]):
    for c in range(demon_mask.shape[1]):
        if demon_mask[r, c]:
            demon_rgba[r, start_col + c] = color_d
np.save(os.path.join(target_dir, "demon.npy"), demon_rgba)

# --- 4. Projectiles and Structural Bounds ---
laser_rgba = np.zeros((*LASER_SIZE, 4), dtype=np.uint8)
laser_rgba[:, :] = (252, 124, 254, 255)
np.save(os.path.join(target_dir, "projectile_player.npy"), laser_rgba)

bomb_rgba = np.zeros((*BOMB_SIZE, 4), dtype=np.uint8)
bomb_rgba[:, :] = (251, 135, 140, 255)
np.save(os.path.join(target_dir, "projectile_demon.npy"), bomb_rgba)

# Explosion
explosion_mask = np.array([
    [1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
], dtype=np.uint8)
explosion_rgba = np.zeros((*PLAYER_SIZE, 4), dtype=np.uint8)
for r in range(explosion_mask.shape[0]):
    for c in range(explosion_mask.shape[1]):
        if explosion_mask[r, c]:
            explosion_rgba[r, c] = color_p
np.save(os.path.join(target_dir, "explosion.npy"), explosion_rgba)

# UI Hud Elements
lives_bg_rgba = np.zeros((16, WIDTH, 4), dtype=np.uint8)
lives_bg_rgba[:, :, :3] = (0, 0, 176)
lives_bg_rgba[:, :, 3] = 255
np.save(os.path.join(target_dir, "lives_bg.npy"), lives_bg_rgba)

small_p_mask = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
small_p_rgba = np.zeros((2, 3, 4), dtype=np.uint8)
color_sp = (242, 128, 135, 255)
for r in range(2):
    for c in range(3):
        if small_p_mask[r, c]:
            small_p_rgba[r, c] = color_sp
np.save(os.path.join(target_dir, "small_player.npy"), small_p_rgba)

# --- 5. Generate Digit Sprites ---
patterns = [
    [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],  # 0
    [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 1
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],  # 2
    [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 3
    [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],  # 4
    [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 5
    [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 6
    [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],  # 7
    [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 8
    [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 9
]

for i, pattern in enumerate(patterns):
    digit_matrix = np.zeros((8, 8, 4), dtype=np.uint8)
    color_rgba = (194, 169, 53, 255)
    for r, row in enumerate(pattern):
        for c, val in enumerate(row):
            if val:
                digit_matrix[r + 1, c + 2] = color_rgba
    np.save(os.path.join(target_dir, f"score_digit_{i}.npy"), digit_matrix)

print(f"All files compiled successfully inside workspace: {target_dir}")