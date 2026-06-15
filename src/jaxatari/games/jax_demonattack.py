import os

from functools import partial

from typing import Tuple

import numpy as np

import chex

import jax.lax

import jax.numpy as jnp

from flax import struct

import jaxatari.spaces as spaces

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation

from jaxatari.renderers import JAXGameRenderer

from jaxatari.rendering import jax_rendering_utils as render_utils

INITIAL_WAVE_PATTERNS = 12
REPEATING_WAVE_PATTERN_START = 8
PATTERNS_PER_DIFFICULTY_ENTRY = 2
DIFFICULTY_TABLE_NAMES = (
    "WAVE_X_TABLE",
    "WAVE_Y_TABLE",
    "WAVE_DIR_TABLE",
    "WAVE_DEMON_SPEED_TABLE",
    "ENEMY_SHOT_SPEED_TABLE",
    "WAVE_LASER_SPEED_TABLE",
)
FORMATION_TABLE_NAMES = ("WAVE_X_TABLE", "WAVE_Y_TABLE", "WAVE_DIR_TABLE")

def _create_digit_sprites(consts: "DemonAttackConstants") -> jnp.ndarray:
    digits = np.zeros((10, 8, 8, 4), dtype=np.uint8)

    color = (*consts.SCORE_COLOR, 255)

    patterns = [

        [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],

        [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],

        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],

        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],

        [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],

        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],

        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],

        [[1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],

        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],

        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],

    ]

    for i, pattern in enumerate(patterns):

        for r, row in enumerate(pattern):

            for c, val in enumerate(row):

                if val:
                    digits[i, r + 1, c + 2] = color

    return jnp.array(digits)


def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'Background.npy'},
        {'name': 'player', 'type': 'single', 'file': 'Player.npy'},
        {'name': 'player_missile', 'type': 'single', 'file': 'PlayerMissile.npy'},
        {'name': 'projectile_demon', 'type': 'single', 'file': 'Bomb_1.npy'},
        {'name': 'demon_1', 'type': 'group', 'files': [
            'Enemy_1/Enemy_1.npy',
            'Enemy_1/Enemy_2.npy',
            'Enemy_1/Enemy_3.npy',
            'Enemy_1/Enemy_4.npy',
        ]},
        {'name': 'demon_2', 'type': 'group', 'files': [
            'Enemy_2/Enemy_1.npy',
            'Enemy_2/Enemy_2.npy',
            'Enemy_2/Enemy_3.npy',
            'Enemy_2/Enemy_4.npy',
        ]},
        {'name': 'demon_3', 'type': 'group', 'files': [
            'Enemy_3/Enemy_1.npy',
            'Enemy_3/Enemy_2.npy',
            'Enemy_3/Enemy_3.npy',
            'Enemy_3/Enemy_4.npy',
        ]},
        {'name': 'demon_4', 'type': 'group', 'files': [
            'Enemy_4/Enemy_1.npy',
            'Enemy_4/Enemy_2.npy',
            'Enemy_4/Enemy_3.npy',
            'Enemy_4/Enemy_4.npy',
        ]},
        {'name': 'demon_5', 'type': 'group', 'files': [
            'Enemy_5/Enemy_1.npy',
            'Enemy_5/Enemy_2.npy',
            'Enemy_5/Enemy_3.npy',
            'Enemy_5/Enemy_4.npy',
        ]},
        {'name': 'demon_6', 'type': 'group', 'files': [
            'Enemy_6/Enemy_1.npy',
            'Enemy_6/Enemy_2.npy',
            'Enemy_6/Enemy_3.npy',
            'Enemy_6/Enemy_4.npy',
        ]},
        {'name': 'demon_7', 'type': 'group', 'files': [
            'Enemy_7/Enemy_1.npy',
            'Enemy_7/Enemy_2.npy',
            'Enemy_7/Enemy_3.npy',
            'Enemy_7/Enemy_4.npy',
        ]},
        {'name': 'demon_8', 'type': 'group', 'files': [
            'Enemy_8/Enemy_1.npy',
            'Enemy_8/Enemy_2.npy',
            'Enemy_8/Enemy_3.npy',
            'Enemy_8/Enemy_4.npy',
        ]},
        {'name': 'demon_9', 'type': 'group', 'files': [
            'Enemy_9/Enemy_1.npy',
            'Enemy_9/Enemy_2.npy',
            'Enemy_9/Enemy_3.npy',
            'Enemy_9/Enemy_4.npy',
        ]},
        {'name': 'demon_10', 'type': 'group', 'files': [
            'Enemy_10/Enemy_1.npy',
            'Enemy_10/Enemy_2.npy',
            'Enemy_10/Enemy_3.npy',
            'Enemy_10/Enemy_4.npy',
        ]},
        {'name': 'demon_11', 'type': 'group', 'files': [
            'Enemy_11/Enemy_1.npy',
            'Enemy_11/Enemy_2.npy',
            'Enemy_11/Enemy_3.npy',
            'Enemy_11/Enemy_4.npy',
        ]},
        {'name': 'demon_12', 'type': 'group', 'files': [
            'Enemy_12/Enemy_1.npy',
            'Enemy_12/Enemy_2.npy',
            'Enemy_12/Enemy_3.npy',
            'Enemy_12/Enemy_4.npy',
        ]},
        {'name': 'small_demon_5', 'type': 'group', 'files': [
            'Enemy_Small_5/Enemy_1.npy',
            'Enemy_Small_5/Enemy_2.npy',
            'Enemy_Small_5/Enemy_3.npy',
            'Enemy_Small_5/Enemy_4.npy',
        ]},
        {'name': 'small_demon_6', 'type': 'group', 'files': [
            'Enemy_Small_6/Enemy_1.npy',
            'Enemy_Small_6/Enemy_2.npy',
            'Enemy_Small_6/Enemy_3.npy',
            'Enemy_Small_6/Enemy_4.npy',
        ]},
        {'name': 'small_demon_7', 'type': 'group', 'files': [
            'Enemy_Small_7/Enemy_1.npy',
            'Enemy_Small_7/Enemy_2.npy',
            'Enemy_Small_7/Enemy_3.npy',
            'Enemy_Small_7/Enemy_4.npy',
        ]},
        {'name': 'small_demon_8', 'type': 'group', 'files': [
            'Enemy_Small_8/Enemy_1.npy',
            'Enemy_Small_8/Enemy_2.npy',
            'Enemy_Small_8/Enemy_3.npy',
            'Enemy_Small_8/Enemy_4.npy',
        ]},
        {'name': 'small_demon_9', 'type': 'group', 'files': [
            'Enemy_Small_9/Enemy_1.npy',
            'Enemy_Small_9/Enemy_2.npy',
            'Enemy_Small_9/Enemy_3.npy',
            'Enemy_Small_9/Enemy_4.npy',
        ]},
        {'name': 'small_demon_10', 'type': 'group', 'files': [
            'Enemy_Small_10/Enemy_1.npy',
            'Enemy_Small_10/Enemy_2.npy',
            'Enemy_Small_10/Enemy_3.npy',
            'Enemy_Small_10/Enemy_4.npy',
        ]},
        {'name': 'small_demon_11', 'type': 'group', 'files': [
            'Enemy_Small_11/Enemy_1.npy',
            'Enemy_Small_11/Enemy_2.npy',
            'Enemy_Small_11/Enemy_3.npy',
            'Enemy_Small_11/Enemy_4.npy',
        ]},
        {'name': 'small_demon_12', 'type': 'group', 'files': [
            'Enemy_Small_12/Enemy_1.npy',
            'Enemy_Small_12/Enemy_2.npy',
            'Enemy_Small_12/Enemy_3.npy',
            'Enemy_Small_12/Enemy_4.npy',
        ]},
        {'name': 'enemy_spawn_left', 'type': 'group', 'files': [
            'EnemySpawnAnimation/EnemySpawn_left_1.npy',
            'EnemySpawnAnimation/EnemySpawn_left_2.npy',
            'EnemySpawnAnimation/EnemySpawn_left_3.npy',
        ]},
        {'name': 'enemy_spawn_right', 'type': 'group', 'files': [
            'EnemySpawnAnimation/EnemySpawn_right_1.npy',
            'EnemySpawnAnimation/EnemySpawn_right_2.npy',
            'EnemySpawnAnimation/EnemySpawn_right_3.npy',
        ]},
        {'name': 'enemy_death_animation', 'type': 'group', 'files': [
            'EnemyDeathAnimation/EnemyPart_0.npy',
            'EnemyDeathAnimation/EnemyPart_1.npy',
            'EnemyDeathAnimation/EnemyPart_2.npy',
        ]},
        {'name': 'player_death_animation', 'type': 'group', 'files': [
            'PlayerDeathAnimation/Explode_1.npy',
            'PlayerDeathAnimation/Explode_2.npy',
            'PlayerDeathAnimation/Explode_3.npy',
            'PlayerDeathAnimation/Explode_4.npy',
            'PlayerDeathAnimation/Explode_5.npy',
            'PlayerDeathAnimation/Explode_6.npy',
            'PlayerDeathAnimation/Explode_7.npy',
        ]},
        {'name': 'bunker', 'type': 'single', 'file': 'Bunker.npy'},
    )


class DemonAttackConstants(struct.PyTreeNode):
    # Static Configuration

    WIDTH: int = struct.field(pytree_node=False, default=160)

    HEIGHT: int = struct.field(pytree_node=False, default=160)

    PLAYER_SPEED: int = struct.field(pytree_node=False, default=2)

    MAX_DEMONS: int = struct.field(pytree_node=False, default=3)

    DEMON_SPEED: int = struct.field(pytree_node=False, default=1)

    WAVE_DEMON_SPEED_TABLE: Tuple[int, ...] = struct.field(pytree_node=False, default=(1, 1, 2, 2, 3, 3))
    WAVE_LASER_SPEED_TABLE: Tuple[int, ...] = struct.field(pytree_node=False, default=(3, 4, 5, 5, 6, 6))
    ENEMY_SHOT_ACTION_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(8, 6, 6, 3, 5, 4, 5, 4, 5, 4, 5, 4),
    )
    ENEMY_SHOT_SPEED_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(2, 2, 2, 2, 3, 3),
    )
    # Coordinates & Sizes. Sizes are (height, width).
    PLAYER_Y: int = struct.field(pytree_node=False, default=174)
    PLAYER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(12, 7))
    DEMON_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(9, 18))
    LASER_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 1))
    PLAYER_LASER_DEPTH: int = struct.field(pytree_node=False, default=2)
    PLAYER_DEATH_ANIMATION_DURATION: int = struct.field(pytree_node=False, default=70)
    PLAYER_DEATH_FLASH_DURATION: int = struct.field(pytree_node=False, default=20)
    BOMB_SIZE: Tuple[int, int] = struct.field(pytree_node=False, default=(4, 1))
    MAX_BOMBS: int = struct.field(pytree_node=False, default=7)
    BOMB_BURST_RATES: int = struct.field(pytree_node=False, default=4)
    BOMB_BURST_RATE_INTERVAL: int = struct.field(pytree_node=False, default=3)
    # Assign the seven bomb slots to four timed volleys: 2 + 2 + 2 + 1.
    BOMB_BURST_RATE_BY_SLOT: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(0, 0, 1, 1, 2, 2, 3),
    )
    BOMB_BURST_X_OFFSETS: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(-4, 4, -4, 4, -4, 4, -2),
    )
    BOMB_JITTER_X_TABLE: Tuple[int, ...] = struct.field(
        pytree_node=False,
        default=(0, 0, 1, 0, 0, -1, 0),
    )
    MAX_BUNKERS: int = struct.field(pytree_node=False, default=6)
    INIT_BUNKERS: int = struct.field(pytree_node=False, default=3)
    BUNKER_X: int = struct.field(pytree_node=False, default=16)
    BUNKER_Y: int = struct.field(pytree_node=False, default=188)
    BUNKER_SPACING: int = struct.field(pytree_node=False, default=7)

    # Boundaries

    PLAYER_MIN_X: int = struct.field(pytree_node=False, default=16)

    PLAYER_MAX_X: int = struct.field(pytree_node=False, default=136)

    DEMON_MIN_X: int = struct.field(pytree_node=False, default=16)

    DEMON_MAX_X: int = struct.field(pytree_node=False, default=136)

    DEMON_MIN_Y: int = struct.field(pytree_node=False, default=20)

    DEMON_MAX_Y: int = struct.field(pytree_node=False, default=85)

    # Wave progression: how many pixels each wave pushes demons down

    WAVE_LEVEL_STEP: int = struct.field(pytree_node=False, default=32)

    # Y at which newly spawned demons appear at the top of the arena

    DEMON_SPAWN_TOP_Y: int = struct.field(pytree_node=False, default=20)

    # Colors
    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(194, 169, 53))

    LIVES_BG_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 176))

    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)


class DemonAttackState(struct.PyTreeNode):
    player_x: chex.Array

    laser_x: chex.Array

    laser_y: chex.Array

    laser_active: chex.Array

    demons_x: chex.Array

    demons_y: chex.Array

    demons_dir: chex.Array

    demons_y_dir: chex.Array

    demons_alive: chex.Array

    demons_spawn_y: chex.Array

    bomb_x: chex.Array

    bomb_y: chex.Array

    bomb_active: chex.Array
    bomb_source_idx: chex.Array
    bomb_burst_step: chex.Array
    bomb_burst_timer: chex.Array
    bomb_action_counter: chex.Array

    score: chex.Array

    lives: chex.Array

    player_exploding: chex.Array

    explosion_timer: chex.Array

    step_counter: chex.Array

    wave: chex.Array

    key: chex.PRNGKey


class DemonAttackObservation(struct.PyTreeNode):
    player: ObjectObservation

    demons: ObjectObservation

    laser: ObjectObservation

    bomb: ObjectObservation

    score: jnp.ndarray

    lives: jnp.ndarray


class DemonAttackInfo(struct.PyTreeNode):
    time: jnp.ndarray


class JaxDemonAttack(JaxEnvironment[DemonAttackState, DemonAttackObservation, DemonAttackInfo, DemonAttackConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(

        [Action.NOOP, Action.FIRE, Action.RIGHT, Action.LEFT, Action.RIGHTFIRE, Action.LEFTFIRE],

        dtype=jnp.int32,

    )

    def __init__(self, consts: DemonAttackConstants = None):
        consts = consts or DemonAttackConstants()

        super().__init__(consts)

        self.renderer = DemonAttackRenderer(self.consts)

    @staticmethod
    def _validate_wave_configuration(consts: DemonAttackConstants) -> None:
        """Fail early when custom wave tables cannot be indexed consistently."""
        expected_difficulty_entries = (
            INITIAL_WAVE_PATTERNS // PATTERNS_PER_DIFFICULTY_ENTRY
        )
        invalid_tables = [
            name
            for name in DIFFICULTY_TABLE_NAMES
            if len(getattr(consts, name)) != expected_difficulty_entries
        ]
        if invalid_tables:
            raise ValueError(
                f"Difficulty tables need {expected_difficulty_entries} entries: "
                f"{', '.join(invalid_tables)}"
            )
        if len(consts.WAVE_DEMON_TABLE) != INITIAL_WAVE_PATTERNS:
            raise ValueError(
                f"WAVE_DEMON_TABLE needs {INITIAL_WAVE_PATTERNS} pattern entries"
            )
        if len(consts.ENEMY_SHOT_ACTION_TABLE) != INITIAL_WAVE_PATTERNS:
            raise ValueError(
                f"ENEMY_SHOT_ACTION_TABLE needs {INITIAL_WAVE_PATTERNS} pattern entries"
            )

    def _wave_level_mod12(self, wave_number: chex.Array) -> chex.Array:
        return jnp.where(wave_number < 12, wave_number, 8 + jnp.mod(wave_number, 4)).astype(jnp.int32)

        initial_spawn_y = jnp.linspace(

            self.consts.DEMON_MIN_Y,

            self.consts.DEMON_MAX_Y,

            self.consts.MAX_DEMONS,

            dtype=jnp.int32,

        )

        state = DemonAttackState(

        spawn_anim_total = self._spawn_animation_duration()

        return {
            "wave_number": wave_number,
            "wave_pattern": wave_pattern,
            "wave_total": wave_total,
            "wave_spawned": initial_alive_count,
            "spawn_timer": spawn_timer,
            "spawn_anim_timer": jnp.where(demons_alive, spawn_anim_total, 0),
            "spawn_pause_timer": jnp.where(demons_alive, self.consts.SPAWN_MOVE_PAUSE, 0),
            "demons_x": demons_x,
            "demons_y": demons_y,
            "demons_dir": demons_dir,
            "demons_y_dir": jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.int32),
            "demons_alive": demons_alive,
        }

    def _initialize_wave_state(
        self, state: DemonAttackState, wave_number: chex.Array
    ) -> DemonAttackState:
        """Replace the previous wave state with a freshly initialized wave."""
        wave_values = self._build_wave_start_values(wave_number)

        # Active starting demons play the spawn animation, then remain stationary for
        # SPAWN_MOVE_PAUSE frames before movement and bomb drops are allowed.
        return state.replace(
            **wave_values,
            game_frozen=jnp.array(False, dtype=jnp.bool_),
            bomb_x=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.int32),
            bomb_y=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.int32),
            bomb_active=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.bool_),
            bomb_source_idx=jnp.array(0, dtype=jnp.int32),
            bomb_burst_step=jnp.array(self.consts.BOMB_BURST_RATES, dtype=jnp.int32),
            bomb_burst_timer=jnp.array(0, dtype=jnp.int32),
            bomb_action_counter=jnp.array(0, dtype=jnp.int32),
        )

            laser_x=jnp.array(0, dtype=jnp.int32),

            laser_y=jnp.array(0, dtype=jnp.int32),

        return jax.lax.cond(
            should_freeze,
            lambda s: s.replace(
                wave_number=next_wave_number,
                demons_alive=jnp.zeros((self.consts.MAX_DEMONS,), dtype=jnp.bool_),
                bomb_active=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.bool_),
                bomb_source_idx=jnp.array(0, dtype=jnp.int32),
                bomb_burst_step=jnp.array(self.consts.BOMB_BURST_RATES, dtype=jnp.int32),
                bomb_burst_timer=jnp.array(0, dtype=jnp.int32),
                bomb_action_counter=jnp.array(0, dtype=jnp.int32),
                laser_active=jnp.array(False, dtype=jnp.bool_),
                game_frozen=jnp.array(True, dtype=jnp.bool_),
            ),
            lambda s: self._initialize_wave_state(
                s.replace(
                    lives=jnp.minimum(
                        s.lives + 1,
                        jnp.array(self.consts.MAX_BUNKERS, dtype=jnp.int32)
                    )
                ),
                next_wave_number
            ),
            operand=state,
        )

            demons_x=jnp.linspace(20, 120, self.consts.MAX_DEMONS, dtype=jnp.int32),

            demons_y=initial_spawn_y,

            demons_dir=jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.int32),

            demons_y_dir=jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.int32),

            demons_alive=jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.bool_),

            demons_spawn_y=initial_spawn_y,

        state = DemonAttackState(
            player_x=jnp.array(76, dtype=jnp.int32),
            laser_x=jnp.array(0, dtype=jnp.int32),
            laser_y=jnp.array(0, dtype=jnp.int32),
            laser_active=jnp.array(False, dtype=jnp.bool_),
            demons_x=wave_values["demons_x"],
            demons_y=wave_values["demons_y"],
            demons_dir=wave_values["demons_dir"],
            demons_y_dir=wave_values["demons_y_dir"],
            demons_alive=wave_values["demons_alive"],
            bomb_x=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.int32),
            bomb_y=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.int32),
            bomb_active=jnp.zeros((self.consts.MAX_BOMBS,), dtype=jnp.bool_),
            bomb_source_idx=jnp.array(0, dtype=jnp.int32),
            bomb_burst_step=jnp.array(self.consts.BOMB_BURST_RATES, dtype=jnp.int32),
            bomb_burst_timer=jnp.array(0, dtype=jnp.int32),
            bomb_action_counter=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),

            lives=jnp.array(6, dtype=jnp.int32),

            player_exploding=jnp.array(False, dtype=jnp.bool_),

            explosion_timer=jnp.array(0, dtype=jnp.int32),

            step_counter=jnp.array(0, dtype=jnp.int32),

            wave=jnp.array(0, dtype=jnp.int32),

            key=key

        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DemonAttackState, action: chex.Array) -> Tuple[

        DemonAttackObservation, DemonAttackState, float, bool, DemonAttackInfo]:
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        prev_state = state

        def update_explosion(s):
            new_timer = s.explosion_timer - 1

            exploding = new_timer > 0

            return s.replace(explosion_timer=new_timer, player_exploding=exploding)

        def normal_step(s, act):
            s = self._player_step(s, act)

            s = self._laser_step(s, act)

            s = self._demons_step(s)

            s = self._bomb_step(s)

            s = self._handle_collisions(s)

            return s

        state = jax.lax.cond(

            state.player_exploding,

            update_explosion,

            lambda s: normal_step(s, atari_action),

            operand=state

        )

        key, next_key = jax.random.split(state.key)

        state = state.replace(key=next_key, step_counter=state.step_counter + 1)

        observation = self._get_observation(state)

        reward = self._get_reward(prev_state, state)

        done = self._get_done(state)

        info = self._get_info(state)

        return observation, state, reward, done, info

    def _player_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        move_right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)

        move_left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)

        dx = jax.lax.select(move_right, self.consts.PLAYER_SPEED,

                            jax.lax.select(move_left, -self.consts.PLAYER_SPEED, 0))

        new_x = jnp.clip(state.player_x + dx, self.consts.PLAYER_MIN_X, self.consts.PLAYER_MAX_X)

        return state.replace(player_x=new_x)

    def _laser_step(self, state: DemonAttackState, action: chex.Array) -> DemonAttackState:
        fire = jnp.logical_or(jnp.logical_or(action == Action.FIRE, action == Action.RIGHTFIRE),

                              action == Action.LEFTFIRE)

        should_fire = jnp.logical_and(fire, jnp.logical_not(state.laser_active))

        laser_x = jax.lax.select(should_fire, state.player_x + self.consts.PLAYER_SIZE[1] // 2, state.laser_x)

        laser_y = jax.lax.select(should_fire,

                                 jnp.array(self.consts.PLAYER_Y - self.consts.LASER_SIZE[0], dtype=jnp.int32),

                                 state.laser_y)

        laser_active = jnp.logical_or(should_fire, state.laser_active)

        laser_y = jax.lax.select(laser_active, laser_y - self.consts.LASER_SPEED, laser_y)

        laser_active = jnp.logical_and(laser_active, laser_y > 0)

        return state.replace(laser_x=laser_x, laser_y=laser_y, laser_active=laser_active)

    def _demons_step(self, state: DemonAttackState) -> DemonAttackState:
        demon_speed = self._difficulty_value_for_pattern(
            self.consts.WAVE_DEMON_SPEED_TABLE, state.wave_pattern
        )
        can_move = self._demons_ready(state)
        burst_in_progress = state.bomb_burst_step < self.consts.BOMB_BURST_RATES
        source_ids = jnp.arange(self.consts.MAX_DEMONS, dtype=jnp.int32) == state.bomb_source_idx
        can_move = jnp.logical_and(
            can_move,
            jnp.logical_not(jnp.logical_and(burst_in_progress, source_ids)),
        )

        # Horizontal movement
        new_x = jnp.where(
            can_move,
            state.demons_x + state.demons_dir * demon_speed,
            state.demons_x,
        )

        at_right_edge = new_x >= self.consts.DEMON_MAX_X

        at_left_edge = new_x <= self.consts.DEMON_MIN_X

        new_dir = jnp.where(at_right_edge, -1, jnp.where(at_left_edge, 1, state.demons_dir))

        new_x = jnp.clip(new_x, self.consts.DEMON_MIN_X, self.consts.DEMON_MAX_X)

        new_y = state.demons_y + state.demons_y_dir * self.consts.DEMON_SPEED

        at_bottom_edge = new_y >= self.consts.DEMON_MAX_Y

        at_top_edge = new_y <= self.consts.DEMON_MIN_Y

        new_y_dir = jnp.where(at_bottom_edge, -1, jnp.where(at_top_edge, 1, state.demons_y_dir))

        new_y = jnp.clip(new_y, self.consts.DEMON_MIN_Y, self.consts.DEMON_MAX_Y)

        all_dead = jnp.logical_not(jnp.any(state.demons_alive))

        # Shift existing spawn-Y targets one level down for the new wave

        shifted_spawn_y = jnp.clip(

            state.demons_spawn_y + self.consts.WAVE_LEVEL_STEP,

            self.consts.DEMON_MIN_Y,

            self.consts.DEMON_MAX_Y,

        )

        new_spawn_y = jnp.where(all_dead, shifted_spawn_y, state.demons_spawn_y)

        new_wave = jnp.where(all_dead, state.wave + 1, state.wave)

        key, spawn_key = jax.random.split(state.key)

        spawn_x = jax.random.randint(spawn_key, (self.consts.MAX_DEMONS,),

                                     self.consts.DEMON_MIN_X, self.consts.DEMON_MAX_X)

        # Choose respawn Y per demon

        respawn_y_single = state.demons_spawn_y  # same height as when killed

        respawn_y_wave = jnp.full(

            (self.consts.MAX_DEMONS,),

            self.consts.DEMON_SPAWN_TOP_Y,

            dtype=jnp.int32,

        )

        respawn_y = jnp.where(all_dead, respawn_y_wave, respawn_y_single)

        # Apply respawn only to dead demons (or all if wave cleared)

        new_x = jnp.where(state.demons_alive, new_x, spawn_x)

        new_y = jnp.where(state.demons_alive, new_y, respawn_y)

        # Dead demons that respawn should head toward their spawn-Y row

        # (move downward when spawning at top, keep current dir otherwise)

        respawn_y_dir = jnp.where(

            respawn_y < state.demons_y,  # spawn point is above current pos → move down
            jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.int32),

            jnp.full((self.consts.MAX_DEMONS,), -1, dtype=jnp.int32),

        )

        new_y_dir = jnp.where(state.demons_alive, new_y_dir, respawn_y_dir)

        demons_alive = jnp.ones((self.consts.MAX_DEMONS,), dtype=jnp.bool_)

        return state.replace(

            demons_x=new_x,

            demons_y=new_y,

            demons_dir=new_dir,

            demons_y_dir=new_y_dir,

            demons_alive=demons_alive,

            demons_spawn_y=new_spawn_y,

            wave=new_wave,

            key=key

        )

    def _bomb_step(self, state: DemonAttackState) -> DemonAttackState:
        """Advance enemy bomb movement and burst-firing state by one frame.

        Existing bombs move downward at the speed selected for the current wave
        pattern and receive their slot-specific horizontal jitter. Bombs that
        reach the bunker boundary are deactivated.

        The method also advances the enemy firing scheduler. Once the wave's
        action delay has elapsed, no previous bombs remain active, and at least
        one demon is ready, a random roll may begin a burst from a selected
        demon. Each burst retains that source demon and activates the bomb slots
        assigned to its current rate after the configured interval. The burst
        source is released when all rates have been processed.
        """
        key, drop_key, demon_idx_key, burst_length_key = jax.random.split(
            state.key, 4
        )
        ready_demons = self._demons_ready(state)

        slot_ids = jnp.arange(self.consts.MAX_BOMBS, dtype=jnp.int32)

        # First branch: wave/action timing logic
        action_limit = jnp.asarray(
            self.consts.ENEMY_SHOT_ACTION_TABLE,
            dtype=jnp.int32,
        )[self._wave_level_mod12(state.wave_number)]

        action_counter = state.bomb_action_counter + 1
        any_bomb_active = jnp.any(state.bomb_active)

        # Advance existing bombs before adding the current frame's bomb
        bomb_speed = self._difficulty_value_for_pattern(
            self.consts.ENEMY_SHOT_SPEED_TABLE,
            state.wave_pattern,
        )
        moved_y = state.bomb_y + jnp.where(state.bomb_active, bomb_speed, 0)
        bomb_active = jnp.logical_and(
            state.bomb_active,
            moved_y < self.consts.BUNKER_Y - self.consts.BOMB_SIZE[0],
        )

        # First branch: per-slot jitter logic
        jitter_table = jnp.asarray(
            self.consts.BOMB_JITTER_X_TABLE,
            dtype=jnp.int32,
        )
        jitter_phase = jnp.mod(
            state.step_counter + slot_ids,
            len(self.consts.BOMB_JITTER_X_TABLE),
        )
        jitter_x = jitter_table[jitter_phase]

        moved_x = jnp.clip(
            state.bomb_x + jnp.where(bomb_active, jitter_x, 0),
            self.consts.BOUNDARY,
            self.consts.WIDTH - self.consts.BOUNDARY - self.consts.BOMB_SIZE[1],
        )
        bomb_x = jnp.where(bomb_active, moved_x, state.bomb_x)
        bomb_y = jnp.where(bomb_active, moved_y, state.bomb_y)

        # Second branch: choose a random living / ready demon
        picked_demon_idx = jax.random.randint(
            demon_idx_key,
            (),
            0,
            self.consts.MAX_DEMONS,
            dtype=jnp.int32,
        )
        picked_demon_idx = jnp.where(
            ready_demons[picked_demon_idx],
            picked_demon_idx,
            jnp.argmax(ready_demons).astype(jnp.int32),
        )

        # A burst owns one demon until all four bomb rates have fired. New bursts wait
        # until the previous bombs have left the screen.
        burst_in_progress = state.bomb_burst_step < self.consts.BOMB_BURST_RATES
        drop_roll = jax.random.bits(drop_key, (), dtype=jnp.uint8)
        can_start_burst = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_not(burst_in_progress),
                jnp.logical_not(any_bomb_active),
            ),
            jnp.logical_and(
                action_counter >= action_limit,
                jnp.logical_and(jnp.any(ready_demons), drop_roll >= 176),
            ),
        )
        source_idx = jnp.where(
            can_start_burst,
            picked_demon_idx,
            state.bomb_source_idx,
        )
        source_ready = ready_demons[source_idx]
        base_x = (
            state.demons_x[source_idx]
            + self.consts.DEMON_SIZE[1] // 2
            - self.consts.BOMB_SIZE[1] // 2
        )

        burst_length = jax.random.randint(
            burst_length_key,
            (),
            0,
            self.consts.BOMB_BURST_RATES + 1,
            dtype=jnp.int32,
        )
        burst_step = jnp.where(
            can_start_burst,
            self.consts.BOMB_BURST_RATES - burst_length,
            state.bomb_burst_step,
        )
        burst_timer = jnp.where(can_start_burst, 0, state.bomb_burst_timer)
        burst_in_progress = burst_step < self.consts.BOMB_BURST_RATES
        fire_rate_now = jnp.logical_and(
            burst_in_progress,
            jnp.logical_and(source_ready, burst_timer <= 0),
        )

        # Activate every slot assigned to this bomb shot in one vectorized operation.
        safe_burst_step = jnp.minimum(
            burst_step,
            self.consts.BOMB_BURST_RATES - 1,
        )
        rate_by_slot = jnp.asarray(
            self.consts.BOMB_BURST_RATE_BY_SLOT,
            dtype=jnp.int32,
        )
        slots_in_rate = rate_by_slot == safe_burst_step

        x_offsets = jnp.asarray(
            self.consts.BOMB_BURST_X_OFFSETS,
            dtype=jnp.int32,
        )
        fired_x = jnp.clip(
            base_x + x_offsets,
            self.consts.BOUNDARY,
            self.consts.WIDTH - self.consts.BOUNDARY - self.consts.BOMB_SIZE[1],
        )
        fired_y = (
            state.demons_y[source_idx]
            + self.consts.DEMON_SIZE[0]
        )

        should_activate_slot = jnp.logical_and(fire_rate_now, slots_in_rate)
        bomb_x = jnp.where(should_activate_slot, fired_x, bomb_x)
        bomb_y = jnp.where(should_activate_slot, fired_y, bomb_y)
        bomb_active = jnp.logical_or(bomb_active, should_activate_slot)

        # After firing, wait N complete frames before allowing the next bomb shot
        next_burst_step = jnp.where(fire_rate_now, burst_step + 1, burst_step)
        next_burst_timer = jnp.where(
            fire_rate_now,
            jnp.array(self.consts.BOMB_BURST_RATE_INTERVAL, dtype=jnp.int32),
            jnp.maximum(burst_timer - 1, 0),
        )
        burst_done = next_burst_step >= self.consts.BOMB_BURST_RATES
        source_idx = jnp.where(
            burst_done,
            jnp.array(0, dtype=jnp.int32),
            source_idx,
        )
        action_counter = jnp.where(can_start_burst, 0, action_counter)

        return state.replace(
            key=key,
            bomb_x=bomb_x,
            bomb_y=bomb_y,
            bomb_active=bomb_active,
            bomb_source_idx=source_idx,
            bomb_burst_step=next_burst_step,
            bomb_burst_timer=next_burst_timer,
            bomb_action_counter=action_counter,
        )

    def _handle_collisions(self, state: DemonAttackState) -> DemonAttackState:
        def check_demon_collision(i, carry):
            s_alive, s_score, l_active, s_spawn_y = carry

            demon_hit = jnp.logical_and(

                s_alive[i],

                jnp.logical_and(

                    l_active,

                    jnp.logical_and(

                        jnp.abs(state.laser_x - state.demons_x[i]) < self.consts.DEMON_SIZE[0],

                        jnp.logical_and(

                            state.laser_y < state.demons_y[i] + self.consts.DEMON_SIZE[1],

                            state.laser_y + self.consts.LASER_SIZE[1] > state.demons_y[i]

                        )

                    )

                )

            )

            new_alive = s_alive.at[i].set(jnp.logical_and(s_alive[i], jnp.logical_not(demon_hit)))

            new_score = jnp.where(demon_hit, s_score + 10, s_score)

            new_laser_active = jnp.logical_and(l_active, jnp.logical_not(demon_hit))

            # Record current Y so the next spawn lands at the same height

            new_spawn_y = s_spawn_y.at[i].set(

                jnp.where(demon_hit, state.demons_y[i], s_spawn_y[i])

            )

            return (new_alive, new_score, new_laser_active, new_spawn_y)

        init_carry = (state.demons_alive, state.score, state.laser_active, state.demons_spawn_y)

        demons_alive, score, laser_active, demons_spawn_y = jax.lax.fori_loop(

            0, self.consts.MAX_DEMONS, check_demon_collision, init_carry

        )

        player_hit = jnp.logical_and(

            state.bomb_active,

            jnp.logical_and(

                jnp.abs(state.bomb_x - state.player_x) < self.consts.PLAYER_SIZE[0],

                jnp.logical_and(
                    state.bomb_y < self.consts.PLAYER_Y + self.consts.PLAYER_SIZE[0],
                    state.bomb_y + self.consts.BOMB_SIZE[0] > self.consts.PLAYER_Y
                )

            )
        )
        any_player_hit = jnp.any(player_hit)

        bunker_available = state.lives > 0
        lives = jnp.where(
            jnp.logical_and(any_player_hit, bunker_available),
            state.lives - 1,
            state.lives,
        )
        game_over = jnp.logical_or(
            state.game_over,
            jnp.logical_and(any_player_hit, jnp.logical_not(bunker_available)),
        )
        bomb_active = jnp.where(
            any_player_hit,
            jnp.zeros_like(state.bomb_active),
            state.bomb_active,
        )

        # If player hit, start explosion
        player_exploding = jnp.logical_or(state.player_exploding, any_player_hit)
        explosion_timer = jnp.where(
            any_player_hit,
            self.consts.PLAYER_DEATH_ANIMATION_DURATION,
            state.explosion_timer,
        )

        explosion_timer = jnp.where(player_hit, 20, state.explosion_timer)

        return state.replace(

            demons_alive=demons_alive,

            score=score,

            laser_active=laser_active,

            demons_spawn_y=demons_spawn_y,

            lives=lives,

            bomb_active=bomb_active,

            player_exploding=player_exploding,

            explosion_timer=explosion_timer,

        )

    def render(self, state: DemonAttackState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: DemonAttackState):
        player = ObjectObservation.create(

            x=state.player_x,

            y=jnp.array(self.consts.PLAYER_Y),

            width=jnp.array(self.consts.PLAYER_SIZE[0]),

            height=jnp.array(self.consts.PLAYER_SIZE[1]),

        )

        demons = ObjectObservation.create(

            x=state.demons_x,

            y=state.demons_y,

            width=jnp.array(self.consts.DEMON_SIZE[0]),

            height=jnp.array(self.consts.DEMON_SIZE[1]),

            active=state.demons_alive

        )

        laser = ObjectObservation.create(

            x=state.laser_x,

            y=state.laser_y,

            width=jnp.array(self.consts.LASER_SIZE[0]),

            height=jnp.array(self.consts.LASER_SIZE[1]),

            active=state.laser_active

        )

        bomb = ObjectObservation.create(

            x=state.bomb_x,

            y=state.bomb_y,
            width=jnp.full_like(state.bomb_x, self.consts.BOMB_SIZE[1], dtype=jnp.int32),
            height=jnp.full_like(state.bomb_y, self.consts.BOMB_SIZE[0], dtype=jnp.int32),
            active=state.bomb_active

        )

        return DemonAttackObservation(

            player=player, demons=demons, laser=laser, bomb=bomb, score=state.score, lives=state.lives

        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        demons_space = spaces.get_object_space(n=self.consts.MAX_DEMONS,
                                               screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({

            "player": object_space,

            "demons": demons_space,

            "laser": object_space,
            "bomb": spaces.get_object_space(n=self.consts.MAX_BOMBS,
                                            screen_size=(self.consts.HEIGHT, self.consts.WIDTH)),
            "score": spaces.Box(low=0, high=99999, shape=(), dtype=jnp.int32),

            "lives": spaces.Box(low=0, high=6, shape=(), dtype=jnp.int32),

        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DemonAttackState) -> DemonAttackInfo:
        return DemonAttackInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DemonAttackState, state: DemonAttackState):
        return (state.score - previous_state.score).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DemonAttackState) -> bool:
        return state.lives <= 0


class DemonAttackRenderer(JAXGameRenderer):

    def __init__(self, consts: DemonAttackConstants = None, config: render_utils.RendererConfig = None):

        super().__init__(consts)

        self.consts = consts or DemonAttackConstants()

        if config is None:

            self.config = render_utils.RendererConfig(

                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),

                channels=3,

                downscale=None

            )

        else:

            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)

        bg_rgba = jnp.zeros((*self.config.game_dimensions, 4), dtype=jnp.uint8)

        bg_rgba = bg_rgba.at[:, :, :3].set(jnp.array(self.consts.BACKGROUND_COLOR))

        bg_rgba = bg_rgba.at[:, :, 3].set(255)

        player_sprite = _create_player_sprite(self.consts)

        demon_sprite = _create_demon_sprite(self.consts)

        laser_sprite = _create_projectile_sprite(self.consts.LASER_SIZE, self.consts.LASER_COLOR)

        # 2. Create procedural assets
        digit_sprites = _create_digit_sprites(self.consts)

        # Update asset config with procedural data
        final_asset_config.append({'name': 'score_digits', 'type': 'procedural', 'data': digit_sprites})

        (

            self.PALETTE,

            self.SHAPE_MASKS,

            self.BACKGROUND,

            self.COLOR_TO_ID,

            self.FLIP_OFFSETS

        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DemonAttackState):

        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render player or explosion

        player_mask = jax.lax.select(state.player_exploding, self.SHAPE_MASKS["explosion"], self.SHAPE_MASKS["player"])

        raster = self.jr.render_at(raster, state.player_x, self.consts.PLAYER_Y, player_mask)

        # Render demons

        demon_mask = self.SHAPE_MASKS["demon"]

        def render_demon(i, r):
            return jax.lax.cond(

                state.demons_alive[i],

                lambda: self.jr.render_at(r, state.demons_x[i], state.demons_y[i], demon_mask),

                lambda: r

            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_DEMONS, render_demon, raster)

        # Render laser

        laser_mask = self.SHAPE_MASKS["projectile_player"]

        laser_render_x = jax.lax.select(state.laser_active, state.laser_x,
                                        state.player_x + self.consts.PLAYER_SIZE[1] // 2)

        laser_render_y = jax.lax.select(state.laser_active, state.laser_y,
                                        self.consts.PLAYER_Y - self.consts.LASER_SIZE[1] + 2)

        raster = self.jr.render_at(raster, laser_render_x, laser_render_y, laser_mask)

        # Render enemy shot particles.
        bomb_mask = self.SHAPE_MASKS["projectile_demon"]

        def render_bomb(i, r):
            return jax.lax.cond(
                jnp.logical_and(state.bomb_active[i], jnp.logical_not(state.player_exploding)),
                lambda: self.jr.render_at(r, state.bomb_x[i], state.bomb_y[i], bomb_mask),
                lambda: r,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_BOMBS, render_bomb, raster)

        # --- Render Score ---

        score_digits = self.jr.int_to_digits(state.score, max_digits=4)

        digit_masks = self.SHAPE_MASKS["score_digits"]

        is_single_digit = state.score < 10

        is_double_digit = jnp.logical_and(state.score >= 10, state.score < 100)

        is_triple_digit = jnp.logical_and(state.score >= 100, state.score < 1000)

        start_index = jax.lax.select(is_single_digit, 3,

                                     jax.lax.select(is_double_digit, 2,

        frame = self.jr.render_from_palette(raster, self.PALETTE)
        death_elapsed = (
            self.consts.PLAYER_DEATH_ANIMATION_DURATION - state.explosion_timer
        )
        flash_frames_left = jnp.clip(
            self.consts.PLAYER_DEATH_FLASH_DURATION - death_elapsed,
            0,
            self.consts.PLAYER_DEATH_FLASH_DURATION,
        )
        flash_intensity = (
            jnp.array(255, dtype=jnp.int32) * flash_frames_left
        ) // self.consts.PLAYER_DEATH_FLASH_DURATION
        flash_color = jnp.asarray(flash_intensity, dtype=jnp.uint8)
        return jnp.where(
            jnp.logical_and(
                jnp.logical_and(state.player_exploding, flash_frames_left > 0),
                jnp.all(frame == 0, axis=-1, keepdims=True),
            ),
            flash_color,
            frame,
        )

        num_to_render = jax.lax.select(is_single_digit, 1,

                                       jax.lax.select(is_double_digit, 2,

                                                      jax.lax.select(is_triple_digit, 3, 4)))

        score_render_x = jax.lax.select(is_single_digit, 70 + (3 * 8) // 2,

                                        jax.lax.select(is_double_digit, 70 + (2 * 8) // 2,

                                                       jax.lax.select(is_triple_digit, 70 + 8 // 2, 70)))

        raster = self.jr.render_label_selective(raster, score_render_x, 10, score_digits, digit_masks,

                                                start_index, num_to_render, spacing=8)

        # Render Lives Indicator Zone

        lives_bg_mask = self.SHAPE_MASKS["lives_bg"]

        raster = self.jr.render_at(raster, 0, self.consts.LIVES_Y, lives_bg_mask)

        player_icon = self.SHAPE_MASKS["small_player"]

        def render_life_icon(i, r):
            return jax.lax.cond(

                i < state.lives,

                lambda: self.jr.render_at(r, self.consts.LIVES_X + i * self.consts.LIVES_SPACING, self.consts.LIVES_Y,
                                          player_icon),

                lambda: r

            )

        raster = jax.lax.fori_loop(0, 6, render_life_icon, raster)

        return self.jr.render_from_palette(raster, self.PALETTE)
