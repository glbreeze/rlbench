import argparse
import json
import os
import pickle

import numpy as np
from PIL import Image
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape

import rlbench.backend.task as task
from rlbench import ObservationConfig
from pyrep.objects.shape import Shape
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.helper import inject_task_physics, simGetShapeMassAndInertia
from pyrep.backend import sim
import time
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from pyrep.objects.dummy import Dummy
import numpy as np

def T12_to_R_t(T12):
    T12 = np.asarray(T12, float).reshape(12)
    R = np.array([
        [T12[0],  T12[1],  T12[2]],
        [T12[4],  T12[5],  T12[6]],
        [T12[8],  T12[9],  T12[10]],
    ], dtype=float)
    t = np.array([T12[3], T12[7], T12[11]], dtype=float)
    return R, t


def get_or_create_dummy(name, size=0.012):
    try:
        return Dummy(name)
    except Exception:
        d = Dummy.create(size)
        d.set_name(name)
        return d
    

def mark_com_attached(shape: Shape, name_suffix="_COM_MARK", size=0.03):
    h = shape.get_handle()
    m, I, com_local, T = simGetShapeMassAndInertia(h)

    d = get_or_create_dummy(shape.get_name() + name_suffix, size=size)
    d.set_parent(shape)  # attach to the cracker
    d.set_position(com_local.tolist(), relative_to=shape)  # local CoM
    com_world = np.array(d.get_position())
    return com_local, com_world, m, d


def to_jsonable(x):
    """Recursively convert numpy types (ndarray, scalars) into JSON-serializable Python types."""
    import numpy as np

    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def tabletop_push_test(pyrep, obj: Shape, force=np.array([2.0, 0.0, 0.0]), steps=50):
    # assume object already on table, settled
    h = obj.get_handle()
    p0 = np.array(obj.get_position())
    o0 = np.array(obj.get_orientation())

    for _ in range(steps):
        sim.simAddForceAndTorque(h, force.tolist(), [0, 0, 0])
        pyrep.step()

    p1 = np.array(obj.get_position())
    o1 = np.array(obj.get_orientation())
    dp = np.linalg.norm(p1 - p0)
    do = np.linalg.norm(o1 - o0)
    return dp, do, p0, p1, o0, o1

def try_collect_demo(task_env, attempts=3):
    for k in range(attempts):
        try:
            (demo,) = task_env.get_demos(amount=1, live_demos=True)
            return True, demo, None
        except Exception as e:
            err = repr(e)
    return False, None, err









def check_and_make(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)


def save_demo(demo, example_path: str):
    # Save image data first, and then None the image data, and pickle

    left_shoulder_rgb_path = os.path.join(example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(example_path, LEFT_SHOULDER_MASK_FOLDER)

    right_shoulder_rgb_path = os.path.join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(example_path, RIGHT_SHOULDER_MASK_FOLDER)

    overhead_rgb_path = os.path.join(example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(example_path, OVERHEAD_MASK_FOLDER)

    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)

    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    for p in [
        left_shoulder_rgb_path, left_shoulder_depth_path, left_shoulder_mask_path,
        right_shoulder_rgb_path, right_shoulder_depth_path, right_shoulder_mask_path,
        overhead_rgb_path, overhead_depth_path, overhead_mask_path,
        wrist_rgb_path, wrist_depth_path, wrist_mask_path,
        front_rgb_path, front_depth_path, front_mask_path
    ]:
        check_and_make(p)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray((obs.left_shoulder_mask * 255).astype(np.uint8))

        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray((obs.right_shoulder_mask * 255).astype(np.uint8))

        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray((obs.overhead_mask * 255).astype(np.uint8))

        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))

        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))

        right_shoulder_rgb.save(os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))

        overhead_rgb.save(os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(os.path.join(overhead_mask_path, IMAGE_FORMAT % i))

        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))

        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None

        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None

        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None

        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None

        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def make_obs_config(img_size, renderer: str) -> ObservationConfig:
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # Save masks as rgb encodings (not single channel)
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if renderer == 'opengl':
        mode = RenderMode.OPENGL
    elif renderer == 'opengl3':
        mode = RenderMode.OPENGL3
    else:
        raise ValueError(f"Unknown renderer: {renderer}")

    obs_config.right_shoulder_camera.render_mode = mode
    obs_config.left_shoulder_camera.render_mode = mode
    obs_config.overhead_camera.render_mode = mode
    obs_config.wrist_camera.render_mode = mode
    obs_config.front_camera.render_mode = mode

    return obs_config


def collect_single_process(tasks, args):
    img_size = list(map(int, args.image_size))
    obs_config = make_obs_config(img_size, args.renderer)

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete(attach_grasped_objects=False)),
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        headless=args.headless,
    )
    rlbench_env.launch()

    tasks_with_problems = ""

    try:
        for task_id, t in enumerate(tasks):
            task_env = rlbench_env.get_task(t)
            task_name = task_env.get_name()

            var_target = task_env.variation_count()
            if args.variations >= 0:
                var_target = int(np.minimum(args.variations, var_target))

            print(f"\n=== Task {task_id+1}/{len(tasks)}: {task_name} | variations: {var_target} ===")

            for var in range(var_target):
                task_env.set_variation(var)
                descriptions, _ = task_env.reset()
                
                task_env._ballast_mass_override = args.ballast_mass_override
                task_env._ballast_offset_override = args.ballast_offset_override

                gripper = task_env._robot.gripper
                for joint in gripper.joints:
                    joint.set_joint_force(200.0)
                    
                Shape('Panda_leftfinger_respondable').set_bullet_friction(5.0)
                Shape('Panda_rightfinger_respondable').set_bullet_friction(5.0)

                # inject once per variation reset, same physics for all demos in this variation
                physics_meta_for_variation = inject_task_physics(task_env)

                pyrep = rlbench_env._pyrep
                for _ in range(50):
                    pyrep.step()

                

               
                


                variation_path = os.path.join(
                    args.save_path, task_name, VARIATIONS_FOLDER % var
                )
                check_and_make(variation_path)

                with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                    pickle.dump(descriptions, f)

                episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
                check_and_make(episodes_path)

                abort_variation = False

                for ex_idx in range(args.episodes_per_task):
                    print(f"Task: {task_name} | Variation: {var} | Demo: {ex_idx}")

                    attempts = args.max_attempts
                    while attempts > 0:
                        try:
                            (demo,) = task_env.get_demos(amount=1, live_demos=True)
                        except Exception as e:
                            attempts -= 1
                            if attempts > 0:
                                continue

                            problem = (
                                f"Failed collecting task {task_name} "
                                f"(variation: {var}, example: {ex_idx}). "
                                f"Skipping this variation.\n{str(e)}\n"
                            )
                            print(problem)
                            tasks_with_problems += problem
                            abort_variation = True
                            break

                        episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                        check_and_make(episode_path)

                        save_demo(demo, episode_path)

                        with open(os.path.join(episode_path, "physics.json"), "w") as f:
                            json.dump(to_jsonable(physics_meta_for_variation), f, indent=2)


                        break

                    if abort_variation:
                        break

    finally:
        rlbench_env.shutdown()

    return tasks_with_problems


def parse_args():
    parser = argparse.ArgumentParser(description="RLBench Dataset Generator (single-process debug)")
    parser.add_argument('--save_path', type=str, default='/tmp/rlbench_data/',
                        help='Where to save the demos.')
    parser.add_argument('--tasks', nargs='*', default=[],
                        help='The tasks to collect. If empty, all tasks are collected.')
    parser.add_argument('--image_size', nargs=2, type=int, default=[128, 128],
                        help='The size of the images to save.')
    parser.add_argument('--renderer', type=str, choices=['opengl', 'opengl3'], default='opengl3',
                        help='Renderer. opengl does not include shadows, but is faster.')
    parser.add_argument('--episodes_per_task', type=int, default=10,
                        help='The number of episodes to collect per task.')
    parser.add_argument('--variations', type=int, default=-1,
                        help='Number of variations to collect per task. -1 for all.')
    parser.add_argument('--arm_max_velocity', type=float, default=1.0,
                        help='Max arm velocity used for motion planning.')
    parser.add_argument('--arm_max_acceleration', type=float, default=4.0,
                        help='Max arm acceleration used for motion planning.')

    # Debugging / determinism
    parser.add_argument('--seed', type=int, default=1000,
                        help='Random seed for physics injection RNG.')
    parser.add_argument('--debug', action='store_true',
                        help='Drop into pdb after each variation reset (right before inject_task_physics).')
    parser.add_argument('--headless', action='store_true',
                        help='Run headless (no GUI). Omit this flag for GUI debugging.')
    parser.add_argument('--max_attempts', type=int, default=10,
                        help='Retry attempts for demo collection per episode.')

    parser.add_argument('--ballast_mass_override', type=float, default=None,
                    help='If set, force ballast mass (kg) for all groceries to this value.')
    parser.add_argument('--ballast_offset_override', nargs=3, type=float, default=None,
                    help='If set, force ballast local offset (x y z) in meters.')

    return parser.parse_args()


def main():
    args = parse_args()

    check_and_make(args.save_path)

    task_files = [
        t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
        if t != '__init__.py' and t.endswith('.py')
    ]

    if len(args.tasks) > 0:
        for t in args.tasks:
            if t not in task_files:
                raise ValueError(f"Task {t} not recognised!")
        task_files = args.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    problems = collect_single_process(tasks, args)

    print("\nData collection done!")
    if problems:
        print("\nProblems encountered:\n")
        print(problems)
    else:
        print("No problems encountered.")


if __name__ == '__main__':
    main()
