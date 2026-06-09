"""
RFT Data Collection Script
從 OmniGibson rollouts 收集成功軌跡，儲存為 LeRobot parquet 格式。
用法:
  python collect_rft_data.py \
    --task tidying_bedroom \
    --output /media/ML_2025/shawn/b1k/data/rft_data \
    --n_rollouts 50 \
    --max_success 20 \
    --instance_ids 0,1,2,3,4
"""
import argparse
import csv
import json
import logging
import numpy as np
import os
import sys
import torch as th
import pandas as pd
from pathlib import Path

# Set up env vars before importing OG
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# Import OmniGibson eval infrastructure (eval.py must be importable)
EVAL_DIR = Path(__file__).parent / "behavior-1k-solution/BEHAVIOR-1K/OmniGibson/omnigibson/learning"
sys.path.insert(0, str(EVAL_DIR.parent.parent.parent.parent))  # Add OG root

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.learning.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
    ROBOT_CAMERA_NAMES,
)
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
import cv2
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from inspect import getsourcefile

# Import Evaluator from eval.py
sys.path.insert(0, str(EVAL_DIR))
from eval import Evaluator

m = create_module_macros(module_path=__file__)
logger = logging.getLogger("rft_collector")
logger.setLevel(20)

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True


def collect_rft_data(
    task_name: str,
    output_dir: str,
    instance_ids: list,
    n_rollouts: int = 50,
    max_success: int = 20,
    policy_host: str = "localhost",
    write_video_flag: bool = True,
):
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    data_dir = Path(output_dir)
    parquet_dir = data_dir / f"data/task-{task_idx:04d}"
    video_base = data_dir / f"videos/task-{task_idx:04d}"
    for cam in ["observation.images.rgb.head", "observation.images.rgb.left_wrist", "observation.images.rgb.right_wrist"]:
        (video_base / cam).mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Load test instance list
    task_instance_csv_path = os.path.join(
        gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
    )
    with open(task_instance_csv_path, "r") as f:
        lines = list(csv.reader(f))[1:]
    test_instances = lines[task_idx][2].strip().split(",")
    instances_to_run = [int(test_instances[i]) for i in instance_ids]

    # Build hydra config
    register_omegaconf_resolvers()
    config_dir = str(EVAL_DIR / "configs")
    with hydra.initialize_config_dir(config_dir, version_base="1.1"):
        overrides = [
            f"task.name={task_name}",
            f"policy=websocket",
            f"model.host={policy_host}",
            "log_path=/tmp/rft_collect_log",
            "write_video=false",
        ]
        config = hydra.compose("base_config.yaml", overrides=overrides)
    OmegaConf.resolve(config)
    gm.HEADLESS = True

    n_success = 0
    episode_counter = 0  # unique episode index within this collection run

    # Find starting episode index to avoid collision
    existing = sorted(parquet_dir.glob("episode_*.parquet"))
    start_episode_id = int(existing[-1].stem.split("_")[-1]) + 1 if existing else task_idx * 10000

    with Evaluator(config) as evaluator:
        for rollout in range(n_rollouts):
            if n_success >= max_success:
                logger.info(f"Reached max_success={max_success}, stopping.")
                break

            instance_id = instances_to_run[rollout % len(instances_to_run)]
            logger.info(f"Rollout {rollout+1}/{n_rollouts}: instance={instance_id}")

            evaluator.reset()
            evaluator.load_task_instance(instance_id, test_hidden=False)
            evaluator.reset()

            # Buffers
            states, cam_poses, actions, task_infos = [], [], [], []
            rgb_head, rgb_left, rgb_right = [], [], []

            done = False
            success = False
            while not done:
                obs = evaluator.obs
                # Record observations BEFORE step
                if "robot_r1::proprio" in obs:
                    states.append(obs["robot_r1::proprio"].numpy().astype(np.float32))
                if "robot_r1::cam_rel_poses" in obs:
                    cam_poses.append(obs["robot_r1::cam_rel_poses"].numpy().astype(np.float32))
                if "task::low_dim" in obs:
                    task_infos.append(obs["task::low_dim"].numpy().astype(np.float32))

                # Record images for video
                if write_video_flag:
                    head_key = ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"
                    left_key = ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"
                    right_key = ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"
                    if head_key in obs:
                        rgb_head.append(obs[head_key].numpy())
                    if left_key in obs:
                        rgb_left.append(obs[left_key].numpy())
                    if right_key in obs:
                        rgb_right.append(obs[right_key].numpy())

                terminated, truncated = evaluator.step()
                action = evaluator.robot_action

                # Record action
                if isinstance(action, dict):
                    # Flatten action dict (should be R1Pro action)
                    action_arr = np.concatenate([v.numpy().flatten() if hasattr(v, 'numpy') else np.array(v).flatten()
                                                  for v in action.values()]).astype(np.float32)
                else:
                    action_arr = np.array(action).flatten().astype(np.float32)
                actions.append(action_arr)

                if terminated:
                    success = evaluator.n_success_trials > 0
                    done = True
                if truncated:
                    done = True

            if success:
                n_success += 1
                episode_id = start_episode_id + episode_counter
                episode_counter += 1
                T = min(len(states), len(actions))

                logger.info(f"  SUCCESS! Saving episode {episode_id} ({T} steps)")

                # Save parquet
                df = pd.DataFrame({
                    "index": np.arange(T, dtype=np.int64),
                    "episode_index": np.full(T, episode_id, dtype=np.int64),
                    "task_index": np.full(T, task_idx, dtype=np.int64),
                    "timestamp": np.arange(T, dtype=np.float64) / 30.0,
                    "observation.state": list(np.stack(states[:T])),
                    "observation.cam_rel_poses": list(np.stack(cam_poses[:T]) if cam_poses else np.zeros((T, 21), dtype=np.float32)),
                    "action": list(np.stack(actions[:T])),
                    "observation.task_info": list(np.stack(task_infos[:T]) if task_infos else np.zeros((T, 94), dtype=np.float32)),
                })
                pq_path = parquet_dir / f"episode_{episode_id:08d}.parquet"
                df.to_parquet(pq_path, index=False)
                logger.info(f"  Saved parquet: {pq_path}")

                # Save videos
                if write_video_flag and rgb_head:
                    for cam_name, frames in [
                        ("observation.images.rgb.head", rgb_head),
                        ("observation.images.rgb.left_wrist", rgb_left),
                        ("observation.images.rgb.right_wrist", rgb_right),
                    ]:
                        vid_path = str(video_base / cam_name / f"episode_{episode_id:08d}.mp4")
                        h, w = frames[0].shape[:2]
                        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                        for frame in frames[:T]:
                            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        writer.release()
                    logger.info(f"  Saved videos for episode {episode_id}")
            else:
                logger.info(f"  FAILED. success so far: {n_success}/{rollout+1}")

    logger.info(f"\n=== Collection complete: {n_success} successes from {n_rollouts} rollouts ===")
    logger.info(f"Data saved to: {output_dir}")
    return n_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="tidying_bedroom")
    parser.add_argument("--output", default="/media/ML_2025/shawn/b1k/data/rft_data")
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--max_success", type=int, default=20)
    parser.add_argument("--instance_ids", default="0,1,2,3,4")
    parser.add_argument("--policy_host", default="localhost")
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    instance_ids = [int(x) for x in args.instance_ids.split(",")]
    collect_rft_data(
        task_name=args.task,
        output_dir=args.output,
        instance_ids=instance_ids,
        n_rollouts=args.n_rollouts,
        max_success=args.max_success,
        policy_host=args.policy_host,
        write_video_flag=not args.no_video,
    )
