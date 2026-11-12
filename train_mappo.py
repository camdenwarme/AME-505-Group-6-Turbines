"""Train the two-turbine environment using an adjacent light_mappo checkout."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--mappo-path", type=Path, default=ROOT.parent / "light_mappo")
    bootstrap.add_argument("--wind-speed", type=float, default=8.0)
    bootstrap.add_argument("--wind-direction", type=float, default=270.0)
    bootstrap.add_argument("--dt", type=float, default=4.0)
    bootstrap.add_argument("--video-fps", type=float, default=None,
                           help="Override playback FPS; default is 10 / dt (10x simulation speed).")
    integration, _ = bootstrap.parse_known_args()
    mappo = integration.mappo_path.resolve()
    if not (mappo / "config.py").is_file():
        bootstrap.error("Set --mappo-path to your light_mappo checkout.")
    sys.path.insert(0, str(mappo))

    import numpy as np
    import torch
    from config import get_config
    from envs.env_wrappers import DummyVecEnv
    from shared_critic_runner import SharedCriticRunner as EnvRunner
    from train.train import setup_device
    from wind_farm_env import WindFarmEnv

    parser = argparse.ArgumentParser(parents=[bootstrap, get_config()], conflict_handler="resolve")
    parser.set_defaults(env_name="windfarm", experiment_name="two_turbines",
                        num_env_steps=20000, episode_length=200, n_rollout_threads=1,
                        n_eval_rollout_threads=1, gamma=1.0,
                        use_feature_normalization=False)
    args = parser.parse_args()
    # Separate actors need no extra agent identity in the local observation.
    args.num_agents, args.share_policy, args.scenario_name = 2, False, "two_turbines"
    if not args.use_centralized_V:
        parser.error("The turbine setup requires a shared centralized critic.")
    if args.n_rollout_threads != 1 or args.use_eval:
        parser.error("This minimal integration supports one training environment and no --use_eval.")
    if args.episode_length < 1 or args.num_env_steps < args.episode_length:
        parser.error("num_env_steps must cover at least one positive-length episode.")
    if args.video_fps is not None and (not np.isfinite(args.video_fps) or args.video_fps <= 0):
        parser.error("video-fps must be positive.")
    recurrent = args.use_recurrent_policy or args.use_naive_recurrent_policy
    if (args.algorithm_name == "rmappo") != recurrent:
        parser.error("Use --algorithm_name rmappo with a recurrent-policy flag, or mappo without one.")
    if args.use_recurrent_policy and args.episode_length % args.data_chunk_length:
        parser.error("episode_length must be divisible by data_chunk_length for recurrent training.")
    if args.use_recurrent_policy and args.episode_length // args.data_chunk_length < max(2, args.num_mini_batch):
        parser.error("Recurrent training needs at least two sequence chunks and one per minibatch.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = setup_device(args)
    # Keep artifacts in the turbine repository, regardless of launch directory.
    run_dir = ROOT / "rl_runs" / "mappo" / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    import uuid
    run_dir = run_dir / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    env = WindFarmEnv(args.episode_length, integration.dt, integration.wind_speed,
                      integration.wind_direction)
    env.seed(args.seed)
    envs = DummyVecEnv([lambda: env])

    def record_evaluation(episode, policies):
        from wind_farm_video import record_policy_video
        # Separate simulator: evaluation must not reset the training environment
        # or invalidate the observations/recurrent states carried in its buffer.
        evaluation_env = WindFarmEnv(args.episode_length, integration.dt,
                                     integration.wind_speed, integration.wind_direction)
        try:
            evaluation_env.seed(args.seed)
            video = record_policy_video(evaluation_env, policies,
                                        run_dir / f"wind_farm_episode{episode}.mp4",
                                        args.video_fps, episode=episode)
            print(f"Deterministic evaluation after {episode} training episodes: {video}", flush=True)
        finally:
            evaluation_env.close()

    runner = None
    try:
        runner = EnvRunner({"all_args": args, "envs": envs, "eval_envs": None,
                            "num_agents": 2, "device": device, "run_dir": run_dir,
                            "episode_callback": record_evaluation})
        runner.run()
        from plot_training_energy import plot_training_energy
        print(f"Training energy plot: {plot_training_energy(runner.metrics_path)}", flush=True)
        from plot_training_loss import plot_training_loss
        print(f"Training loss plot: {plot_training_loss(runner.metrics_path)}", flush=True)
        from training_report import write_training_report
        print(f"Training plots report: {write_training_report(run_dir)}", flush=True)
    finally:
        envs.close()
        if runner is not None:
            runner.writter.export_scalars_to_json(str(run_dir / "logs" / "summary.json"))
            runner.writter.close()
    print(f"Checkpoints and logs: {run_dir}")


if __name__ == "__main__":
    main()
