"""Train the two-turbine environment using an adjacent light_mappo checkout."""

import argparse
import csv
from importlib import metadata
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
    bootstrap.add_argument("--no-video", action="store_true",
                           help="Run numeric evaluations without rendering videos.")
    bootstrap.add_argument("--evaluate-aligned-baseline", action="store_true",
                           help="Also simulate an aligned reference during evaluation.")
    bootstrap.add_argument("--numeric-eval-interval", type=int, default=10,
                           help="Evaluate the greedy policy after this many completed episodes.")
    bootstrap.add_argument("--team-reward-weight", type=float, default=0.5,
                           help="Weight of the farm advantage in each actor update, from 0 to 1.")
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
                        use_feature_normalization=False, use_valuenorm=False,
                        lr=3.5e-4, critic_lr=3.5e-4,
                        ppo_epoch=10)
    args = parser.parse_args()
    # Separate actors need no extra agent identity in the local observation.
    args.num_agents, args.share_policy, args.scenario_name = 2, False, "two_turbines"
    if not args.use_centralized_V:
        parser.error("The turbine setup requires a shared centralized critic.")
    if args.use_popart or args.use_valuenorm:
        parser.error("Horizon-scaled farm values use per-horizon loss scaling, not PopArt or ValueNorm.")
    if args.n_rollout_threads != 1 or args.use_eval:
        parser.error("This minimal integration supports one training environment and no --use_eval.")
    if args.episode_length < 1 or args.num_env_steps < args.episode_length:
        parser.error("num_env_steps must cover at least one positive-length episode.")
    if args.video_fps is not None and (not np.isfinite(args.video_fps) or args.video_fps <= 0):
        parser.error("video-fps must be positive.")
    if not np.isfinite(args.team_reward_weight) or not 0 <= args.team_reward_weight <= 1:
        parser.error("team-reward-weight must be between 0 and 1.")
    if not np.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        parser.error("gamma must be between 0 and 1.")
    if args.numeric_eval_interval < 1:
        parser.error("numeric-eval-interval must be positive.")
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
    versions = {"python": sys.version}
    for package in ("floris", "numpy", "torch", "gym", "tensorboardX"):
        versions[package] = metadata.version(package)
    (run_dir / "runtime_versions.json").write_text(json.dumps(versions, indent=2))

    env = WindFarmEnv(args.episode_length, integration.dt, integration.wind_speed,
                      integration.wind_direction)
    env.seed(args.seed)
    envs = DummyVecEnv([lambda: env])

    def record_evaluation(episode, policies):
        from evaluate_wind_farm import evaluate_episode
        # Separate simulator: evaluation must not reset the training environment
        # or invalidate the observations/recurrent states carried in its buffer.
        evaluation_env = WindFarmEnv(args.episode_length, integration.dt,
                                     integration.wind_speed, integration.wind_direction)
        try:
            evaluation_env.seed(args.seed)
            # A physically simulated aligned benchmark is optional; individual
            # GAE baseline values are computed analytically without that rollout.
            records = []
            if episode == 0 and args.evaluate_aligned_baseline:
                records.append({"episode": episode, "policy": "aligned_baseline",
                                **evaluate_episode(evaluation_env)})
            records.append({"episode": episode, "policy": "greedy",
                            **evaluate_episode(evaluation_env, policies)})
            path = run_dir / "evaluation_metrics.csv"
            new_file = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                if new_file:
                    writer.writeheader()
                writer.writerows(records)
            for record in records:
                print(f"Evaluation episode {episode} {record['policy']}: "
                      f"{record['farm_energy_mwh']:.6f} MWh", flush=True)
            if not args.no_video:
                from wind_farm_video import record_policy_video
                video = record_policy_video(evaluation_env, policies,
                                            run_dir / f"wind_farm_episode{episode}.mp4",
                                            args.video_fps, episode=episode)
                print(f"Evaluation video: {video}", flush=True)
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
        from plot_rewards import plot_reward_summary
        print(f"Episode reward plot: {plot_reward_summary(run_dir)}", flush=True)
        from plot_training_diagnostics import make_diagnostics
        diagnostics_plot, diagnostics_summary = make_diagnostics(run_dir)
        print(f"Training diagnostics: {diagnostics_plot} and {diagnostics_summary}", flush=True)
        from training_report import write_training_report, write_evaluation_state_report
        print(f"Training plots report: {write_training_report(run_dir)}", flush=True)
        if not args.no_video:
            print(f"Evaluation report: {write_evaluation_state_report(run_dir)}", flush=True)
    finally:
        envs.close()
        if runner is not None:
            runner.writter.export_scalars_to_json(str(run_dir / "logs" / "summary.json"))
            runner.writter.close()
    print(f"Checkpoints and logs: {run_dir}")


if __name__ == "__main__":
    main()
