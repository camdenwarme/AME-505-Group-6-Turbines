"""Train a configurable wind farm using an adjacent light_mappo checkout."""

import argparse
import csv
from importlib import metadata
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--mappo-path", type=Path, default=ROOT.parent / "light_mappo")
    bootstrap.add_argument("--wind-speed", type=float, default=8.0)
    bootstrap.add_argument("--wind-direction", type=float, default=270.0)
    bootstrap.add_argument("--layout-file", type=Path, default=None,
                           help="JSON array of [x,y] or [x,y,z] turbine positions in rotor diameters; default square4.")
    bootstrap.add_argument("--wind-mode", choices=("off", "speed", "direction", "both"), default="off",
                           help="Toggle seeded, time-varying wind inside every episode.")
    bootstrap.add_argument("--wind-seed", type=int, default=None,
                           help="Training wind seed; defaults to the model seed.")
    bootstrap.add_argument("--eval-wind-seed", type=int, default=None,
                           help="Evaluation wind seed; independent stream, defaults to the model seed.")
    bootstrap.add_argument("--wind-speed-min", type=float, default=6.0)
    bootstrap.add_argument("--wind-speed-max", type=float, default=10.0)
    bootstrap.add_argument("--wind-direction-half-width", type=float, default=15.0,
                           help="Random direction knot half-width around --wind-direction, in degrees.")
    bootstrap.add_argument("--wind-knot-interval", type=float, default=120.0,
                           help="Seconds between random wind knots, with linear interpolation.")
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
                           help="Farm weight for --advantage-weight-schedule constant.")
    bootstrap.add_argument("--advantage-weight-schedule", choices=("exponential", "constant"),
                           default="exponential", help="Advantage mixing schedule; default exponential.")
    bootstrap.add_argument("--advantage-decay-beta", type=float, default=0.05,
                           help="Individual weight = (1-beta)**completed_rounds; default 0.05.")
    bootstrap.add_argument("--episode-workers", type=int, default=None,
                           help="Episodes collected in parallel before each update; alias for --n_rollout_threads.")
    bootstrap.add_argument("--threads-per-worker", type=int, default=1,
                           help="Numerical-library threads in each simulator process.")
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
    from wind_profiles import WindProfileConfig
    from parallel_env import ProcessVecEnv, WindFarmFactory

    parser = argparse.ArgumentParser(parents=[bootstrap, get_config()], conflict_handler="resolve")
    parser.set_defaults(env_name="windfarm", experiment_name="square4",
                        num_env_steps=20000, episode_length=200, n_rollout_threads=1,
                        n_eval_rollout_threads=1, gamma=1.0,
                        use_feature_normalization=False, use_valuenorm=False,
                        lr=3.5e-4, critic_lr=3.5e-4,
                        ppo_epoch=1)
    args = parser.parse_args()
    if args.episode_workers is not None:
        args.n_rollout_threads = args.episode_workers
    args.share_policy = False
    if not args.use_centralized_V:
        parser.error("The turbine setup requires a shared centralized critic.")
    if args.use_popart or args.use_valuenorm:
        parser.error("Horizon-scaled farm values use per-horizon loss scaling, not PopArt or ValueNorm.")
    if args.n_rollout_threads < 1 or args.threads_per_worker < 1:
        parser.error("episode-workers and threads-per-worker must be positive.")
    if args.use_eval:
        parser.error("Use --numeric-eval-interval instead of --use_eval.")
    if args.episode_length < 1 or args.num_env_steps < args.episode_length:
        parser.error("num_env_steps must cover at least one positive-length episode.")
    if args.num_env_steps % (args.episode_length * args.n_rollout_threads):
        parser.error("num_env_steps must be a multiple of episode_length * episode-workers (complete rounds).")
    if args.ppo_epoch < 1 or args.num_mini_batch < 1:
        parser.error("ppo_epoch and num_mini_batch must be positive.")
    if args.video_fps is not None and (not np.isfinite(args.video_fps) or args.video_fps <= 0):
        parser.error("video-fps must be positive.")
    if not np.isfinite(args.team_reward_weight) or not 0 <= args.team_reward_weight <= 1:
        parser.error("team-reward-weight must be between 0 and 1.")
    if not np.isfinite(args.advantage_decay_beta) or not 0 <= args.advantage_decay_beta <= 1:
        parser.error("advantage-decay-beta must be between 0 and 1.")
    if not np.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        parser.error("gamma must be between 0 and 1.")
    if args.numeric_eval_interval < 1:
        parser.error("numeric-eval-interval must be positive.")
    args.wind_seed = args.seed if args.wind_seed is None else args.wind_seed
    args.eval_wind_seed = args.seed if args.eval_wind_seed is None else args.eval_wind_seed
    try:
        wind_settings = dict(mode=args.wind_mode, speed_min=args.wind_speed_min,
                             speed_max=args.wind_speed_max,
                             direction_half_width=args.wind_direction_half_width,
                             knot_interval_s=args.wind_knot_interval)
        train_wind = WindProfileConfig(seed=args.wind_seed, **wind_settings)
        eval_wind = WindProfileConfig(seed=args.eval_wind_seed, **wind_settings)
        layout = None if args.layout_file is None else json.loads(args.layout_file.read_text())
        env = WindFarmEnv(args.episode_length, args.dt, args.wind_speed,
                          args.wind_direction, layout=layout, wind_config=train_wind, wind_stream=0)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    args.num_agents = env.num_agents
    args.scenario_name = "square4" if layout is None else f"farm_{args.num_agents}"
    args.layout_d = env.layout_d.tolist()
    args.remaining_time_index = env.remaining_time_index
    args.total_rated_power_w = env.total_rated_power_w
    args.training_wind_config = train_wind.to_dict()
    args.evaluation_wind_config = eval_wind.to_dict()
    recurrent = args.use_recurrent_policy or args.use_naive_recurrent_policy
    if (args.algorithm_name == "rmappo") != recurrent:
        parser.error("Use --algorithm_name rmappo with a recurrent-policy flag, or mappo without one.")
    if args.use_recurrent_policy and args.episode_length % args.data_chunk_length:
        parser.error("episode_length must be divisible by data_chunk_length for recurrent training.")
    if args.use_recurrent_policy and args.episode_length * args.n_rollout_threads // args.data_chunk_length < max(2, args.num_mini_batch):
        parser.error("Recurrent training needs at least two sequence chunks and one per minibatch.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
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

    env.seed(args.seed)
    if args.n_rollout_threads == 1:
        envs = DummyVecEnv([lambda: env])
    else:
        factories = [WindFarmFactory(dict(
            episode_length=args.episode_length, dt=args.dt, wind_speed=args.wind_speed,
            wind_direction=args.wind_direction, layout=env.layout_d,
            wind_config=train_wind, wind_stream=0, environment_id=i), args.seed + i)
            for i in range(args.n_rollout_threads)]
        envs = ProcessVecEnv(factories, gamma=args.gamma, threads=args.threads_per_worker)
    print(f"Collecting {args.n_rollout_threads} episode(s) per round; "
          f"{args.ppo_epoch} training pass(es) after each round.", flush=True)

    def record_evaluation(episode, policies):
        from evaluate_wind_farm import evaluate_episode
        # Separate simulator: evaluation must not reset the training environment
        # or invalidate the observations/recurrent states carried in its buffer.
        evaluation_env = WindFarmEnv(args.episode_length, args.dt, args.wind_speed,
                                     args.wind_direction, layout=env.layout_d,
                                     wind_config=eval_wind, wind_stream=1)
        try:
            evaluation_env.seed(args.seed)
            # One held-out profile is replayed at every checkpoint and by the
            # optional baseline/video. Explicit reset does not advance training
            # wind counters or draw from policy/global RNGs.
            scenario = evaluation_env.make_scenario(episode_id=0)
            scenario_path = run_dir / "evaluation_wind_scenario.json"
            if not scenario_path.exists():
                scenario_path.write_text(json.dumps(scenario.to_dict(), indent=2))
            # A physically simulated aligned benchmark is optional; individual
            # GAE baseline values are computed analytically without that rollout.
            records = []
            if episode == 0 and args.evaluate_aligned_baseline:
                kind = "aligned_baseline" if args.wind_mode == "off" else "yaw_tracking_baseline"
                records.append({"episode": episode, "policy": kind,
                                **evaluate_episode(evaluation_env, scenario=scenario,
                                  trajectory_path=run_dir / f"{kind}_steps.csv")})
            records.append({"episode": episode, "policy": "greedy",
                            **evaluate_episode(evaluation_env, policies, scenario=scenario,
                              trajectory_path=run_dir / f"evaluation_episode{episode}_steps.csv")})
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
                                            args.video_fps, episode=episode, scenario=scenario)
                print(f"Evaluation video: {video}", flush=True)
        finally:
            evaluation_env.close()

    runner = None
    try:
        runner = EnvRunner({"all_args": args, "envs": envs, "eval_envs": None,
                            "num_agents": env.num_agents, "device": device, "run_dir": run_dir,
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
