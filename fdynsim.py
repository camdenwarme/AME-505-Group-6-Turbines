"""Offline multi-agent rollouts using the unchanged OFF/FLORIDyn solver.

policies[i](wind_speed, wind_direction, heading_i) returns deg/s.
W and W_theta contain N+1 samples, giving N actions/transitions per actor.
One process runs one simulation at a time; separate processes are independent.
"""

import contextlib
import errno
import io
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "OFF"
DEFAULT_CASE = REPO / "02_Examples_and_Cases/02_Example_Cases/001_two_turbines_yaw_step_floris.yaml"


def FDynSim(policies, W, W_theta, *, theta0=None, dt=None, lambda_=0.1,
            case=DEFAULT_CASE, work_dir=None, worker_id=None,
            keep_results=False, verbose=False):
    """Precompute actions, run OFF once, and return one trajectory per actor.

    Heading recursion: theta[k+1,i] = theta[k,i] + action[k,i]*dt.
    Rewards are total FLORIS power at the end of each action interval;
    the last reward is REPLACED with lambda_ times trapezoidal farm energy.
    Headings are absolute meteorological headings, not yaw offsets.
    Wind directions are interpolated numerically as in native OFF (no wrap).

    The reusable worker YAML is outside the original case directory. Generated
    FLORIS/log files use a separate temporary output folder per rollout.
    keep_results=True retains those folders and exports the two native CSVs.
    The minimal run_simulation.py is not used or changed.
    """
    W = np.array(W, dtype=float, copy=True)
    W_theta = np.array(W_theta, dtype=float, copy=True)
    if W.ndim != 1 or W_theta.shape != W.shape or len(W) < 2:
        raise ValueError("W and W_theta must be equal-length 1D arrays with at least two samples.")
    if not np.isfinite(W).all() or not np.isfinite(W_theta).all() or np.any(W <= 0):
        raise ValueError("Wind inputs must be finite, with positive wind speed.")
    settings = yaml.safe_load(Path(case).read_text())
    n_turbines = len(settings["wind_farm"]["farm"]["layout_x"])
    if len(policies) != n_turbines:
        raise ValueError(f"Provide {n_turbines} policies, one for each turbine.")
    dt = float(settings["sim"]["sim"]["time step"] if dt is None else dt)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(lambda_):
        raise ValueError("dt must be positive and finite; lambda_ must be finite.")
    times = float(settings["sim"]["sim"]["time start"]) + np.arange(len(W)) * dt
    theta = np.empty((len(W), n_turbines))
    initial = np.full(n_turbines, W_theta[0]) if theta0 is None else np.asarray(theta0, dtype=float)
    if initial.shape != (n_turbines,) or not np.isfinite(initial).all():
        raise ValueError("theta0 needs one finite heading per turbine.")
    theta[0] = initial
    actions = np.empty((len(W)-1, n_turbines))
    for k in range(len(actions)):
        for i, policy in enumerate(policies):
            actions[k, i] = float(policy(W[k], W_theta[k], theta[k, i]))
        if not np.isfinite(actions[k]).all():
            raise ValueError(f"Non-finite policy action at step {k}.")
        theta[k+1] = theta[k] + actions[k] * dt

    settings["sim"]["sim"].update({"time start": float(times[0]), "time end": float(times[-1]), "time step": dt})
    settings["ambient"]["flow_field"].update({
        "wind_speeds": W.tolist(), "wind_speeds_t": times.tolist(),
        "wind_directions": W_theta.tolist(), "wind_directions_t": times.tolist(),
    })
    # Native OFF applies the command after measuring at t[k]. Command theta[k+1]
    # at t[k], so the next recorded heading equals the requested recursion.
    commands = np.vstack((theta[1:], theta[-1:]))
    settings["controller"]["settings"] = {
        "ctl": "prescribed yaw controller", "input_method": "yaml",
        "orientation": True, "path_to_orientation_csv": "",
        "orientation_deg": commands.tolist(), "orientation_t": times.tolist(),
    }
    # Training rollouts omit visualization/OP CSV work; native CSVs are optional.
    settings["vis"]["flow_field_plots"]["plot"] = False
    for key in ("effective_wf_layout", "effective_wf_tile", "turbine_effective_wind_speed"):
        settings["vis"]["debug"][key] = False

    worker_id = str(worker_id) if worker_id is not None else f"worker_{os.getpid()}"
    if not worker_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in worker_id):
        raise ValueError("worker_id may contain letters, digits, underscores and hyphens.")
    worker_dir = (Path(work_dir) if work_dir is not None else ROOT / "rl_runs").resolve() / worker_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_yaml = worker_dir / "case.yaml"
    # Prevent two callers with the same worker identity from overwriting its YAML.
    lock_path = worker_dir / ".running"
    try:
        lock = lock_path.open("x")
    except FileExistsError as exc:
        raise RuntimeError(f"Worker {worker_id} is already running (lock: {lock_path}).") from exc
    output = None
    logger = logging.getLogger("off")
    previous_handlers = tuple(logger.handlers)
    old_cwd = Path.cwd()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import floris  # Use the native import before OFF's package check.
        if str(REPO / "03_Code") not in sys.path:
            sys.path.insert(0, str(REPO / "03_Code"))
        import off.off as off
        from off.off_interface import OFFInterface
        if Path(off.OFF_PATH).resolve() != REPO.resolve():
            raise RuntimeError("Another OFF copy is already imported. Run FDynSim in a fresh Python process.")
        if keep_results:
            run_dir = Path(tempfile.mkdtemp(prefix="rollout_", dir=worker_dir))
        else:
            output = tempfile.TemporaryDirectory(prefix="rollout_", dir=worker_dir)
            run_dir = Path(output.name)
        settings["io"].update({"data folder": str(worker_dir), "simulation folder": str(run_dir)})
        worker_yaml.write_text(yaml.safe_dump(settings, sort_keys=False))
        os.chdir(REPO)  # Native OFF resolves other case inputs from its root.
        with contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO()):
            interface = OFFInterface()
            interface.init_simulation_by_path(worker_yaml)
            # Initialize each turbine's complete historical chain at theta0.
            for i, turbine in enumerate(interface.wind_farm.turbines):
                turbine.set_orientation_yaw(theta[0, i], W_theta[0])
                state = turbine.turbine_states.get_all_states()[0].copy()
                turbine.turbine_states.init_all_states(state)
                turbine.observation_points.init_all_states(
                    turbine.ambient_states.get_turbine_wind_speed_u(),
                    turbine.ambient_states.get_turbine_wind_speed_v(),
                    turbine.get_rotor_pos(), dt)
            interface.run_sim()
            if keep_results:
                interface.store_measurements()
                interface.store_applied_control()
                interface.store_run_file()
        measurements = interface.get_measurements().copy()
        controls = interface.control_applied.copy()
        power_table = measurements.pivot(index="time", columns="t_idx", values="Power_FLORIS").sort_index()
        headings_table = controls.pivot(index="time", columns="t_idx", values="orientation").sort_index()
        if power_table.shape != theta.shape or not np.allclose(power_table.index.to_numpy(), times):
            raise RuntimeError("OFF output timestamps or turbine count do not match the planned trajectory.")
        if not np.allclose(headings_table.to_numpy(), theta, rtol=0, atol=1e-8):
            raise RuntimeError("OFF's recorded headings do not match the planned trajectory.")
        power = power_table.sum(axis=1, skipna=False).to_numpy()
        if not np.isfinite(power).all():
            raise RuntimeError("Invalid FLORIS power: check wind-relative headings and model operating limits.")
        energy = float(np.trapezoid(power, x=times))
        R = power.copy()
        R[-1] = float(lambda_) * energy
        terminated = np.zeros(len(actions), dtype=bool)
        terminated[-1] = True
        trajectories = []
        for i in range(n_turbines):
            states = np.column_stack((W, W_theta, theta[:, i]))
            trajectories.append({"states": states[:-1].copy(), "actions": actions[:, i].copy(),
                "rewards": R[1:].copy(), "next_states": states[1:].copy(), "terminated": terminated.copy()})
        return {"trajectories": trajectories, "times": times, "headings": theta,
            "actions": actions, "farm_power": power, "R": R, "energy": energy,
            "measurements": measurements, "controls": controls, "worker_yaml": str(worker_yaml),
            "run_dir": str(run_dir) if keep_results else None}
    finally:
        os.chdir(old_cwd)
        # Native OFF adds handlers every initialization; remove only this call's.
        for handler in list(logger.handlers):
            if handler not in previous_handlers:
                logger.removeHandler(handler)
                handler.close()
        try:
            if output is not None:
                # Shared filesystems can briefly report a just-emptied directory
                # as nonempty; close OFF's actual logger first, then retry.
                for attempt in range(3):
                    try:
                        output.cleanup()
                        break
                    except OSError as exc:
                        if exc.errno != errno.ENOTEMPTY or attempt == 2:
                            raise
                        time.sleep(0.1)
        finally:
            lock.close()
            lock_path.unlink(missing_ok=True)
