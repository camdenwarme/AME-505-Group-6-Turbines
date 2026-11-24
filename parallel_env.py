"""Spawn one isolated simulator per episode in a synchronous training round.

Actors stay in the parent process. All workers receive actions before any result
is collected; no worker resets or starts another episode until the next reset.
"""

from dataclasses import dataclass
from contextlib import contextmanager
import multiprocessing as mp
import os
import time
import traceback
from types import SimpleNamespace


@contextmanager
def _thread_limits(threads):
    # Set before spawning: unpickling wind settings can import NumPy before the
    # worker function starts. Restore the parent's environment afterward.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class WindFarmFactory:
    kwargs: dict
    seed: int

    def __call__(self):
        from wind_farm_env import WindFarmEnv
        env = WindFarmEnv(**self.kwargs)
        env.seed(self.seed)
        return env


def _worker(connection, factory, gamma, threads):
    env = None
    try:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(threads)
        env = factory()

        def snapshot():
            scenario = getattr(env, "wind_scenario", None)
            return (env.get_centralized_obs(), env.get_individual_baseline_values(gamma),
                    None if scenario is None else scenario.to_dict())

        metadata = {name: getattr(env, name) for name in (
            "action_space", "observation_space", "share_observation_space", "dt",
            "remaining_time_index", "layout_d", "total_rated_power_w") if hasattr(env, name)}
        metadata["pid"] = os.getpid()
        connection.send((True, metadata))
        done = True
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                obs = env.reset()
                done = False
                result = (obs, snapshot())
            elif command == "step":
                if done:
                    raise RuntimeError("Reset workers after the policy update before stepping again.")
                result = env.step(payload)
                done = bool(all(result[2]))
                result = (*result, snapshot())
            else:
                raise ValueError(f"Unknown worker command: {command}")
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class _Snapshot(SimpleNamespace):
    """Parent-side read-only view of critic features and environment metadata."""

    def update(self, snapshot):
        self.central, self.reference, scenario = snapshot
        self.wind_scenario = (None if scenario is None else
                              SimpleNamespace(to_dict=lambda: scenario))

    def get_centralized_obs(self):
        return self.central

    def get_individual_baseline_values(self, gamma):
        return self.reference


class ProcessVecEnv:
    """Windows-safe process environments with a barrier after every step.

    Completing an episode leaves the worker idle at its terminal state. The
    runner updates the policy, then explicitly resets the active workers for the next
    round. The operating system schedules these processes across CPU cores.
    """

    def __init__(self, factories, gamma=1.0, threads=1, timeout=600.0):
        if not factories or threads < 1 or timeout <= 0:
            raise ValueError("Use at least one environment, positive threads and timeout.")
        self.timeout = timeout
        self.connections, self.processes, self._snapshots = [], [], []
        self.closed = False
        context = mp.get_context("spawn")
        try:
            with _thread_limits(threads):
                for factory in factories:
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker, args=(child, factory, gamma, threads))
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        raise
                    finally:
                        child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
            for i in range(len(factories)):
                self._snapshots.append(_Snapshot(**self._receive(i)))
            self.use_workers(len(factories))
            for name in ("action_space", "observation_space", "share_observation_space"):
                setattr(self, name, getattr(self.envs[0], name))
        except BaseException:
            self.close()
            raise

    @property
    def num_envs(self):
        return len(self.envs)

    def use_workers(self, count):
        """Select the workers for the next round while retaining all subprocesses."""
        if not 1 <= count <= len(self._snapshots):
            raise ValueError("Select at least one available episode worker.")
        self.envs = self._snapshots[:count]

    def _receive(self, index):
        connection, process = self.connections[index], self.processes[index]
        deadline = time.monotonic() + self.timeout
        while not connection.poll(0.1):
            if not process.is_alive():
                raise RuntimeError(f"Episode worker {index} exited with code {process.exitcode}.")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Episode worker {index} exceeded {self.timeout:g} seconds.")
        try:
            ok, result = connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"Episode worker {index} disconnected.") from exc
        if not ok:
            raise RuntimeError(f"Episode worker {index} failed:\n{result}")
        return result

    def reset(self):
        import numpy as np
        for connection in self.connections[:self.num_envs]:
            connection.send(("reset", None))
        results = [self._receive(i) for i in range(self.num_envs)]
        for env, (_, snapshot) in zip(self.envs, results):
            env.update(snapshot)
        return np.asarray([result[0] for result in results])

    def step(self, actions):
        import numpy as np
        if len(actions) != self.num_envs:
            raise ValueError("Provide actions for every episode worker.")
        for connection, action in zip(self.connections, actions):
            connection.send(("step", action))
        results = [self._receive(i) for i in range(self.num_envs)]
        for env, result in zip(self.envs, results):
            env.update(result[4])
        return tuple(np.asarray([result[i] for result in results]) for i in range(4))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (OSError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()
