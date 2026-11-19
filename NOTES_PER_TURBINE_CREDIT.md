# Per-turbine credit when the wind direction changes

Design notes, 2026-11-15. **Nothing here is implemented yet.** These notes are for
whoever implements the dynamic-direction training. Values marked *(verify)* are
starting guesses to check against the simulator.

## Personal Description
Aaron personally speaking here - My idea here is that when the network of turbines gets larger, we may want to reincorporate individual optimization critics because the randomness and noise will inhibit the learning from just a single overall critic. These are some approaches I considered and talked to my agent about. The rest of this is fully written by Opus 5.5, but I'm planning on likely implementing option 1, and weighting that modified individual critic and the shared overall critic just about equally (0.5 and 0.5)

## 1. The problem

Each actor learns from a mix of two signals, set by `--team-reward-weight`:
its own turbine's energy (weight 0) and the whole farm's energy (weight 1).

- **Own-turbine energy works against wake steering.** An upstream turbine that
  yaws loses its own power so that a downstream turbine gains more. In the
  2026-11-14 test (`rl_runs/mappo/hyp1_*`, constant 8 m/s from 270°), weight 1
  finished at +6.3% over the aligned baseline. Weight 0.5 peaked and then fell
  back to +0.6%, and weight 0 stayed near the baseline.
- **Farm energy is noisy for each actor.** Each turbine's signal also contains
  the effects of the other turbines' random exploration. That is manageable with
  4 turbines, but the noise grows with the number of turbines.
- **When the wind direction changes, so do the upstream/downstream roles.** In
  the square layout:

  | Wind from | Whose wake hits whom |
  |---|---|
  | 270° (west) | T1→T3, T2→T4 |
  | 180° (south) | T1→T2, T3→T4 |
  | 225° (southwest) | T1→T4 (7.1 D apart) |
  | 90° (east) | T3→T1, T4→T2 |
  | 0° (north) | T2→T1, T4→T3 |
  | 315° (northwest) | T2→T3 |

  At angles in between, a wake can partly or completely miss the next turbine.
- **Wakes take time to arrive.** A wake crosses 5 D (990 m) at 8 m/s in about
  124 s, or 31 steps of 4 s. The wind generator can change direction by up to
  min(2 × half-width, 180°) per knot interval. With the default ±15° that is
  0.25°/s; with ±90° it is 1.5°/s. So the wake reaching a turbine now may have
  left its source turbine when the wind was 30° to 180° different. The right
  question is not "who is upstream now?" but "whose wake is reaching this
  turbine now?"

A fixed weight per turbine (weight 1 for upstream turbines, lower for downstream
ones) assumes the roles never change, so it won't work once the direction varies.
Options 1 and 3 below don't make that assumption.

## 2. Option 1: wake-neighbourhood reward

**Definition.** Turbine i is rewarded for its own energy plus the energy of every
turbine its wake is currently reaching:

```text
r_i(t) = [ E_i(t) + sum over j in W_i(t) of E_j(t) ] / (10 MW * dt)
```

Here `E` is interval energy, and `W_i(t)` is the set of turbines that turbine i's
wake reaches at step t.

**Why it is sound.** Turbines outside `W_i` are not affected by turbine i, so
dropping them removes noise without changing what turbine i is optimizing. As
long as every turbine that i *does* affect is included at full weight, this
reward pushes turbine i the same way the farm reward does.

### Design rules

1. **Build `W_i` from OFF's observation points, not from the current wind
   direction.**
   - Each turbine carries a chain of 200 observation points
     (`turbine.observation_points.get_world_coord()`) that trace its wake as the
     wind carries it.
   - Row k was released k steps ago. Each point moves with the ambient wind that
     was in place when it was released (`wake_solver._get_wind_speeds_op`).
   - So the trail curves when the direction changes, and the wake delay is
     built in.
2. **Use yes/no membership with a generous margin.**
   - Turbine j belongs to `W_i` if j's rotor lies within a sideways margin of
     turbine i's trail, past the source rotor.
   - The trail is the undeflected wake centreline: FLORIS adds wake spreading
     and yaw deflection when it computes each turbine's inflow. The margin must
     cover both. Start with **2 D** *(verify)*.
   - Leaving out a turbine that is affected makes the reward more selfish.
     Including one that isn't affected only adds noise. So err wide.
   - Don't use partial weights between 0 and 1: a turbine that is affected
     would then be undercounted.
3. **Membership must not depend on turbine i's yaw.** It doesn't: the trail
   moves with the stored ambient wind, not with wake-induced flow. So the
   membership rule itself never rewards any particular yaw.
4. **Keep the normalisation constant.** Always divide by 10 MW × dt. Don't divide
   by the size of `W_i`, or the reward will jump whenever membership changes.
5. **Let the critics see what decides membership.**
   - The reward changes when `W_i` changes, and a critic can only absorb that if
     it can predict it.
   - Add the membership matrix (N² entries, 16 for the square) and the ambient
     direction history (sin and cos at the existing 5–40 step lags) to the
     centralized state.
   - This changes the state size, so old checkpoints won't load. Bump the
     checkpoint format.
6. **Update the individual reference value.**
   - `get_individual_baseline_values` should return turbine i's own aligned
     power plus the aligned power of each turbine in `W_i`, each at its current
     local wind speed, times the remaining discounted horizon.
   - This assumes the current membership persists. The learned residual covers
     the rest.

With an aligned neighbourhood reward there is no conflict left to balance.
Weight 0 (neighbourhood only) is the natural setting, and mixing in the farm
reward only adds noise. Re-run the weight comparison to confirm.

### Where the code goes

| File | Change |
|---|---|
| `wind_farm_env.py` | Add `wake_membership()`, returning the membership mask and the wake age per pair. Call it in `step()` right after `_measure()`; OFF moves the observation points only after the measurement, so they match it. Compute the neighbourhood reward and log mask and age in `infos`. |
| `train_mappo.py` | Add `--individual-reward {own,neighbourhood}`, defaulting to `own` so existing runs still reproduce. Save it in `config.json`. |
| `wind_farm_env.py` → `get_centralized_obs()` | Add the mask and the direction history (rule 5). |
| `wind_farm_env.py` → `get_individual_baseline_values()` | Use the neighbourhood reference (rule 6). |
| `shared_critic_runner.py` | No change needed; it already takes per-turbine rewards from the environment. |

Sketch. On 2026-11-15 it was checked against OFF in constant wind. It returned
exactly the pairs in the §1 table at 270°, 180°, 225° and 0°, with wake ages
30.9 steps (5 D) and 43.8 steps (7.1 D). It has not been tried under a
direction ramp yet.

```python
def wake_membership(self, margin_d=2.0):
    """mask[i, j]: turbine i's wake trail passes within margin_d diameters of turbine j.
    age[i, j]: steps since that part of the wake left turbine i (nan if no contact)."""
    turbines = self.interface.wind_farm.turbines
    rotors = np.array([t.get_rotor_pos()[:2] for t in turbines])
    diameter = turbines[0].diameter
    n = len(turbines)
    mask, age = np.zeros((n, n), dtype=bool), np.full((n, n), np.nan)
    for i, source in enumerate(turbines):
        ops = source.observation_points.get_world_coord()[:, :2]  # row k released k steps ago
        start, seg = ops[:-1], np.diff(ops, axis=0)
        length2 = np.maximum((seg ** 2).sum(axis=1), 1e-9)
        for j in range(n):
            if j == i:
                continue
            frac = np.clip(((rotors[j] - start) * seg).sum(axis=1) / length2, 0.0, 1.0)
            gap = np.linalg.norm(start + frac[:, None] * seg - rotors[j], axis=1)
            k = int(np.argmin(gap))
            if gap[k] < margin_d * diameter and k + frac[k] > 0:
                mask[i, j], age[i, j] = True, k + frac[k]
    return mask, age
```

### Tests to write with it

- **Constant direction.** At 270° the mask should contain only T1→T3 and T2→T4;
  at 180°, T1→T2 and T3→T4; at 225°, T1→T4.
  - The wake age should be about 31 steps for 5 D at 8 m/s, and about 44 steps
    for 7.1 D.
  - OFF starts the trail straight downstream at reset, so contact exists from
    step 0.
- **Direction ramp from 270° to 180°.** The T1→T3 contact should end, and T1→T2
  begin, about one wake travel time after the ambient direction changes, not
  immediately.
- **Margin check.** For pairs outside the mask, zeroing the source turbine's yaw
  should not change the target turbine's power. That is option 3's
  counterfactual, so the same code can check the margin.

### Optional: credit the gain to when the wake left

`age[i, j]` says when the wake now reaching turbine j left turbine i. Training
waits for the full 200-step episode, so the downstream part of `r_i` could be
credited to step `t - age` instead of step `t`. That shortens the ~31-step gap
that GAE has to bridge, where it currently keeps only about 0.95^31 ≈ 0.2 of
the gain.

Caveats:
- Actions are yaw *rates*, so the yaw at release is the sum of all earlier
  actions, not the effect of one action.
- Wakes released in the last ~31 steps never arrive before the episode ends.
- It changes what a "return" means.

Run it as a separate experiment after plain option 1.

## 3. Option 3: difference reward

**Definition.** Turbine i's reward is the farm power minus what the farm would
have produced if turbine i had stayed aligned:

```text
D_i(t) = P_farm(t) - P_farm(t | turbine i aligned)
```

In words: how much turbine i's yaw choices added to the farm.

**Why it is attractive.**
- The subtracted term doesn't depend on turbine i's actions, so `D_i` pushes
  turbine i exactly the way the farm reward does. There is no conflict like the
  own-turbine reward's.
- The other turbines' exploration and the ambient wind changes appear in both
  terms and cancel. That leaves much less noise than the farm reward, with no
  margin to tune.
- `D_i = 0` for a turbine that stays aligned. That makes it a readable
  per-turbine "steering gain" metric, and a natural reference value of 0 for the
  individual critic.

**How to compute it.** There are two ways, and the choice decides whether it
works when the direction changes:

- **(a) Steady-state FLORIS on the current farm,** with turbine i's current yaw
  set to 0. That is N + 1 cheap solves per step.
  - Fine for constant wind.
  - **Misleading under large direction changes.** A steady-state solve assumes
    turbine i's current yaw already shapes the wake all the way downstream,
    along the *current* direction. In reality the effect arrives ~31 steps later
    and, with the wind turning, lands somewhere else. So (a) would reward yaw for
    gains that never happen and miss gains that do.
- **(b) Inside OFF's own wake calculation.**
  - For each target turbine j, OFF already builds a small temporary wind farm
    from each upstream turbine's state *at the moment its wake (now at j) was
    released* (`wake_solver._get_wind_speeds_rp`).
  - Re-solving that farm with turbine i's released yaw set to 0 gives j's power
    without turbine i's steering. This uses the same model and history as the
    real measurement, so wake delay and curved wakes are handled.
  - For turbine i's own term, use its power with its current yaw set to 0:

  ```text
  D_i(t) = [P_i - P_i(own yaw = 0)] + sum over j != i of [P_j - P_j(i's released yaw = 0)]
  ```

**Costs and risks of (b).**
- **Extra wake solves.** One per (i, j) pair per step: N(N−1) = 12 for the
  square, on top of the N that OFF already does. Using option 1's mask to skip
  pairs whose wake doesn't reach cuts this to typically 2. Measure the cost
  before committing, since simulation is already the bottleneck (about 67 s per
  episode).
- **Depends on OFF internals.** It needs a private method and the interpolated
  state objects. The bundled OFF source is meant to stay unmodified, so
  implement it as a subclass or wrapper of the wake solver in this repo. It may
  break if OFF changes.
- **Approximations.**
  - It is a one-step counterfactual: the other turbines' states are taken as
    they actually were, although they reacted slightly to turbine i's wake.
  - When wakes overlap, the `D_i` don't add up to the total farm gain, but each
    still pushes its turbine the right way.
- **The delay remains.** The gain at turbine j is still credited ~31 steps after
  the yaw. The observation-point index from (b) gives that delay, so the
  delay-crediting idea in §2 applies here too.

**Verdict.**
- Option 3(b) *would* work under large direction changes, and it gives the
  cleanest per-turbine signal of the options. The price is more simulation time
  and code that depends on OFF internals.
- Option 3(a) would *not* work under large direction changes.

## 4. Suggested order

1. Implement option 1 (it is cheap and needs no OFF changes).
2. Add `D_i` from option 3(b) as a **logged diagnostic only**, skipping pairs
   outside option 1's mask. It checks option 1's margin and reports each
   turbine's steering gain.
3. Train on `D_i` only if option 1's results are still noisy.

Compare these with matched seeds (2–3) and the same direction settings:
farm reward (weight 1), neighbourhood reward (weight 0), and neighbourhood plus
farm (weight 0.5). Evaluate greedy farm energy against the yaw-tracking baseline
(`--evaluate-aligned-baseline`) on the held-out evaluation wind.
