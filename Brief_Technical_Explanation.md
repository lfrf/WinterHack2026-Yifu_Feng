# Brief Technical Explanation

## 1) Overview
This block selects the **next target cell** for navigation in a partially known grid map. It is designed to (a) go **directly to the goal** as soon as this is safe and feasible, and (b) otherwise **continue exploration** with robust fallbacks that guarantee progress.

Key ideas:
- **Goal-readiness + quick A\***: if the goal region is sufficiently scanned and A* finds a path, use the **goal** immediately (even if it’s not a frontier).
- **Unknown-is-navigable policy**: only **occupied** cells block motion; **free** and **unknown** are allowed. This lets the robot push through lightly mapped areas near the goal.
- **Validation & fallbacks**: every chosen target is validated by a quick A*; if planning fails, a small **fallback chain** selects an alternative target that still moves us closer.

---

## 2) Inputs & Outputs
**Inputs (from `state` and helpers):**
- `state.ogm`: log-odds occupancy grid with `grid`, `res`, `minx/miny`, thresholds `prob_free_max`, `prob_occ_min`.
- `state.world`: `cell_size_m`, map size.
- `state.pose`: robot pose `(x, y, theta)`.
- `state.goal["cell"]`: goal cell in grid coordinates.
- Helpers: `cell_center`, `pose_to_cell`, `plan_unknown_world(state, start, target)`, `detect_frontiers(state)`.

**Output:**
- `best_frontier_cell` — the selected next target cell.

---

## 3) Algorithm (step-by-step)

### Step A. Goal readiness + quick A\*
1. Convert log-odds to probability (`prob = sigmoid(grid)`).
2. Check **goal cell known** (prob ≤ free_max or ≥ occ_min).
3. Inspect a **0.20 m radius** disk around the goal; compute **unknown ratio**. If unknown ratio < **0.50**, we say the goal area is “scanned enough”.
4. Run a **quick A\*** (`plan_unknown_world`) from current cell to the goal. If a path exists → **select the goal** immediately.

> Rationale: avoids waiting for the goal to be a frontier and does not require straight-line LOS; any feasible path suffices.

### Step B. Frontier scoring (if goal not ready)
1. Get `frontiers, distances = detect_frontiers(state)`.
2. If no frontiers → fall back to goal cell (later validated).
3. Otherwise score each frontier:
   - **Heading alignment** (prefer less turning).
   - **BFS path-cost gain** (closer from current cell).
   - **Goal bias** (closer to the final goal).
   Score = `0.5*align + 0.3*bfs_gain + 0.2*goal_gain`.
4. Pick the **max-score** frontier.

### Step C. Planning validation & fallback chain
Before committing, validate the chosen target with a quick **A\***:
- If **A\*** succeeds → use it.
- Else try fallbacks, in order:
  1) **Forward-hemisphere nearest frontier** (avoid backtracking).
  2) **Step one cell toward the goal** (x/y ±1) to reveal narrow gaps.
  3) **Any navigable neighbor** that strictly reduces goal distance.
  4) **Hold position** as a last resort.
Each fallback is validated with A* before acceptance.

---

## 4) Unknown-is-Navigable Policy
- Classification:
  - `p ≥ prob_occ_min` → **occupied (blocked)**
  - `p ≤ prob_free_max` → **free**
  - otherwise → **unknown (allowed)**
- Out-of-bounds during index checks is treated as free (conservative in sim; adjust for real robot).

**Effect:** the robot can enter partially known corridors and “finish the map” near the goal, rather than circling until every cell becomes free.

---

## 5) Important Tunables (typical)
- **Goal neighborhood radius:** `0.20 m`.
- **Unknown ratio threshold:** `0.50` (lower → more conservative; higher → more aggressive).
- **Frontier score weights:** `(align, bfs, goal) = (0.5, 0.3, 0.2)`.
- **Forward hemisphere cutoff:** `±90°` for fallback #1.

---

## 6) Edge Cases & Recoveries
- **No OGM or out-of-range indices:** treat as navigable to avoid false negatives in sim.
- **No frontiers & goal not ready:** step-toward-goal and neighbor fallbacks ensure progress.
- **Planner returns empty path:** immediate fallback prevents oscillation on infeasible targets.

---

## 7) Complexity & Runtime
A\* validations are **local and short** (grid size is small per episode); cost is dominated by 1–3 quick A\* calls per cycle. Frontier scoring is linear in the number of candidate frontiers.

---

## 8) Limitations & Future Work
- **Hand-tuned thresholds/weights**; could be adapted online (e.g., by success rate or info gain).
- **Unknown-is-navigable** can add small detours; adding **risk-aware costs** (inflate unknown near obstacles) would refine paths.
- Integrate **information-gain maps** or **ring sampling** for richer exploration when frontier quality is poor.
- Add **short-term memory** (tabu) and **anti-oscillation** filters for busy junctions.

---

## 9) Data Flow Summary
1. Read OGM + pose + goal.
2. **Goal check** (probability & unknown ratio) → **A\*** to goal → success? **Yes → goal**.
3. Else **frontier scoring** → pick best → **A\*** validate → success? **Yes → target**.
4. Else run **fallbacks** (each with A\* validation) → pick first success → otherwise **hold**.

This design makes the system **goal-aware**, **robust under partial maps**, and **progress-guaranteeing**.
