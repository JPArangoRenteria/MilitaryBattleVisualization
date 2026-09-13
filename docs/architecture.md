# Lanchester Dynamics: Architecture and Transformation Plan

## Phase 0: Current Implementation Problems

### Critical Bugs

1. **Incorrect exponentiation operator (lines 28, 29, 32, 33)**
   - Uses `^` (bitwise XOR) instead of `**` (exponentiation)
   - `b^2` computes bitwise XOR, not b squared
   - Impact: Square law computations are mathematically incorrect

2. **Undefined variables in `simulation_result()` (line 93)**
   - Function calls `lanchesterlaw(y, ...)` but `y` is never defined or passed
   - Should use `initial_conditions` or similar
   - Impact: Function crashes at runtime

3. **Variable shadowing (line 96)**
   - Parameters include `t` which shadows the time array from outer scope
   - `lotka_volterra()` expects `(x, y, t, ...)` but receives function parameters incorrectly

4. **Incorrect ODE function construction**
   - `lanchesterlaw()` is defined to accept `(y, beta, alpha, ca, cb, helmbold, case_type)`
   - Called as `ode_f = lanchesterlaw(y, ...)` but never creates a closure or partial function
   - `solve_ode()` expects function signature `(y, t)` for use with `odeint()`
   - Impact: ODE solver cannot call the derivative function correctly

5. **Duplicated `compare_with_historical_data()` (lines 100-122)**
   - Two definitions with different behavior (one plots, one returns score)
   - Second definition overwrites the first

6. **Broken multi-force branch (lines 53-65)**
   - References undefined variables `a`, `b` instead of using `A`, `B`
   - Beta and alpha are treated as 1D arrays but indexed as 2D matrices

7. **No return statement in `simulation_result()` (line 89)**
   - Function computes solution but never returns it

8. **Problematic recursive parameter optimization (line 139)**
   - `optimize_parameters()` calls itself recursively without proper base case
   - Uses random perturbation without proper optimization algorithm
   - Can lead to stack overflow or infinite recursion

### Design Issues

1. **Mixing concerns**
   - Plotting code mixed with analysis
   - Optimization logic mixed with simulation
   - Parameter tuning mixed with model definition

2. **Unclear variable naming**
   - `helmbold` parameter purpose unclear
   - `ca`, `cb` ambiguous (coefficients? capacities?)
   - `case_type` mixed with model/instance control flow

3. **API confusion**
   - Lotka-Volterra model included but distinct from Lanchester models
   - Generalized models (logarithmic, exponential) included but underspecified
   - No clear mathematical documentation of these "generalized" models

4. **Missing mathematical documentation**
   - No definition of what "linear" vs "square" law means
   - Terminology varies across literature
   - Assumptions about helmbold modifier undocumented
   - No specification of units or interpretation

5. **Scipy API misuse**
   - Uses deprecated `odeint()` 
   - Modern practice: `solve_ivp()` with event handling
   - No event handling for force reaching zero

6. **No validation**
   - No tests
   - No analytical invariant checking
   - No convergence analysis

### Missing Features

- No project structure (no setup.py, no package layout)
- No dependencies declaration
- No type hints
- No docstrings (minimal)
- No tests
- No example that actually runs
- No metrics module

---

## Proposed Architecture

### Mathematical Conventions (Phase 1)

**Classical Lanchester Square Law:**
```
dA/dt = -β·B
dB/dt = -α·A
```

where:
- A(t) = strength of force A at time t
- B(t) = strength of force B at time t  
- α = effectiveness coefficient of force A
- β = effectiveness coefficient of force B

**Invariant:** The quantity I = α·A² - β·B² remains constant during engagement.

**Classical Lanchester Linear Law:**
```
dA/dt = -β
dB/dt = -α
```

where α and β are constant attrition rates (not dependent on opponent strength).

**Rationale for linear law:**
- Represents ancient warfare where each unit kills at a fixed rate (e.g., archers can only kill so many per time)
- Distinct from "linear" ODE solver behavior
- Not universally standardized terminology across literature; this project uses this explicit convention

### Package Structure

```
lanchester-dynamics/
├── pyproject.toml              # Build config, dependencies
├── README.md                   # User documentation
├── LICENSE                     # MIT or similar
├── .gitignore
│
├── src/
│   └── lanchester/
│       ├── __init__.py
│       ├── models.py           # Model definitions (Square, Linear, etc.)
│       ├── simulation.py       # ODE solver integration
│       ├── metrics.py          # MAE, RMSE, R²
│       └── validation.py       # Invariant checking, convergence tests
│
├── tests/
│   ├── test_linear.py
│   ├── test_square.py
│   ├── test_simulation.py
│   └── test_validation.py
│
├── examples/
│   └── basic_simulation.py     # Runnable example
│
└── docs/
    └── architecture.md         # This file
```

### Core Design Principles

1. **Mathematical Model Abstraction**
   - Abstract base class or protocol for models
   - `model.derivatives(t, state) -> state_derivatives`
   - Stateless: pure function behavior
   - No plotting, optimization, or historical data knowledge in models

2. **Numerical Solver Layer**
   - `simulate(model, t_span, y0, t_eval, events=None) -> SimulationResult`
   - Uses `scipy.integrate.solve_ivp()`
   - Handles event detection (force reaches zero)
   - Returns structured result object with t, y, status, message

3. **Validation and Metrics**
   - Separate module for analytical checks
   - `square_law_invariant(t, A, B, alpha, beta)` to verify theoretical invariant
   - Standard metrics: MAE, RMSE, R²

4. **Examples and Utilities**
   - Plotting in examples only, not in core
   - Parameter tuning deferred to Phase 2+
   - Historical fitting deferred to Phase 2+

### Type System

- Use Python dataclasses for configuration and results
- Type hints throughout
- No `Any` without justification

### Removed from Phase 1

- Logarithmic and exponential interaction models (Phase 2)
- Multi-force models (Phase 2)
- Lotka-Volterra model (out of scope; include as reference only if needed)
- Parameter optimization (Phase 2+)
- Historical data fitting (Phase 2+)
- Monte Carlo analysis (Phase 3+)
- Interactive interface (future phases)

---

## Migration Plan

### STEP 1: Inspect (DONE)
- Identified all bugs and design issues

### STEP 2: Create `docs/architecture.md` (DOING NOW)
- Document this plan
- Establish mathematical conventions

### STEP 3: Create package structure
- `pyproject.toml` (Python 3.11+, numpy, scipy, pytest, ruff)
- `src/lanchester/__init__.py`
- Empty module stubs

### STEP 4: Implement Linear Law model
- Class or dataclass for Linear Law configuration
- Pure derivative function

### STEP 5: Write and run Linear Law tests
- Test derivatives
- Test with toy ODE solver
- Boundary cases (zero rate, negative forces warning)

### STEP 6: Implement Square Law model
- Class or dataclass
- Derivative function
- Include invariant calculation helper

### STEP 7: Write and run Square Law tests
- Test derivatives
- Test invariant preservation (tolerance ~1e-4 to 1e-6)
- Symmetric cases
- Unequal initial conditions
- Zero effectiveness edge cases

### STEP 8: Implement simulation layer
- `SimulationResult` dataclass
- `simulate()` function wrapping `solve_ivp()`
- Event handlers for force reaching zero
- Clear solver status reporting

### STEP 9: Write and run simulation tests
- Convergence (finer mesh → consistent result)
- Event detection
- Status reporting

### STEP 10-11: Implement and test validation module
- `square_law_invariant()` function
- Tolerance handling

### STEP 12-13: Implement and test metrics
- MAE, RMSE, R²
- Unit tests with simple arrays

### STEP 14: Create basic example
- Load `basic_simulation.py`
- Run Square Law scenario
- Compute outcome
- Plot with matplotlib
- Print statistics

### STEP 15: Write comprehensive README
- Overview
- Mathematical models with equations
- Installation
- Quick start
- Numerical methods note
- Validation approach
- Roadmap (what's not included)

### STEP 16: Lint and type check
- `ruff check .`
- `mypy src/` if configured

### STEP 17: Run example
- Verify it produces sensible output
- Verify plot is generated

### STEP 18: Summary report
- Files created
- Bugs fixed
- Test results
- Lint results
- Example output
- Remaining limitations
- Recommendations for Phase 2

---

## Key Implementation Decisions

1. **solve_ivp over odeint**
   - Modern scipy API
   - Better event handling
   - Adaptive stepping strategies
   - Clearer parameter passing

2. **Dataclasses over hand-rolled dicts**
   - Type-safe
   - Clear interface contracts
   - Easy serialization later

3. **Pure functions for models**
   - Testable
   - Composable
   - No hidden state

4. **Explicit mathematical documentation**
   - Each model has its differential equations in docstrings
   - Assumptions listed
   - Invariants stated

5. **No generalization in Phase 1**
   - Focus on correct implementation of two canonical cases
   - Avoid premature abstraction
   - Extend systematically in Phase 2

---

## Testing Strategy

1. **Unit tests for each module**
   - Models: verify derivative calculations against hand-computed values
   - Simulation: verify solver integration, event detection
   - Validation: verify invariant calculation
   - Metrics: verify score calculations against simple examples

2. **Integration tests**
   - Full simulation workflow
   - Known analytical solutions
   - Comparison of different mesh densities

3. **No flaky tests**
   - Avoid random number seeding ambiguity
   - Use fixed tolerance values
   - Document numerical precision expectations

4. **Run all tests before proceeding to next step**

---

## Backwards Compatibility

This project is not backward compatible with the original `Lanchester.py`.
The original contained critical mathematical bugs (^ vs **) and design flaws that make faithful preservation impossible while also fixing them.

A clean break allows us to:
1. Fix the bugs
2. Document the mathematical model clearly
3. Provide a solid foundation for Phase 2

Users who want the old script can still find it in the git history.

