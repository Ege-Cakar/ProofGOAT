```lean
import Mathlib.Data.Real.Irrational
import Mathlib.Tactic.Ring

open Real

theorem sum_of_irrational_and_rational_is_irrational (x : ℝ) (y : ℚ) : irrational x → irrational (x + y) := by
  intro hx_irrational
  by_contra h_sum_rational
  simp only [irrational_iff_not_rational] at h_sum_rational
  have hy_rational : rational y := rational_coe_rat y
  have hx_rational : rational x := by
    convert rational.sub h_sum_rational hy_rational using 1
    ring
  exact hx_irrational hx_rational

```