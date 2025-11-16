import .common

open real complex
open topological_space
open filter
open_locale real topology big_operators complex_conjugate filter

noncomputable theory

theorem Rudin_ex_1_1a
  (x : ℝ) (y : ℚ) : irrational x → irrational (x + y) := by
  intro hx
  simpa using hx.add_rat (y : ℝ)
