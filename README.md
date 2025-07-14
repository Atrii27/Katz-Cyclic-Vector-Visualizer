Katz Cyclic Vector Visualizer (n = 3)
This Streamlit app computes **cyclic vectors** in differential modules using **Katz’s algorithm** (symbolic approach only).
What It Does
- Accepts user input for:
  - A differential matrix **D**
  - Basis vectors **e₀, e₁, ..., eₙ₋₁**
  - Multiple shift values **a**
- Computes:
  - Katz derivative vectors: D⁰(c), D¹(c), ..., Dⁿ⁻¹(c)
  - Rank and determinant at a fixed value of **t**
- Checks if a Katz vector is **cyclic** based on matrix rank
How to Run
1. Install requirements:
```bash
pip install streamlit sympy
