import streamlit as st  # importing for web publishing
import sympy as sp  # Symbolic Python
from sympy import Matrix, symbols, binomial, factorial, simplify  # To create matrix, store symbols, compute binomial, factorial and simplify to get the candidate cyclic vector (c)

# Recursive c(i, j)
def compute_cij_table(n, D, basis):
    cij = {}
    # Step 1: c(0, j) using the formula
    for j in range(n):
        vec = Matrix.zeros(n, 1)  # creates a vector
        for k in range(j + 1):
            coeff = (-1)**k * binomial(j, k)
            ej_k = basis[j - k]
            term = ej_k
            for _ in range(k):
                term = D * term
            vec += coeff * term
        cij[(0, j)] = vec  # the cyclic vector c(0,j)

    # Step 2: c(i+1, j) = D(c(i, j)) + c(i, j+1)
    for i in range(1, n):
        for j in range(n):
            left = D * cij.get((i - 1, j), Matrix.zeros(n, 1))
            right = cij.get((i - 1, j + 1), Matrix.zeros(n, 1))
            cij[(i, j)] = left + right
    return cij

# D^i(c(e, t - a))
def compute_Di_c(i, t_sym, a_val, cij_table):
    vec = Matrix.zeros(cij_table[(0, 0)].rows, 1)
    x = t_sym - a_val
    for j in range(len([k for k in cij_table.keys() if k[0] == 0])):
        coeff = (x**j) / factorial(j)
        term = cij_table.get((i, j), Matrix.zeros(vec.rows, 1))
        vec += term.multiply(coeff)
    return simplify(vec)

# Katz vector and derivatives
def get_katz_derivatives(n, t_sym, a_val, D, basis):
    cij_table = compute_cij_table(n, D, basis)
    derivatives = [compute_Di_c(i, t_sym, a_val, cij_table) for i in range(n)]
    return derivatives  # Computes if P(x) is cyclic or not (i.e. |P(x)| = 0 or not)

# Check cyclicity of a list of vectors with if |P(x)| is 0 or not
def check_cyclicity(derivatives, t_sym, t_val):
    mat = Matrix.hstack(*derivatives)
    mat_eval = mat.subs(t_sym, t_val).evalf()
    rank = mat_eval.rank()
    if mat_eval.shape[0] == mat_eval.shape[1]:
        det = mat_eval.det().evalf()
    else:
        det = "N/A"
    return rank, det, mat_eval

# === Streamlit UI ===
st.title(" Katz Cyclic Vector Visualizer ")
n = st.sidebar.selectbox("Select rank n", [3])
t_sym = symbols('t')
t_val = st.sidebar.number_input("Enter fixed value of t", value=1.0)
k = 1 + n * (n - 1)
st.sidebar.markdown(f"**Katz: 1 + n(n−1) = {k} values of a needed**")

# === Inputs ===
D_input = st.sidebar.text_area("Enter D matrix ")  # Entering the D of the (V,D) where D:V->V
basis_input = st.sidebar.text_area("Enter basis vectors e ")  # Entering the e basis
default_a_list = ",".join([str(i + 1) for i in range(k)])
a_input = st.sidebar.text_input(f"Enter {k} values of a (comma-separated)", value=default_a_list)

# === Compute Katz vector and check cyclicity ===
try:
    D = Matrix([
        [sp.sympify(x.strip()) for x in row.strip().split(',') if x.strip()]
        for row in D_input.strip().split(';') if row.strip()
    ])

    basis = [
        Matrix([[sp.sympify(x.strip())] for x in row.strip().split(',') if x.strip()])
        for row in basis_input.strip().split(';') if row.strip()
    ]

    a_vals = [sp.sympify(a.strip()) for a in a_input.split(',') if a.strip()]
    if len(a_vals) != k:
        st.error(f" Enter exactly {k} values of a.")
    else:
        for i, a_val in enumerate(a_vals):
            st.markdown(f"---")
            st.subheader(f" Case {i+1}: a = {a_val}, X = t - a = {t_val - a_val}")
            derivs = get_katz_derivatives(n, t_sym, a_val, D, basis)
            for j, dvec in enumerate(derivs):
                st.write(f"**D^{j}(c)**:")
                st.write(dvec)
            rank, det, mat = check_cyclicity(derivs, t_sym, t_val)
            st.write(f"**Cyclicity Check at t = {t_val}**")
            st.write(f"Rank: {rank}")
            st.write(f"Determinant: {det}")
            st.write("Matrix:")
            st.write(mat)
            if rank == n:
                st.success(" Katz vector is cyclic.")
            else:
                st.error(" Katz vector is not cyclic.")
except Exception as e:
    st.error(f"Error: {e}")

# === User-defined vector v cyclicity check for v an element of V ===
st.markdown("---")
st.subheader(" Cyclicity Check for Arbitrary Vector v ∈ V")
user_v_input = st.text_input("Enter vector v (comma-separated)")
try:
    v = Matrix([sp.sympify(x.strip()) for x in user_v_input.split(',') if x.strip()])
    if v.shape[0] != n:
        st.error(f" Vector v must be of length {n}")
    else:
        vlist = [v]
        for _ in range(1, n):
            vlist.append(D * vlist[-1])
        mat_v = Matrix.hstack(*vlist)
        mat_v_eval = mat_v.subs(t_sym, t_val).evalf()
        rank_v = mat_v_eval.rank()
        if mat_v_eval.shape[0] == mat_v_eval.shape[1]:
            det_v = mat_v_eval.det().evalf()
        else:
            det_v = "N/A"
        st.write("Matrix formed by [v, Dv, D²v, ...]:")
        st.write(mat_v_eval)
        st.write(f"Rank: {rank_v}")
        st.write(f"Determinant: {det_v}")
        if rank_v == n:
            st.success(" Vector v is cyclic.")
        else:
            st.error(" Vector v is not cyclic.")
except Exception as e:
    st.error(f"Error parsing vector: {e}")
