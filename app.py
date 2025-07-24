#MODULE IMPORTS
# Streamlit is a lightweight web app framework used here to create an interactive GUI
import streamlit as st
# SymPy is a symbolic mathematics library in Python for algebraic computations
import sympy as sp
from sympy import Matrix, symbols, binomial, factorial, simplify
# PART 1: FUNCTION DEFINITIONS
# FUNCTION: compute_cij_table
# Description:
#   Constructs the recursive coefficient table c(i, j), which is the cornerstone of Katz’s algorithm.
#   These coefficients are recursively used to generate the candidate cyclic vector and its derivatives.
# Parameters:
#   n     - Rank of the differential module V
#   D     - Matrix representing the connection D: V → V
#   basis - List of basis vectors [e0, e1, ..., en-1] for the free R-module V
# Returns:
#   A dictionary mapping tuple (i, j) to symbolic column vectors c(i, j)
def compute_cij_table(n, D, basis):
    cij = {}
    # First construct the base level: c(0, j)
    # Katz's formula: c(0,j) = Σ_{k=0}^{j} (-1)^k * binom(j, k) * D^k(e_{j-k})
    for j in range(n):
        vec = Matrix.zeros(n, 1)  # Initialize a column vector of size n × 1
        for k in range(j + 1):
            coeff = (-1)**k * binomial(j, k)      # Alternating binomial coefficient
            ej_k = basis[j - k]                   # Corresponding basis vector
            term = ej_k
            for _ in range(k):                    # Compute D^k(e_{j-k})
                term = D * term
            vec += coeff * term                   # Weighted sum
        cij[(0, j)] = vec                         # Store the result in the dictionary
    # Now recursively compute c(i+1, j) = D(c(i, j)) + c(i, j+1)
    for i in range(1, n):  # Up to c(n-1, j)
        for j in range(n):
            left = D * cij.get((i - 1, j), Matrix.zeros(n, 1))
            right = cij.get((i - 1, j + 1), Matrix.zeros(n, 1))
            cij[(i, j)] = left + right            # Recursive step from Katz’s paper
    return cij

# FUNCTION: compute_Di_c
# Description:
#   Computes the i-th derivative D^i(c) of the Katz vector c using Taylor expansion.
#   This is used to construct the matrix [c, Dc, ..., D^{n-1}c] needed for checking cyclicity.
def compute_Di_c(i, t_sym, a_val, cij_table):
    vec = Matrix.zeros(cij_table[(0, 0)].rows, 1)  # Empty vector of same shape
    x = t_sym - a_val  # Shifted variable X = t - a

    # Construct D^i(c) = Σ_{j} X^j / j! * c(i, j)
    for j in range(len([k for k in cij_table.keys() if k[0] == 0])):
        coeff = (x**j) / factorial(j)  # Taylor expansion coefficient
        term = cij_table.get((i, j), Matrix.zeros(vec.rows, 1))
        vec += term.multiply(coeff)    # Multiply each term and add to total
    return simplify(vec)
# FUNCTION: get_katz_derivatives
# Description:
#   Builds the full set of derivatives [c, Dc, D²c, ..., D^{n-1}c]
#   These vectors together will be tested for linear independence
def get_katz_derivatives(n, t_sym, a_val, D, basis):
    cij_table = compute_cij_table(n, D, basis)
    return [compute_Di_c(i, t_sym, a_val, cij_table) for i in range(n)]
# FUNCTION: check_cyclicity
# Description:
#   Evaluates the symbolic matrix formed by [c, Dc, ..., D^{n-1}c] at t = t₀
#   Then checks its rank and (if square) its determinant to determine cyclicity
def check_cyclicity(derivatives, t_sym, t_val):
    mat = Matrix.hstack(*derivatives)          # Stack vectors horizontally to form matrix
    mat_eval = mat.subs(t_sym, t_val).evalf()  # Substitute symbolic t with numerical t₀
    rank = mat_eval.rank()                     # Check rank
    # If square matrix, also compute determinant
    if mat_eval.shape[0] == mat_eval.shape[1]:
        det = mat_eval.det().evalf()
    else:
        det = "N/A"
    return rank, det, mat_eval
# PART 2: STREAMLIT FRONTEND 
# Title
st.title("📐 Katz Cyclic Vector Visualizer")
# This is a symbolic visual app that shows how Katz vectors are constructed and tested for cyclicity.
# Sidebar Setup
n = st.sidebar.selectbox("Select rank n", [3])  # We fix n = 3 for this experiment
t_sym = symbols('t')  # Symbolic variable
t_val = st.sidebar.number_input("Enter fixed value of t", value=1.0)
k = 1 + n * (n - 1)  # Number of shift values 'a' needed for full generality (from Katz theorem)
st.sidebar.markdown(f"**Katz: requires {k} distinct values of a**")
# Inputs for matrix D and basis e vectors
D_input = st.sidebar.text_area("Enter D matrix (semicolon-separated rows, comma-separated entries)")
basis_input = st.sidebar.text_area("Enter basis vectors e (semicolon-separated rows, comma-separated entries)")
# Default values of a: 1,2,...,k
default_a_list = ",".join([str(i + 1) for i in range(k)])
a_input = st.sidebar.text_input(f"Enter {k} values of a (comma-separated)", value=default_a_list)
# === MAIN LOGIC: Part A - Katz Vector Cyclicity ===
try:
    # Parse D matrix
    D = Matrix([
        [sp.sympify(x.strip()) for x in row.strip().split(',') if x.strip()]
        for row in D_input.strip().split(';') if row.strip()
    ])
    # Parse basis vectors
    basis = [
        Matrix([[sp.sympify(x.strip())] for x in row.strip().split(',') if x.strip()])
        for row in basis_input.strip().split(';') if row.strip()
    ]
    # Parse shift values a
    a_vals = [sp.sympify(a.strip()) for a in a_input.split(',') if a.strip()]
    if len(a_vals) != k:
        st.error(f"Please enter exactly {k} values of a.")
    else:
        for i, a_val in enumerate(a_vals):
            st.markdown("---")
            st.subheader(f"Case {i+1}: a = {a_val},  X = t - a = {t_val - a_val}")
            derivs = get_katz_derivatives(n, t_sym, a_val, D, basis)
            # Display each D^j(c)
            for j, dvec in enumerate(derivs):
                st.write(f"**D^{j}(c)**:")
                st.write(dvec)
            # Cyclicity check: is [c, Dc, ...] of full rank?
            rank, det, mat = check_cyclicity(derivs, t_sym, t_val)
            st.write(f"**Cyclicity Check at t = {t_val}**")
            st.write(f"Rank: {rank}")
            st.write(f"Determinant: {det}")
            st.write("Matrix:")
            st.write(mat)
            if rank == n:
                st.success("✅ Katz vector is cyclic.")
            else:
                st.error("❌ Katz vector is not cyclic.")
except Exception as e:
    st.error(f"Error: {e}")
# === PART B: User-Defined Vector v ∈ V Cyclicity Check ===
st.markdown("---")
st.subheader("🔍 Cyclicity Check for Arbitrary Vector v ∈ V")
user_v_input = st.text_input("Enter vector v (comma-separated)")
try:
    v = Matrix([sp.sympify(x.strip()) for x in user_v_input.split(',') if x.strip()])
    if v.shape[0] != n:
        st.error(f"Vector v must be of length {n}")
    else:
        vlist = [v]
        for _ in range(1, n):
            vlist.append(D * vlist[-1])  # Generate Dv, D²v, ..., D^{n-1}v
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
            st.success("✅ Vector v is cyclic.")
        else:
            st.error("❌ Vector v is not cyclic.")
except Exception as e:
    st.error(f"Error parsing vector: {e}")
