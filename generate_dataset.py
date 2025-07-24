# === scripts/generate_dataset.py ===
# Author: ATRII ROY (Summer Research Program 2025, IISER Mohali)
# Purpose: Generate a labeled dataset using Katz’s cyclic vector algorithm
#          for training supervised ML models to predict cyclicity of vectors
# Date: July 2025
# IMPORTING REQUIRED LIBRARIES
import sympy as sp                  # For symbolic math (binomial, factorial, matrix algebra)
import random                      # For generating random matrices and basis
import csv                         # For writing dataset into CSV format
import os                          # For creating directories and file paths
from sympy import Matrix, symbols, binomial, factorial, simplify
# FUNCTION: compute_cij_table
# PURPOSE: Compute the recursive Katz table of vectors c(i, j)
def compute_cij_table(n, D, basis):
    """
    Builds recursive table of vectors c(i, j) as defined by Katz's algorithm.
    c(0, j) is built directly from the basis and D.
    c(i+1, j) is computed recursively as D(c(i, j)) + c(i, j+1).
    Parameters:
    - n (int): Dimension of the module
    - D (Matrix): The connection matrix (∂ operator)
    - basis (List[Matrix]): A list of basis column vectors of shape (n x 1)
    Returns:
    - cij (dict): Dictionary mapping (i, j) to corresponding Katz vector
    """
    cij = {}
    # Compute c(0, j) using the binomial formula
    for j in range(n):
        vec = Matrix.zeros(n, 1)
        for k in range(j + 1):
            coeff = (-1)**k * binomial(j, k)
            ej_k = basis[j - k]
            term = ej_k
            for _ in range(k):    # Apply D^k to ej−k
                term = D * term
            vec += coeff * term
        cij[(0, j)] = vec
    # Recursively compute higher entries c(i+1, j) = D(c(i, j)) + c(i, j+1)
    for i in range(1, n):
        for j in range(n):
            left = D * cij.get((i - 1, j), Matrix.zeros(n, 1))
            right = cij.get((i - 1, j + 1), Matrix.zeros(n, 1))
            cij[(i, j)] = left + right

    return cij
# FUNCTION: compute_Di_c
# PURPOSE: Computes the i-th derivative of the Katz vector symbolically
def compute_Di_c(i, t_sym, a_val, cij_table):
    """
    Computes the Katz vector D^i(c(t - a)) from the c(i, j) table.

    Parameters:
    - i (int): Order of derivative
    - t_sym (symbol): Symbolic variable t
    - a_val (float/int): Shift value 'a' used in Katz’s method
    - cij_table (dict): Precomputed c(i, j) values

    Returns:
    - (Matrix): Symbolic vector representing D^i(c(t - a))
    """
    vec = Matrix.zeros(cij_table[(0, 0)].rows, 1)
    x = t_sym - a_val
    for j in range(len([k for k in cij_table.keys() if k[0] == 0])):
        coeff = (x**j) / factorial(j)
        vec += coeff * cij_table[(i, j)]
    return simplify(vec)
# FUNCTION: get_katz_derivatives
# PURPOSE: Returns [c, Dc, D²c, ..., Dⁿ⁻¹c] as symbolic vector columns
def get_katz_derivatives(n, t_sym, a_val, D, basis):
    """
    Wrapper function to get full list of derivatives.

    Returns:
    - List[Matrix]: List of symbolic vectors
    """
    cij_table = compute_cij_table(n, D, basis)
    return [compute_Di_c(i, t_sym, a_val, cij_table) for i in range(n)]
# FUNCTION: check_cyclicity
def check_cyclicity(derivatives, t_sym, t_val):
    """
    Determines if the Katz matrix has full rank after evaluation.
    Returns:
    - int: 1 if cyclic (full rank), 0 otherwise
    """
    mat = Matrix.hstack(*derivatives)
    mat_eval = mat.subs(t_sym, t_val).evalf()
    return int(mat_eval.rank() == mat_eval.shape[0])
# Helper Functions for Feature Flattening (to store as CSV)
def flatten_matrix(mat):
    return [float(val.evalf()) for row in mat.tolist() for val in row]
def flatten_basis(basis):
    return [float(val.evalf()) for vec in basis for val in vec]
# FUNCTION: generate_random_D
# PURPOSE: Generate random connection matrix D with entries in [0,4]
def generate_random_D(n=3):
    return Matrix([[random.randint(0, 4) for _ in range(n)] for _ in range(n)])
# FUNCTION: generate_random_basis
# PURPOSE: Randomly generate a basis, with 20% chance of being degenerate
def generate_random_basis(n=3):
    if random.random() < 0.2:
        # Create a rank-deficient basis with one zero vector
        basis = [Matrix([[1 if i == j else 0] for i in range(n)]) for j in range(n)]
        basis[1] = Matrix.zeros(n, 1)  # Zero column → guaranteed non-cyclic
        return basis
    else:
        # Random binary basis vectors
        return [Matrix([[random.randint(0, 1)] for _ in range(n)]) for _ in range(n)]
# FUNCTION: generate_dataset
# PURPOSE: Main driver to generate symbolic data labeled as cyclic / non-cyclic
def generate_dataset(output_file='katz_cyclic_vector_ml/data/raw/samples_n3.csv', num_samples=10000, desired_ratio=0.5):
    """
    Generates a labeled symbolic dataset by computing Katz cyclic vectors.
    Each row contains:
    - Flattened D matrix (n^2 features)
    - Flattened basis matrix (n^2 features)
    - Scalar a
    - Scalar t (fixed at 1.0)
    - Label (1 if cyclic, 0 otherwise)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    t_sym = symbols('t')
    t_val = 1.0
    n = 3
    samples = []
    count_cyclic = 0
    count_noncyclic = 0
    target_cyclic = int(num_samples * desired_ratio)
    target_noncyclic = num_samples - target_cyclic
    attempts = 0
    max_attempts = 3000000  # Large cap to allow enough retries for balanced classes
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            D = generate_random_D(n)
            basis = generate_random_basis(n)
            a = random.randint(1, 5)
            derivs = get_katz_derivatives(n, t_sym, a, D, basis)
            label = check_cyclicity(derivs, t_sym, t_val)
            # Enforce class balance
            if (label == 1 and count_cyclic >= target_cyclic) or \
               (label == 0 and count_noncyclic >= target_noncyclic):
                continue
            row = flatten_matrix(D) + flatten_basis(basis) + [a, t_val, label]
            samples.append(row)
            if label == 1:
                count_cyclic += 1
            else:
                count_noncyclic += 1
            print(f"✅ Sample {len(samples)} | Label={label} | Cyclic={count_cyclic} Non-Cyclic={count_noncyclic}")
        except Exception as e:
            print(f"⚠️ Skipped due to error: {e}")
            continue
    # Save Collected Data to CSV File
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'D_{i}{j}' for i in range(n) for j in range(n)] + \
                 [f'e{i}_{j}' for i in range(n) for j in range(n)] + \
                 ['a', 't', 'cyclic']
        writer.writerow(header)
        writer.writerows(samples)

    print(f"\n🎯 Dataset saved to: {output_file}")
    print(f"📊 Total: {len(samples)} | Cyclic: {count_cyclic}, Non-Cyclic: {count_noncyclic}")
# Entry Point for Command-Line Execution
if __name__ == "__main__":
    generate_dataset()
