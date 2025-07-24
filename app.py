# === scripts/generate_dataset.py ===
import sympy as sp
import random
import csv
import os
from sympy import Matrix, symbols, binomial, factorial, simplify
def compute_cij_table(n, D, basis):
    # Compute c(i, j) vectors recursively for Katz's construction
    cij = {}
    for j in range(n):
        vec = Matrix.zeros(n, 1)  # Start with zero vector
        for k in range(j + 1):
            coeff = (-1)**k * binomial(j, k)  # Binomial coefficient with sign
            ej_k = basis[j - k]               # Select basis vector
            term = ej_k
            for _ in range(k):
                term = D * term               # Apply D k times
            vec += coeff * term               # Add term to vector
        cij[(0, j)] = vec                     # Store base case
    for i in range(1, n):
        for j in range(n):
            left = D * cij.get((i - 1, j), Matrix.zeros(n, 1))      # D applied to c(i-1, j)
            right = cij.get((i - 1, j + 1), Matrix.zeros(n, 1))     # c(i-1, j+1)
            cij[(i, j)] = left + right                              # Recursive sum
    return cij
def compute_Di_c(i, t_sym, a_val, cij_table):
    # Compute D^i(c(t - a)) using the c(i, j) table
    vec = Matrix.zeros(cij_table[(0,0)].rows, 1)
    x = t_sym - a_val
    for j in range(len([k for k in cij_table.keys() if k[0] == 0])):
        coeff = (x**j) / factorial(j)           # Taylor coefficient
        vec += coeff * cij_table[(i, j)]        # Weighted sum
    return simplify(vec)                        # Simplify result
def get_katz_derivatives(n, t_sym, a_val, D, basis):
    # Return list of derivatives [c, Dc, D^2c, ..., D^{n-1}c]
    cij_table = compute_cij_table(n, D, basis)
    derivatives = [compute_Di_c(i, t_sym, a_val, cij_table) for i in range(n)]
    return derivatives
def check_cyclicity(derivatives, t_sym, t_val):
    # Check if the set of derivatives forms a cyclic vector (full rank)
    mat = Matrix.hstack(*derivatives)
    mat_eval = mat.subs(t_sym, t_val).evalf()
    rank = mat_eval.rank()
    return int(rank == mat_eval.shape[0])       # 1 if cyclic, 0 otherwise
def flatten_matrix(mat):
    # Flatten a sympy Matrix into a 1D list of floats (row-wise)
    return [float(val.evalf()) for row in mat.tolist() for val in row]
def flatten_basis(basis):
    # Flatten a list of sympy column vectors into a 1D list of floats
    return [float(val.evalf()) for vec in basis for val in vec]
def generate_random_D(n=3):
    # Generate a random n x n integer matrix for D
    return Matrix([[random.randint(0, 4) for _ in range(n)] for _ in range(n)])
def generate_random_basis(n=3):
    # Generate a random basis (list of n column vectors)
    if random.random() < 0.2:
        # 20% chance to create a rank-deficient basis
        basis = [Matrix([[1 if i == j else 0] for i in range(n)]) for j in range(n)]
        basis[1] = Matrix.zeros(n, 1)  # Make one vector zero
        return basis
    else:
        return [Matrix([[random.randint(0, 1)] for _ in range(n)]) for _ in range(n)]
def generate_dataset(output_file='katz_cyclic_vector_ml/data/raw/samples_n3.csv', num_samples=10000, desired_ratio=0.5):
    # Generate a dataset of random (D, basis, a, t) with cyclicity labels and save to CSV
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
    max_attempts = 3000000  # Prevent infinite loop
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            D = generate_random_D(n)                  # Random D matrix
            basis = generate_random_basis(n)          # Random basis
            a = random.randint(1, 5)                  # Random shift valu
            derivs = get_katz_derivatives(n, t_sym, a, D, basis)  # Compute derivatives
            label = check_cyclicity(derivs, t_sym, t_val)         # Check cyclicity
            # Balance cyclic/non-cyclic samples
            if (label == 1 and count_cyclic >= target_cyclic) or (label == 0 and count_noncyclic >= target_noncyclic):
                continue
            row = flatten_matrix(D) + flatten_basis(basis) + [a, t_val, label]  # Prepare row
            samples.append(row)
            if label == 1:
                count_cyclic += 1
            else:
                count_noncyclic += 1
            print(f"✅ Sample {len(samples)} | Label={label} | Cyclic={count_cyclic} Non-Cyclic={count_noncyclic}")
        except Exception as e:
            print(f"⚠️ Skipped due to error: {e}")
            continue
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f'D_{i}{j}' for i in range(n) for j in range(n)] + \
                 [f'e{i}_{j}' for i in range(n) for j in range(n)] + \
                 ['a', 't', 'cyclic']
        writer.writerow(header)
        writer.writerows(samples)
    print(f"\n🎯 Dataset saved to: {output_file}")
    print(f"📊 Total: {len(samples)} | Cyclic: {count_cyclic}, Non-Cyclic: {count_noncyclic}")
if __name__ == "__main__":
    generate_dataset()  # Run dataset generation if script is executed directly