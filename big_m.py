import numpy as np

# Big-M value
M = 1000000

# ---------------------------------------------------------
# Problem:
# Maximize Z = 3x1 + 2x2
#
# Subject to:
# x1 + x2 >= 4
# 2x1 + x2 <= 6
# x1, x2 >= 0
# ---------------------------------------------------------

# Variables:
# x1, x2, s1, s2, A1

variable_names = ["x1", "x2", "s1", "s2", "A1"]

# Objective coefficients
C = np.array([3, 2, 0, 0, -M], dtype=float)

# Constraint matrix
A = np.array([
    [1, 1, -1, 0, 1],
    [2, 1,  0, 1, 0]
], dtype=float)

# RHS
b = np.array([4, 6], dtype=float)

# Initial basic variables: A1 and s2
basis = [4, 3]

iteration = 0


def calculate_zj():
    """Calculate Zj row."""
    zj = np.zeros(len(variable_names))

    for i in range(len(basis)):
        zj += C[basis[i]] * A[i]

    return zj


print("=" * 55)
print("          BIG-M SIMPLEX METHOD")
print("=" * 55)

print("\nProblem:")
print("Maximize Z = 3x1 + 2x2")
print("Subject to:")
print("x1 + x2 >= 4")
print("2x1 + x2 <= 6")
print("x1, x2 >= 0")

print("\nStandard Form:")
print("x1 + x2 - s1 + A1 = 4")
print("2x1 + x2 + s2 = 6")

while True:

    iteration += 1

    # Calculate Zj and Cj-Zj
    Zj = calculate_zj()
    Cj_minus_Zj = C - Zj

    print("\n" + "-" * 55)
    print("Iteration", iteration)
    print("-" * 55)

    print("Basic variables:",
          [variable_names[i] for i in basis])

    print("Zj =", np.round(Zj, 3))
    print("Cj-Zj =", np.round(Cj_minus_Zj, 3))

    # For maximization, choose largest positive Cj-Zj
    entering = np.argmax(Cj_minus_Zj)

    # If no positive value, optimum reached
    if Cj_minus_Zj[entering] <= 1e-9:
        break

    # Ratio test
    ratios = []

    for i in range(len(basis)):

        if A[i, entering] > 0:
            ratios.append(b[i] / A[i, entering])
        else:
            ratios.append(np.inf)

    leaving_row = np.argmin(ratios)

    if ratios[leaving_row] == np.inf:
        print("The problem is unbounded.")
        break

    print("Entering variable:",
          variable_names[entering])

    print("Leaving variable:",
          variable_names[basis[leaving_row]])

    # Pivot
    pivot = A[leaving_row, entering]

    A[leaving_row] = A[leaving_row] / pivot
    b[leaving_row] = b[leaving_row] / pivot

    # Make other entries in entering column zero
    for i in range(len(basis)):

        if i != leaving_row:

            factor = A[i, entering]

            A[i] = A[i] - factor * A[leaving_row]
            b[i] = b[i] - factor * b[leaving_row]

    # Update basis
    basis[leaving_row] = entering


# ---------------------------------------------------------
# Final Solution
# ---------------------------------------------------------

solution = np.zeros(len(variable_names))

for i in range(len(basis)):
    solution[basis[i]] = b[i]

optimal_value = np.dot(C, solution)

print("\n" + "=" * 55)
print("              OPTIMAL SOLUTION")
print("=" * 55)

print("x1 =", round(solution[0], 4))
print("x2 =", round(solution[1], 4))

print("Maximum Z =", round(optimal_value, 4))

print("\nTherefore:")
print("Optimal decision: x1 = 2, x2 = 2")
print("Maximum objective value: Z = 10")
