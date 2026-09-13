import numpy as np

# Transportation cost matrix
cost = np.array([
    [19, 30, 50, 10],
    [70, 30, 40, 60],
    [40,  8, 70, 20]
], dtype=float)

# Supply at each source
supply = [7, 9, 18]

# Demand at each destination
demand = [5, 8, 7, 14]

m, n = cost.shape

# Allocation matrix
allocation = np.zeros((m, n))

s = supply.copy()
d = demand.copy()

print("=" * 60)
print("       VOGEL'S APPROXIMATION METHOD (VAM)")
print("=" * 60)

print("\nCost Matrix:")
print(cost)

print("\nSupply:", supply)
print("Demand:", demand)

# VAM procedure
while sum(s) > 0 and sum(d) > 0:

    row_penalty = [-1] * m
    col_penalty = [-1] * n

    # Calculate row penalties
    for i in range(m):
        if s[i] > 0:
            values = [
                cost[i][j]
                for j in range(n)
                if d[j] > 0
            ]

            if len(values) >= 2:
                values.sort()
                row_penalty[i] = values[1] - values[0]
            elif len(values) == 1:
                row_penalty[i] = values[0]

    # Calculate column penalties
    for j in range(n):
        if d[j] > 0:
            values = [
                cost[i][j]
                for i in range(m)
                if s[i] > 0
            ]

            if len(values) >= 2:
                values.sort()
                col_penalty[j] = values[1] - values[0]
            elif len(values) == 1:
                col_penalty[j] = values[0]

    # Find maximum penalty
    max_row = max(row_penalty)
    max_col = max(col_penalty)

    # Select row or column
    if max_row >= max_col:

        i = row_penalty.index(max_row)

        available = [j for j in range(n) if d[j] > 0]

        j = min(available, key=lambda x: cost[i][x])

    else:

        j = col_penalty.index(max_col)

        available = [i for i in range(m) if s[i] > 0]

        i = min(available, key=lambda x: cost[x][j])

    # Allocation
    quantity = min(s[i], d[j])

    allocation[i][j] += quantity

    s[i] -= quantity
    d[j] -= quantity

    print(
        f"\nAllocate {quantity} units from S{i+1} to D{j+1}"
    )

# Calculate total transportation cost
total_cost = np.sum(allocation * cost)

print("\n" + "=" * 60)
print("           INITIAL BASIC FEASIBLE SOLUTION")
print("=" * 60)

print("\nAllocation Matrix:")
print(allocation)

print("\nInitial Transportation Cost =", total_cost)

print("\nSupply remaining:", s)
print("Demand remaining:", d)

print("\nVAM completed successfully.")
