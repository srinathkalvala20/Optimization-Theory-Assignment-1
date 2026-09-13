import numpy as np
from collections import deque

# ---------------------------------------------------------
# MODI METHOD FOR TRANSPORTATION PROBLEM
# ---------------------------------------------------------

cost = np.array([
    [19, 30, 50, 10],
    [70, 30, 40, 60],
    [40,  8, 70, 20]
], dtype=float)

# Initial solution obtained using VAM
allocation = np.array([
    [0, 0, 0, 7],
    [5, 0, 4, 0],
    [0, 8, 3, 7]
], dtype=float)

m, n = cost.shape


def find_potentials():

    u = [None] * m
    v = [None] * n

    # Assume u1 = 0
    u[0] = 0

    changed = True

    while changed:

        changed = False

        for i in range(m):
            for j in range(n):

                if allocation[i][j] > 0:

                    if u[i] is not None and v[j] is None:
                        v[j] = cost[i][j] - u[i]
                        changed = True

                    elif v[j] is not None and u[i] is None:
                        u[i] = cost[i][j] - v[j]
                        changed = True

    return u, v


def find_cycle(entering):

    start = ('r', entering[0])
    target = ('c', entering[1])

    graph = {}

    for i in range(m):
        graph[('r', i)] = []

    for j in range(n):
        graph[('c', j)] = []

    # Create graph using occupied cells
    for i in range(m):
        for j in range(n):

            if allocation[i][j] > 0:

                graph[('r', i)].append(('c', j))
                graph[('c', j)].append(('r', i))

    # BFS to find path
    queue = deque([start])
    parent = {start: None}

    while queue:

        node = queue.popleft()

        if node == target:
            break

        for neighbour in graph[node]:

            if neighbour not in parent:
                parent[neighbour] = node
                queue.append(neighbour)

    # Convert path to cells
    path = []
    node = target

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()

    cells = []

    for a, b in zip(path, path[1:]):

        if a[0] == 'r':
            cells.append((a[1], b[1]))
        else:
            cells.append((b[1], a[1]))

    return [entering] + cells


print("=" * 60)
print("             MODI METHOD")
print("=" * 60)

print("\nInitial VAM Allocation:")
print(allocation)

iteration = 0

while True:

    iteration += 1

    # Calculate u and v
    u, v = find_potentials()

    print("\n" + "-" * 60)
    print("Iteration", iteration)
    print("-" * 60)

    print("u values:", u)
    print("v values:", v)

    # Calculate opportunity costs
    delta = np.full((m, n), np.nan)

    for i in range(m):
        for j in range(n):

            if allocation[i][j] == 0:
                delta[i][j] = cost[i][j] - (u[i] + v[j])

    print("\nOpportunity Cost Matrix:")
    print(delta)

    # Check optimality
    minimum = np.nanmin(delta)

    if minimum >= 0:
        print("\nAll opportunity costs are non-negative.")
        print("Therefore, the solution is OPTIMAL.")
        break

    # Select most negative opportunity cost
    entering = np.unravel_index(
        np.nanargmin(delta),
        delta.shape
    )

    print(
        "\nEntering cell: S{} -> D{}".format(
            entering[0] + 1,
            entering[1] + 1
        )
    )

    # Find closed loop
    cycle = find_cycle(entering)

    print("Closed loop:", cycle)

    # Alternating + and -
    signs = []

    for k in range(len(cycle)):
        if k % 2 == 0:
            signs.append(1)
        else:
            signs.append(-1)

    # Find theta
    theta = min(
        allocation[i][j]
        for (i, j), sign in zip(cycle, signs)
        if sign == -1
    )

    print("Theta =", theta)

    # Update allocation
    for (i, j), sign in zip(cycle, signs):
        allocation[i][j] += sign * theta


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

total_cost = np.sum(cost * allocation)

print("\n" + "=" * 60)
print("              OPTIMAL SOLUTION")
print("=" * 60)

print("\nOptimal Allocation:")
print(allocation)

print("\nMinimum Transportation Cost =", total_cost)

print("\nMODI completed successfully.")
