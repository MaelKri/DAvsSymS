import numpy as np


def compute_curvature(nodes):
    """
    Compute curvature using three consecutive nodes in local coordinates.
    """
    if nodes is None or len(nodes) < 3:
        return 0  # Not enough points to compute curvature

    # Convert nodes into NumPy arrays
    try:
        p1 = np.array(nodes[0])[:2]  # Extract only X, Y
        p2 = np.array(nodes[len(nodes) // 2])[:2]
        p3 = np.array(nodes[-1])[:2]

        # Compute curvature as 1 / radius of the circle formed by three points
        A = np.array([
            [p1[0], p1[1], 1],
            [p2[0], p2[1], 1],
            [p3[0], p3[1], 1]
        ])
        B = np.array([
            [- (p1[0]**2 + p1[1]**2)],
            [- (p2[0]**2 + p2[1]**2)],
            [- (p3[0]**2 + p3[1]**2)]
        ])

        X = np.linalg.solve(A, B)
        xc, yc = -0.5 * X[0, 0], -0.5 * X[1, 0]
        radius = np.sqrt(xc**2 + yc**2 - X[2, 0])

        return 1 / radius if radius != 0 else 0

    except (np.linalg.LinAlgError, IndexError, ValueError):
        return 0  # If any error occurs, return zero curvature
