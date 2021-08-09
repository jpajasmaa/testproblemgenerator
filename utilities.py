import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from typing import Tuple


def get_2D_version(x, pi1, pi2):
    """
    Project n > 2 dimensional vector to 2-dimensional space

    Args:
        x (np.ndarray): A given vector to project to 2-dimensional space
    
    Returns:
        np.ndarray: A 2-dimensional vector
    """
    if (x.shape[1] <= 2):
        print("Skipping projection, vector already 2 dimensional or less")
        return x
    l = np.divide(np.dot(x, pi1), np.sum(pi1)) # Left side of vector
    r = np.divide(np.dot(x, pi2), np.sum(pi2)) # Right side of vector
    return np.hstack((l, r))

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1-x2, axis = -1)
    #return np.sqrt(np.power(np.sum((x1-x2),axis = -1), 2))

def repmat(t, x, y): # could do this...
    pass 

def convhull(points):
    """
    Construct a convex hull of given set of points

    Args:
        points (np.ndarray): the points used to construct the convex hull
    
    Returns:
        np.ndarray: The indices of the simplices that form the convex hull
    """
    # TODO validate that enough unique points and so on
    hull = ConvexHull(points)
    return hull


def in_hull(x: np.ndarray, points: np.ndarray):
    """
    Is a point inside a convex hull 

    Args:
        x (np.ndarray): The point that is checked
        points (np.ndarray): The point cloud of the convex hull
    
    Returns:
        bool: is x inside the convex hull given by points 
    """
    p = (np.concatenate(points)) # wont work for most cases?
    n_points = len(p)
    c = np.zeros(n_points)
    A = np.r_[p.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def get_random_angles(n):
    return np.random.rand(n,1) * 2 * np.pi


def in_region(centres, radii, x) -> Tuple[bool, np.ndarray]:
        if centres is None or len(centres) < 1: return (False, np.array([]))
        dist = euclidean_distance(centres, x)
        in_region = np.any(dist <= radii)
        return (in_region, dist)
    
# this and above could be combined/one method
def in_region_excluding_boundary(centres, radii, x):
    if (centres is None or len(centres) < 1): return (False, np.array([]))
    d = euclidean_distance(centres, x)
    in_region = np.any(d < radii)
    return (in_region, d)


def between_lines_rooted_at_pivot(x, pivot_loc, loc1, loc2) -> bool:
    """
    Plaaplaa
    """
    d1 = ( x[0] - pivot_loc[0])*(pivot_loc[1] - pivot_loc[1])
    - (x[1] - pivot_loc[1])*(loc1[0] - pivot_loc[0])

    d2 = ( x[0] - pivot_loc[0])*(pivot_loc[1] - pivot_loc[1])
    - (x[1] - pivot_loc[1])*(loc2[0] - pivot_loc[0])

    return d1 == 0 or d2 == 0 or np.sign(d1) != np.sign(d2)