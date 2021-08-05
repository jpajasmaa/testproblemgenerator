from typing import Dict, Tuple
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from time import localtime, time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import divide, matlib # i guess we could implement repmat ourselves
from desdeo_problem.problem import *

# utilities
# TODO: figure out the structure

class attractorRegion:
    def __init__(self):
        self.locations = None
        self.objective_indices = None
        self.centre = None
        self.radius = None
        self.convhull = None
    
    def plot(self, ax, color = 'b'):
        """
        Very basic atm, Just plot the outer lines
        """
        if self.convhull is None: return
        p = self.locations
        
        for s in self.convhull.simplices:
            ax.plot(p[s,0], p[s,1], color = 'black')
        ax.fill(p[self.convhull.vertices,0], p[self.convhull.vertices, 1], color=color, alpha = 0.7)

class DBMOPPobject:
    def __init__(self):
        self.rescaleConstant = 0 # What the hell is up with these two attributes
        self.rescaleMultiplier = 1 # They are only used once and even there they do nothing...
        self.attractors = []
        self.attractor_regions = [] # array of attractorRegions 
        self.pi1 = None
        self.pi2 = None
        self.neutral_region_objective_values = np.sqrt(8)
        self.centre_radii = None
        self.pareto_set_indices = 0
        self.centre_list = None
        self.pareto_angles = None
        self.rotations = None

        self.neutral_region_centres = None
        self.neutral_region_radii = None

        self.hard_constraint_centres = None
        self.hard_constraint_radii = None

        self.soft_constraint_centres = None
        self.soft_constraint_radii = None

        self.discontinuous_region_centres = None
        self.discontinuous_region_objective_value_offset = None
        self.discontinuous_region_radii = None

        self.pivot_locations = None
        self.bracketing_locations_lower = None
        self.bracketing_locations_upper = None


class DBMOPP:
    """
    Lirum larum

    Args:
        k (int): Number of objectives
        n (int): Number of variables
        nlp (int): Number of local pareto sets
        ndr (int): Number of dominance resistance regions
        ngp: (int): Number of global Pareto sets
        prop_constraint_checker (float): Proportion of constrained 2D space if checker type is used
        pareto_set_type (int): A set type for global Pareto set. Should be one of these
            0: duplicate performance, 1: partially overlapping performance,
            or 2: non-intersecting performance
        constraint_type (int): A constraint type. Should be one of these
            0: No constraint, 1-4: Hard vertex, centre, moat, extended checker, 
            5-8: soft vertex, centre, moat, extended checker.
        ndo (int): Number of regions to apply whose cause discontinuities in objective functions. Defaults to 0
        vary_sol_density (bool): Should solution density vary in maping down to each of the two visualized dimensions.
            Default to False
        vary_objective_scales (bool): Are objective scale varied. Defaults to False
        prop_neutral (float): Proportion of neutral space. Defaults to 0
        nm (int): Number of samples used for approximation checker and neutral space coverage. Defaults to 10000

    Raises:
        Argument was invalid
    """
    def __init__(
        self,
        k: int,
        n: int,
        nlp: int,
        ndr: int,
        ngp: int,
        prop_constraint_checker: float,
        pareto_set_type: int,
        constraint_type: int,
        ndo: int = 0,
        vary_sol_density: bool = False,
        vary_objective_scales: bool = False,
        prop_neutral: float = 0,
        nm: int = 10000,
    ) -> None:
        msg = self._validate_args(k, n, nlp, ndr, ngp, prop_constraint_checker, pareto_set_type, constraint_type,
            ndo, prop_neutral, nm) # I guess one could also validate the types but ehh
        if msg != "":
            raise Exception(msg)
        self.k = k
        self.n = n
        self.nlp = nlp
        self.ndr = ndr
        self.ngp = ngp
        self.prop_contraint_checker = prop_constraint_checker
        self.pareto_set_type = pareto_set_type
        self.constraint_type = constraint_type
        self.ndo = ndo
        self.vary_sol_density = vary_sol_density
        self.vary_objective_scales = vary_objective_scales
        self.prop_neutral = prop_neutral
        self.nm = nm

        self.obj = DBMOPPobject() # The obj in the matlab implementation

        self.initialize()

    
    def _validate_args(
        self,
        k: int,
        n: int,
        nlp: int,
        ndr: int,
        ngp: int,
        prop_constraint_checker: float,
        pareto_set_type: str,
        constraint_type: str,
        ndo: int,
        prop_neutral: float,
        nm: int
    ) -> None:
        """
        Validate arguments given to the constructor of the class. 

        Args:
            See __init__
        
        Returns:
            str: A error message which contains everything wrong with the arguments. Empty string if arguments are valid
        """
        msg = ""
        if k < 1:
            msg += f"Number of objectives should be greater than zero, was {k}.\n"
        if n < 2:
            msg += f"Number of variables should be greater than two, was {n}.\n"
        if nlp < 0: 
            msg += f"Number of local Pareto sets should be greater than or equal to zero, was {nlp}.\n"
        if ndr < 0: 
            msg += f"Number of dominance resistance regions should be greater than or equal to zero, was {ndr}.\n"
        if ngp < 1:
            msg += f"Number of global Pareto sets should be greater than one, was {ngp}.\n"
        if not 0 <= prop_constraint_checker <= 1:
            msg += f"Proportion of constrained 2D space should be between zero and one, was {prop_constraint_checker}.\n"
        if pareto_set_type not in np.arange(3):
            msg += f"Global pareto set type should be a integer number between 0 and 2, was {pareto_set_type}.\n"
        if constraint_type not in np.arange(9):
            msg += f"Constraint type should be a integer number between 0 and 8, was {constraint_type}.\n"
        if ndo < 0:
            msg += f"Number of discontinuous objective function regions should be greater than or equal to zero, was {ndo}.\n"
        if not 0 <= prop_neutral <= 1:
            msg += f"Proportion of neutral space should be between zero and one, was {prop_neutral}.\n"
        if nm < 1000:
            msg += f"Number of samples should be at least 1000, was {nm}.\n"
        return msg
    
    def get_random_angles(self, n):
        return np.random.rand(n,1) * 2 * np.pi
    
    def is_pareto_set_member(self, z):
        self.check_valid_length(z)
        x = self.get_2D_version(z)
        return self.is_pareto_2D(x)
    
    # HIDDEN METHODS, not really but in MATLAB :D

    def evaluate(self, x):
        # x = np.atleast_2d(x)
        self.check_valid_length(x)
        z = self.get_2D_version(x)
        return self.evaluate_2D(z)

    def evaluate_2D(self, x) -> Dict:
        """
        Evaluate x in problem instance in 2 dimensions
        
        Args:
            x (np.ndarray): The decision vector to be evaluated
        
        Returns:
            Dict: A dictionary object with the following entries:
                'obj_vector' : np.ndarray, the objective vector
                'soft_constr_viol' : boolean, soft constraint violation
                'hard_constr_viol' : boolean, hard constraint violation
        """
        # x = np.atleast_2d(x)

        ans = {
            "obj_vector": np.array([None] * self.k),
            "soft_constr_viol": False,
            "hard_constr_viol": False,
        }
        if self.get_hard_constraint_violation(x):
            ans["hard_constr_viol"] = True
            if self.constraint_type == 3:
                if self.in_convex_hull_of_attractor_region(x):
                    ans["hard_constr_viol"] = False
                    ans["obj_vector"] = self.get_objectives(x)
            return ans
        
        if self.get_soft_constraint_violation(x):
            ans["soft_constr_viol"] = True
            if (self.constraint_type == 7):
                if (self.in_convex_hull_of_attractor_region(x)):
                    ans["soft_constr_viol"] = False
                    ans["obj_vector"] = self.get_objectives(x)
            return ans
        
        # Neither constraint breached

        if self.in_region(self.obj.neutral_region_centres, self.obj.neutral_region_radii, x)[0]:
            ans["obj_vector"] = self.obj.neutral_region_objective_values
        else: 
            ans["obj_vector"] = self.get_objectives(x)
        return ans

    def is_pareto_2D(self, x: np.ndarray):
        """
        
        """
        if self.get_hard_constraint_violation():
            return False
        if self.get_soft_constraint_violation():
            return False
        return self.is_in_limited_region()

    def in_convex_hull_of_attractor_region(self, y: np.ndarray):
        """
        
        """
        print("in_convex_hull_of_attractor_region")
        print("TODO indexi jumppa\n\n")
        self.check_valid_length()
        x = self.get_2D_version(y)

        dist = np.linalg.norm(self.obj.centre_list, x)
        if np.any(dist < self.obj.centre_radii):
            return self.in_hull(x, self.obj.attractor_regions) # TODO indeksijumppa
        return False


    def check_valid_length(self, x):
        # x = np.atleast_2d(x)
        if (x.shape[0] != self.n): 
            msg = f"Number of design variables in the argument does not match that required in the problem instance, was {x.shape[0]}, should be {self.n}"
            raise Exception(msg)

    def set_up_attractor_centres(self):
        """
        Calculate max maximum region radius given problem properties

        This aka setUpAttractorCentres
        """
        # number of local PO sets, global PO sets, dominance resistance regions
        n = self.nlp + self.ngp + self.ndr 

        max_radius = 1/(2*np.sqrt(n)+1) * (1 - (self.prop_neutral + self.prop_contraint_checker)) # prop 0 and 0.
        radius = self.place_regions(n, max_radius)

        self.obj.centre_radii = np.ones((n,1)) * radius # We need this because nlp might be <= 0

        if self.nlp > 0:
            # TODO: when locals taken into account. Does not work yet
            self.obj.centre_radii[self.nlp + 1 : -1] = radius / 2
            w = np.linspace(1, 0.5, self.nlp + 1)
            # linearly decrease local front radii
            self.obj.centre_radii[0:self.nlp] = self.obj.centre_radii[0:self.nlp] *  w[0:self.nlp]

        # save indices of PO set locations
        self._pareto_set_indices = self.nlp + self.ngp

    def place_regions(self, n: int, r: float):
        """

        Args:

        """
        effective_bound = 1 - r
        threshold = 4*r
        self.obj.centre_list = np.zeros((n,2))

        time_start = time()
        max_elapsed = 5 # Max seconds after reattempt. THIS IS VERY DUMP!
        rand_coord = (np.random.rand(1, 2)*2*effective_bound) - effective_bound
        self.obj.centre_list[0,:] = rand_coord  #random cordinate pair between -(1-radius) and +(1-radius)
        print('Radius: ', r)

        for i in np.arange(1, n):
            while True:
                rand_coord = (np.random.rand(1, 2)*2*effective_bound) - effective_bound
                t = np.min(np.linalg.norm(self.obj.centre_list[0:i,:] - rand_coord))
                print(t)
                if t > threshold:
                    print("assigned centre", i)
                    break
                too_long = time() - time_start > max_elapsed
                if (too_long): # Took longer than max_elapsed... Still very dump
                    print('restarting attractor region placement with smaller radius...\n')
                    return self.place_regions(n, r*0.95)
            self.obj.centre_list[i,:] = rand_coord
        return r

    def place_attractors(self):
        """
            Randomly place attractor regions in 2D space
        """
        print("place_attractors")
        l = self.nlp + self.ngp
        ini_locs = np.zeros((l, 2, self.k))

        self.obj.attractor_regions = np.array([None] * l)

        for i in np.arange(0, l):
            self.obj.attractor_regions[i] = attractorRegion()
            B = np.hstack((
                np.cos(self.obj.pareto_angles + self.obj.rotations[i]),
                np.sin(self.obj.pareto_angles + self.obj.rotations[i])
            ))

            locs = (
                matlib.repmat(self.obj.centre_list[i,:], self.k, 1) + 
                (matlib.repmat(self.obj.centre_radii[i], self.k, 2) * B)
            )

            self.obj.attractor_regions[i].locations = locs
            self.obj.attractor_regions[i].objective_indices = np.arange(self.k) 
            self.obj.attractor_regions[i].centre = self.obj.centre_list[i,:] # matlab comments these two as duplicate storage.. 
            self.obj.attractor_regions[i].radius = self.obj.centre_radii[i]  # we prob should get rid of these later
            # need to feed the points in right shape 
            self.obj.attractor_regions[i].convhull = self.convhull(locs)

            for k in np.arange(self.k):
                ini_locs[i,:,k] = locs[k,:]
            
        # matlabcode copies locations to the attractors for easier use for plotting
        self.obj.attractors = np.zeros((self.k, self.nlp + self.ngp, 2)) # Not sure about this
        for i in range(self.k):
            self.obj.attractors[i] = ini_locs[:,:,i]

        print("TODO assign dominacne resistance regions\n\n")
        # code assigns dominance resistance regions. ignore for now


    def initialize(self):
        #place attractor centres for regions defining attractor points
        self.set_up_attractor_centres()
        #set up angles for attractors on regin cicumferences and arbitrary rotations for regions
        self.obj.pareto_angles = self.get_random_angles(self.k) # arbitrary angles for Pareto set
        print(self.obj.centre_radii)
        self.obj.rotations = self.get_random_angles(self.obj.centre_radii.shape[0])
        # now place attractors
        self.place_attractors()
        if self.pareto_set_type != 0:
            self.place_disconnected_pareto_elements()
        self.place_discontinunities_neutral_and_checker_constraints()
        # set the neutral value to be the same in all neutral locations
        self.obj.neutral_region_objective_values = np.ones((1,self.k))*self.obj.neutral_region_objective_values; # CHECK
        self.place_vertex_constraint_locations()
        self.place_centre_constraint_locations()
        self.place_moat_constraint_locations()
        self.assign_design_dimension_projection()

    def place_disconnected_pareto_elements(self):
        n = self.ngp - 1
        pivot_index = np.random.randint(self.k)

        # sort from smallest to largest and get the indices
        indices = np.argsort(self.obj.pareto_angles, axis = 0)

        offset_angle_1 = (self.obj.pareto_angles[indices[self.k - 1]] if pivot_index == 0
            else self.obj.pareto_angles[indices[pivot_index-1]]) # check this minus
        
        offset_angle_2 = (self.obj.pareto_angles[indices[0]] if pivot_index == self.k-1
            else self.obj.pareto_angles[indices[pivot_index + 1]]) # check plus
        
        pivot_angle = self.obj.pareto_angles[indices[pivot_index]]

        if pivot_angle == (offset_angle_1 or offset_angle_2):
            raise Exception("Angle should not be duplicated!")
        
        if offset_angle_1 < offset_angle_2:
            range_covered = offset_angle_1 + 2 * np.pi - offset_angle_2
            p1 = offset_angle_1 / range_covered
            r = np.random.rand(n)
            #r = temp ## whats the point of temp
            p1 = np.sum(r < p1)
            r[:p1] = 2*np.pi + np.random.rand(p1) * offset_angle_1
            r[p1:n] = np.random.rand(n-p1) * (2*np.pi - offset_angle_2) + offset_angle_2
            r = np.sort(r)
            r_angles = np.zeros(n+2)
            r_angles[0] = offset_angle_2
            r_angles[n+1] = offset_angle_1
            r_angles[1:n+1] = r
        else:
            r = r = np.random.rand(n)
            r = np.sort(r)
            r_angles = np.zeros(n+2)
            r_angles[0] = offset_angle_2 # doing almost the same thing above
            r_angles[n+1] = offset_angle_1
            r_angles[1:n+1] = r 

        k = self.nlp + self.ngp
        self.obj.pivot_locations = np.zeros((k, 2)) 
        self.obj.bracketing_locations_lower = np.zeros((k,2))
        self.obj.bracketing_locations_upper = np.zeros((k,2))

        def calc_location(ind, a):
            radiis = matlib.repmat(self.obj.centre_radii[i], 1, 2)
            return (
                self.obj.centre_list[ind,:] + radiis
                * np.hstack((
                    np.cos(a + self.obj.rotations[i]),
                    np.sin(a + self.obj.rotations[i])
                ))
            )

        index = 0
        for i in range(self.nlp, self.nlp + self.ngp): # verify indexing
            self.obj.pivot_locations[i,:] = calc_location(i, pivot_angle)
            
            self.obj.bracketing_locations_lower[i,:] = calc_location(i, r_angles[index])

            if self.pareto_set_type == 0:
                raise Exception('should not be calling this method with an instance with identical Pareto set regions')
            
            elif self.pareto_set_type == 2:
                self.obj.bracketing_locations_upper[i,:] = calc_location(i, r_angles[index+1])

            elif self.pareto_set_type == 1:
                if index == self.ngp-1:
                    self.obj.bracketing_locations_lower[i,:] = calc_location(i, r_angles[2])
                    self.obj.bracketing_locations_upper[i,:] = calc_location(i, r_angles[n])
                else:
                    self.obj.bracketing_locations_upper[i,:] = calc_location(i, r_angles[index+2])
            index += 1
                    

    def place_vertex_constraint_locations(self):
        """
        Place constraints located at attractor points
        """
        pass

    def place_centre_constraint_locations(self):
        """
        Place center constraint regions
        """
        print("Assigning any centre soft/hard constraint regions.\n")
        if self.constraint_type == 2:
            self.obj.hard_constraint_centres = self.obj.centre_list
            self.obj.hard_constraint_radii = self.obj.centre_radii
        elif self.constraint_type == 5:
            self.obj.soft_constraint_centres = self.obj.centre_list
            self.obj.soft_constraint_radii = self.obj.centre_radii

    def place_moat_constraint_locations(self):
        """
        Place moat constraint regions
        """
        print('Assigning any moat soft/hard constraint regions\n')
        r = np.random.rand() + 1
        if self.constraint_type == 3:
            self.obj.hard_constraint_centres = self.obj.centre_list
            self.obj.hard_constraint_radii = self.obj.centre_radii * r
        elif self.constraint_type == 6:
            self.obj.soft_constraint_centres = self.obj.centre_list
            self.obj.soft_constraint_radii = self.obj.centre_radii * r

         

    def place_discontinunities_neutral_and_checker_constraints(self):
        pass

    def setNotAttractorRegionsAsProportionOfSpace(self, S, proportion_to_attain, other_center, other_radii):
        pass

    def get_hard_constraint_violation(self, x):
        print("get_hard_constraint_violation NOT DONE")
        return False

    def get_soft_constraint_violation(self, x):
        print("get_soft_constraint_violation NOT DONE")
        return False

    def assign_design_dimension_projection(self):
        """
        if more than two design dimensions in problem, need to assign
        the mapping down from this higher space to the 2D version
        which will be subsequantly evaluated
        """
        if self.n > 2:
            mask = np.random.permutation(self.n)
            if self.vary_sol_density:
                diff = np.random.randint(0, self.n)
                mask = mask[:diff] # Take the diff first elements
            else: 
                half = int(np.ceil(self.n/2))
                mask = mask[:half] # Take half first elements
            self.obj.pi1 = np.array([False]*self.n)
            self.obj.pi1[mask] = True
            self.obj.pi2 = np.logical_not(self.obj.pi1)

    def get_2D_version(self, x):
        """
        Project n > 2 dimensional vector to 2-dimensional space

        Args:
            x (np.ndarray): A given vector to project to 2-dimensional space
        
        Returns:
            np.ndarray: A 2-dimensional vector
        """
        if (x.shape[0] <= 2):
            print("Skipping projection, vector already 2 dimensional or less")
            return x
        l = np.divide(np.dot(x, self.obj.pi1), np.sum(self.obj.pi1)) # Left side of vector
        r = np.divide(np.dot(x, self.obj.pi2), np.sum(self.obj.pi2)) # Right side of vector
        return np.hstack((l, r))

    def get_minimun_distance_to_attractors(self, x: np.ndarray):
        """
        
        """
        y = np.zeros(self.k)
        for i in range(self.k):
            d = np.linalg.norm(self.obj.attractors[i] - x)
            y[i] = np.min(d)
        y *= self.obj.rescaleMultiplier
        y += self.obj.rescaleConstant
        return y
    
    def get_objectives(self, x):
        print("Get objectives")
        if (self.pareto_set_type == 0):
            y = self.get_minimun_distance_to_attractors(x)
        else:
            y = self.get_minimum_distances_to_attractors_overlap_or_discontinuous_form(x)
        
        y = self.update_with_discontinuity(x,y)
        y = self.update_with_neutrality(x,y)
        return y

    def get_minimum_distances_to_attractors_overlap_or_discontinuous_form(self, x):
        print("get_minimum_distances_to_attractors_overlap_or_discontinuous_form")
        y = self.get_minimun_distance_to_attractors(x)
        in_pareto_region, in_hull, index  = self.is_in_limited_region(x).values()
        if in_hull:
            if not in_pareto_region:
                y += self.obj.centre_radii[index]
        return y

    def is_in_limited_region(self, x, eps = 1e-06):
        """
        
        """
        print("is_in_limited_region")
        print("TODO: between_lines_rooted_at_pivot, verify that does the same thing\n\n")
        ans = {
            "in_pareto_region": False,
            "in_hull": False,
            "index": -1
        }
        dist = np.linalg.norm(self.obj.centre_list - x)
        I = np.where(dist <= (self.obj.centre_radii + eps))
        I = np.concatenate(I)
        if I.size > 0: # is not empty 
            if self.nlp < I[0] <= self.nlp + self.ngp:
                if self.constraint_type in [2,6]: 
                    # Smaller of dist
                    r = np.min(np.abs(dist[I[0]]), np.abs(self.obj.centre_radii[I[0]]))
                    # THIS if + elif could be a oneliner ans["inhull"] = np.abs .. or in_hull
                    if np.abs(dist[I[0]]) - self.obj.centre_radii(I(0)) < 1e4 * eps * r:
                        ans["in_hull"] = True
                    elif self.in_hull(x, self.obj.attractor_regions):
                        ans["in_hull"] = True 
        
        if self.pareto_set_type == 0 or self.constraint_type in [2,6]:
            ans["in_pareto_region"] = ans["in_hull"]
            ans["in_hull"] = False
        else:
            if ans["in_hull"]:
                ans["index"] = I[0]
                ans["in_pareto_region"] = self.between_lines_rooted_at_pivot(x,x,x) # TODO
                if self.pareto_set_type == 1:
                    if I[0] == self.nlp + self.ngp:
                        ans["in_pareto_region"] = not ans["in_pareto_region"]
        return ans



    def update_with_discontinuity(self, x, y):
        if self.obj.discontinuous_region_centres is None: return y # or len = 0 ?
        dist = np.linalg.norm(self.obj.discontinuous_region_centres - x)
        dist[dist >= self.obj.discontinuous_region_radii] = 0 
        if np.sum(dist) > 0: # Could check more efficiently
            i = np.argmin(dist) # Umm should it be min of a value which is greater than 0
            y = y + self.obj.discontinuous_region_objective_value_offset[i,:]
        return y


    def update_with_neutrality(self, x, y):
        if self.obj.neutral_region_centres is None: return y # or len = 0 ?
        dist = np.linalg.norm(self.obj.neutral_region_centres - x)
        dist[dist >= self.obj.neutral_region_radii] = 0 
        if np.sum(dist) > 0: # Could check more efficiently
            i = np.argmin(dist) # Umm should it be min of a value which is greater than 0
            y = y + self.obj.neutral_region_objective_values[i,:]
        return y


    def set_objective_rescaling_variables(self):
        """
        Set offset and multiplier for objectives
        """
        pass

    
    # DBMOPP methods

    def in_region(self, centres, radii, x) -> Tuple[bool, np.ndarray]:
        if centres is None or len(centres) < 1: return (False, np.array([]))
        dist = np.linalg.norm(centres - x)
        in_region = np.any(dist <= radii)
        return (in_region, dist)

    def between_lines_rooted_at_pivot(self, x, pivot_loc, loc1, loc2) -> bool:
        """
        Plaaplaa
        """
        d1 = ( x(1) - pivot_loc(1))*(pivot_loc(2) - pivot_loc(2))
        - (x(2) - pivot_loc(2))*(loc1(1) - pivot_loc(1))

        d2 = ( x(1) - pivot_loc(1))*(pivot_loc(2) - pivot_loc(2))
        - (x(2) - pivot_loc(2))*(loc2(1) - pivot_loc(1))

        return d1 == 0 or d2 == 0 or np.sign(d1) != np.sign(d2)
    
    # PLOTTING

    def plot_problem_instance(self):
        """
        """
        fig, ax = plt.subplots()
        # Plot local Pareto regions

        plt.xlim([-1, 1])
        plt.ylim([-1,1])
        for i in range(self.nlp):
            self.obj.attractor_regions[i].plot(ax, 'g') # Green
        
        # global pareto regions

        for i in range(self.nlp, self.nlp + self.ngp):
            self.obj.attractor_regions[i].plot(ax, 'r')
            print("the fill here is different than above")
        
        # dominance resistance set regions
        for i in range(self.nlp + self.ngp, self.nlp + self.ngp + self.ndr):
            # attractor regions should take care of different cases
            self.obj.attractor_regions[i].plot(ax, 'b') 
        
        # Plot contraint region rectangles
        def plot_constraint_regions(centres, radii, color):
            if radii is None: return
            for i in range(len(radii)):
                x = centres[i,0] - radii[i]
                y = centres[i,1] - radii[i]
                r = radii[i] * 2
                self.plot_rectangle(ax, x, y, r, r, color)
            
        plot_constraint_regions(self.obj.hard_constraint_radii, self.obj.hard_constraint_centres, 'yellow')
        plot_constraint_regions(self.obj.soft_constraint_radii, self.obj.soft_constraint_centres, 'orange')
        plot_constraint_regions(self.obj.neutral_region_radii, self.obj.neutral_region_centres, 'grey')

        # PLOT DISCONNECTED PENALTY
        print("disconnected Pareto penalty regions not yet plotted. THIS IS NOT IMPLEMENTED IN MATLAB")

        # plot attractor points
        # This could propably be done better in just the attractor region place...
        # ugh double loop
        for i in range(self.k):
            for j in range(self.ngp):
                locs = self.obj.attractors[i]
                ax.scatter(locs[j,0], locs[j,1], color = 'b')
                ax.annotate(i, (locs[j,0], locs[j,1]))

        plt.show()
        


    # Methods matlab has built in

    def repmat(t, x, y): # could do this...
        pass 

    def plot_rectangle(self, ax, x, y, rx, ry, color):
        rectangle = Rectangle((x,y), rx, ry, fc = 'none', color=color, linewidth = 5)
        ax.add_patch(rectangle)

    def convhull(self,points):
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

    
    def in_hull(self, x: np.ndarray, points: np.ndarray):
        """
        Is a point inside a convex hull 

        Args:
            x (np.ndarray): The point that is checked
            points (np.ndarray): The point cloud of the convex hull
        
        Returns:
            bool: is x inside the convex hull given by points 
        """
        n_points = len(points)
        c = np.zeros(n_points)
        A = np.r_[points.T,np.ones((1,n_points))]
        b = np.r_[x, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        return lp.success


    # THESE WERE NOT IN THE MATLAB CODE BUT IN THE ARTICLE
    
    def assign_attractor_region_rotations(self):
        """
        Set up rotations to be used by each attractor region

        this obj.ParetoAngles and obj.rotations attributes,
        """
        pass


    def uniform_sample_from_2D_domain(self):
        """
        Generate 'nm' number of uniform samples in 2D

        Returns:
            np.ndarray: uniform 2D samples
        """
        pass

    def remove_samples_in_attractor_regions(self, M: np.ndarray):
        """
        Remove any samples falling in attractor regions

        Args:
            M (np.ndarray): Uniform 2D samples
        
        Returns:
            np.ndarray: M without samples in attractor regions
        """
        pass

    def place_checker_constraint_locations(self, M: np.ndarray):
        """
        Place checker constraint regions

        Args: 
            M (np.ndarray): Uniform 2D samples
        
        Returns:
            np.ndarray: idk
        """
        pass

    def place_neutral_regions(self, M: np.ndarray):
        """
        Place neural regions
        """
        pass

    def place_discontinuous_regions(self):
        """
        Place regions whose boundaries cause discontinuities in objectives
        """
        pass
    
    # Pseudo code in the article
    def generate_problem(self):
        """
        Generate the test problem

        Returns:
            MOProblem: A test problem
        """
        objectives = [ScalarObjective("objective", lambda x: self.evaluate(x)['obj_vector'])]

        var_names = [f'x{i}' for i in range(self.n)]
        initial_values = (np.random.rand(self.n) * 2) - 1
        lower_bounds = np.ones(self.n) * -1
        upper_bounds = np.ones(self.n)
        variables = variable_builder(var_names, np.random.rand(self.n), lower_bounds, upper_bounds)

        constraints = None # DUNNO // Maybe from evaluate the constraints
        return MOProblem(objectives, variables, constraints)


if __name__=="__main__":
    n_objectives = 5
    n_variables = 4
    n_local_pareto_regions = 0
    n_disconnected_regions = 0
    n_global_pareto_regions = 6
    pareto_set_type = 2
    # global PO set style 0.
    # if k < ngp this fails. FIX
    problem = DBMOPP(
        n_objectives,
        n_variables,
        n_local_pareto_regions,
        n_disconnected_regions,
        n_global_pareto_regions,
        0,
        pareto_set_type,
        0, 0,False, False, 0, 10000
    )
    print("Initializing works!")
    x = np.random.rand(n_variables)
    print(problem.evaluate(x))
    moproblem = problem.generate_problem()

    # wont work because x will be converted to 2d array and then some of the indexing fail. 
    # Fix: Either check for 2d or modify the existing code to use the same kind of arrays as desdeo
    # print(moproblem.evaluate(x)) 

    problem.plot_problem_instance()