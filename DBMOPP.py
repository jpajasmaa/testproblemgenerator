from typing import Dict, Tuple
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
#from desdeo_problem.problem import MOProblem

# utilities
# TODO: figure out the structure



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
        pareto_set_type: str,
        constraint_type: str,
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

        # Attributes. This is ugly :D Could change it to own class with just attributes. thoughts?
        self.rescaleConstant = 0 # What the hell is up with these two attributes
        self.rescaleMultiplier = 1 # They are only used once and even there they do nothing...
        self.attractors = None # aka attractorsList what is this?
        self.attractor_regions = None # What is this
        self._pi1 = None
        self._pi2 = None
        self.neutral_region_objective_values = np.sqrt(8)
        self._centre_radii = None
        self._pareto_set_indices = 0
        self._centre_list = None

        self.neutral_region_centres = []
        self.neutral_region_radii = None
    
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
            msg += f"Number of samples should be greater than 1000, was {nm}.\n"
        return msg
    
    def get_random_angles(self, n):
        return np.random.rand(n,1) * 2 * np.pi
    
    # HIDDEN METHODS, not really but in MATLAB :D

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
        ans = {
            "obj_vector": np.array([None] * self.k),
            "soft_constr_viol": False,
            "hard_constr_viol": False,
        }
        print("evaluate_2D:")
        print("This should propably be done with moproblem.evaluate at some point")
        print("or maybe this can be passed to moproblem...")
        print("TODO get hard and soft constraint violation, get objectives")
        print("neutral things not defined\n")

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

        if self.in_neutral_region(self.neutral_region_centres, self.neutral_region_radii, x):
            ans["obj_vector"] = self.neutral_region_objective_values
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
        print("What is attractor regions, check indices and stuff\n")

        self.check_valid_length()
        x = self.get_2D_version(y)

        dist = np.linalg.norm(self._centre_list, x)
        if np.any(dist < self._centre_radii):
            return self.in_hull(x, self.attractor_regions) # TODO
        return False


    def check_valid_length(self, x):
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
        print(n)

        max_radius = 1/(2*np.sqrt(n)+1) * (1 - (self.prop_neutral + self.prop_contraint_checker)) # prop 0 and 0.
        radius = self.place_regions(n, max_radius)

        self._centre_radii = np.ones((n,1)) * radius # We need this because nlp might be <= 0

        if self.nlp > 0:
            # TODO: when locals taken into account. Does not work yet
            self._centre_radii[self.nlp + 1 : -1] = radius / 2
            w = np.linspace(1, 0.5, self.nlp + 1)
            # linearly decrease local front radii
            #self._centre_radii[0:self.nlp] = self._centre_radii[0:self.nlp] * w[0:self.nlp]

        # save indices of PO set locations
        self._pareto_set_indices = self.nlp + self.ngp

    def place_regions(self, n: int, r: float):
        """
        ignoring the time thingy

        Args:

        """
        effective_bound = 1 - r
        threshold = 4*r
        self._centre_list = np.zeros((n,2))

        self._centre_list[0,:] = (np.random.rand(1,2)*2*effective_bound) - effective_bound  #random cordinate pair between -(1-radius) and +(1-radius)
        print('Radius: ', r)

        for i in np.arange(1, n):
            while True:
                rand_coord = (np.random.rand(1, 2)*2*effective_bound) - effective_bound
                t = np.min(np.linalg.norm(self._centre_list[0:i,:] - rand_coord))
                print(t)
                if t > threshold:
                    print("assigned centre", i)
                    break

        self._centre_list[i,:] = rand_coord
        return r

    def place_attractors(self):
            """
            Randomly place attractor regions in 2D space
            """
            pass

    def initialize(self):
        #place attractor centres for regions defining attractor points
        self.set_up_attractor_centres()
        #set up angles for attractors on regin cicumferences and arbitrary rotations for regions
        self._pareto_angles = self.get_random_angles(self.n) # arbitrary angles for Pareto set
        print(self._centre_radii)
        self._rotations = self.get_random_angles(self._centre_radii.shape[0])
        # now place attractors
        self.place_attractors()
        if self.pareto_set_type != 0:
            self.place_disconnected_pareto_elements()
        self.place_discontinunities_neutral_and_checker_constraints()
        # set the neutral value to be the same in all neutral locations
        self.neutral_region_objective_values = np.ones((1,self.k))*self.neutral_region_objective_values; # CHECK
        self.place_vertex_constraint_locations()
        self.place_centre_constraint_locations()
        self.place_moat_constraint_locations()
        self.assign_design_dimension_projection()

    def place_disconnected_pareto_elements(self):
        pass

    def place_vertex_constraint_locations(self):
        """
        Place constraints located at attractor points
        """
        pass

    def place_centre_constraint_locations(self):
        """
        Place center constraint regions
        """
        pass

    def place_moat_constraint_locations(self):
        """
        Place moat constraint regions
        """
        pass 

    def place_discontinunities_neutral_and_checker_constraints(self):
        pass

    def setNotAttractorRegionsAsProportionOfSpace(self, S, proportion_to_attain, other_center, other_radii):
        pass

    def get_hard_constraint_violation(self, x):
        pass

    def get_soft_constraint_violation(self, x):
        pass

    def assign_design_dimension_projection(self):
        """
        if more than two design dimensions in problem, need to assign
        the mapping down from this higher space to the 2D version
        which will be subsequantly evaluated
        """
        if self.n > 2:
            mask = np.random.permutation(self.n) + 1 # + 1 because we want to start from 1 and include n
            if self.vary_sol_density:
                diff = np.random.randint(1, self.n)
                mask = mask[:diff] # Take the diff first elements
            else: 
                half = int(np.ceil(self.n/2))
                mask = mask[:half] # Take half first elements
        
            self._pi1 = np.array([False]*self.n)
            self._pi1[:mask] = True
            self._pi2 = np.logical_not(self._pi1)

    def get_2D_version(self, x):
        """
        Project n > 2 dimensional vector to 2-dimensional space

        Args:
            x (np.ndarray): A given vector to project to 2-dimensional space
        
        Returns:
            np.ndarray: A 2-dimensional vector
        """
        if (x.shape[0] < 2):
            print("Skipping projection, vector already 2 dimensional or less")
            return x
        l = np.divide(np.dot(x, self._pi1)/np.sum(self._pi1)) # Left side of vector
        r = np.divide(np.dot(x, self._pi2)/np.sum(self._pi2)) # Right side of vector
        return np.hstack((l, r))

    def get_minimun_distance_to_attractors(self, x: np.ndarray):
        """
        
        """
        print("Missing self.attractors")
        y = np.zeros(self.n)
        for i in range(self.n):
            d = np.linalg.norm(self.attractors[i] - x)
            y[i] = np.min(d)
        
        y *= self.rescaleMultiplier
        y += self.rescaleConstant
        return y
    
    def get_objectives(self, x):
        pass

    def get_minimum_distances_to_attractors_overlap_or_discontinuous_form(self):
        pass

    def is_in_limited_region(self, x, eps = 1e-06):
        """
        
        """
        print("is_in_limited_region")
        print("TODO: between_lines_rooted_at_pivot, verify that does the same thing\n")
        ans = {
            "in_pareto_region": False,
            "in_hull": False,
            "index": -1
        }
        dist = np.linalg.norm(self._centre_list - x)
        I = np.where(dist <= self._centre_radii + eps)
        if len(I) > 0: # is not empty 
            if self.nlp < I[0] <= self.nlp + self.ngp:
                if self.constraint_type in [2,6]: 
                    # Smaller of dist
                    r = np.min(np.abs(dist[I[0]]), np.abs(self._centre_radii[I[0]]))
                    # THIS if + elif could be a oneliner ans["inhull"] = np.abs .. or in_hull
                    if np.abs(dist[I[0]]) - self._centre_radii(I(0)) < 1e4 * eps * r:
                        ans["in_hull"] = True
                    elif self.in_hull(x, self.attractor_regions):
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
        pass

    def update_with_neutrality(self, x, y):
        pass

    def set_objective_rescaling_variables(self):
        """
        Set offset and multiplier for objectives
        """
        pass

    
    # DBMOPP methods

    def in_neutral_region(self, centres, radii, x) -> Tuple[bool, np.ndarray]:
        if len(centres) < 1: return (False, np.array([]))
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


    # Methods matlab has built in

    def convhull(self, points):
        """
        Construct a convex hull of given set of points

        Args:
            points (np.ndarray): the points used to construct the convex hull
        
        Returns:
            np.ndarray: The indices of the simplices that form the convex hull
        """
        hull = ConvexHull(points)
        return hull.simplices
    
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
        n_dim = len(x)
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
    
    def generate_problem(self):
        """
        Generate the test problem

        Returns:
            MOProblem: A test problem
        """
        self.set_attractor_regions()
        self.assign_attractor_region_rotations()
        self.place_attractor_points()
        M = self.uniform_sample_from_2D_domain()
        M = self.remove_samples_in_attractor_regions(M)
        if self.constraint_type in [4, 8]: # Either soft or hard extended checker
            M = self.place_checker_constraint_locations(M)
        if self.prop_neutral > 0: 
            M = self.place_neutral_regions(M)
        if self.ndo > 0:
            self.place_discontinuous_regions(M)
        if self.constraint_type in [1, 5]: # Either soft or hard vertex
            self.place_vertex_constraint_locations()
        elif self.constraint_type in [2,6]: # Either soft or hard center
            self.place_center_constraint_locations()
        elif self.constraint_type in [3,7]: # Either soft or hard moat
            self.place_moat_constraint_locations()
        # if self.vary_sol_density:
        #     self.set_projection_vectors()
        if self.vary_objective_scales:
            self.set_objective_rescaling_variables()
        
        return self #MOProblem() # hmmm 



if __name__=="__main__":
    # global PO set style 0.
    x = np.array([1,2])
    my_instance = DBMOPP(4, 2, 0, 0, 5, 0,1,0,0,True, False, 0)
    my_instance.initialize()
    print(my_instance._centre_list)
    print(my_instance.evaluate_2D(x))
    print("runs")
