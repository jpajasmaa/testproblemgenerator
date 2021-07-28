import numpy as np
#from desdeo_problem.problem import MOProblem

# utilities
# TODO: figure out the structure

# should probably be inside the dbmopp class..
def get2DVersion(dbmopp, x) -> float:
    pass


def getMinimumDistancesToAttractors(dbmopp, x) -> list:
    pass

def getObjectives(dbmopp, x) -> list:
    pass





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

        # some more attributes
        self._centre_radii = []
        self._pareto_set_indices = 0
        self._centre_list = []
    
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
        if self.vary_sol_density:
            self.set_projection_vectors()
        if self.vary_objective_scales:
            self.set_objective_rescaling_variables()
        
        return self #MOProblem() # hmmm 

    def set_attractor_regions(self):
        """
        Calculate max maximum region radius given problem properties

        This aka setUpAttractorCentres
        """
        # number of local PO sets, global PO sets, dominance resistance regions
        n = self.nlp + self.ngp + self.ndr 
        print(n)

        max_radius = 1/(2*np.sqrt(n)+1) * (1 - (self.prop_neutral + self.prop_contraint_checker)) # prop 0 and 0.
        radius = self.placeRegions(n, max_radius)

        if self.nlp > 0:
            # TODO: when locals taken into account. Does not work yet
            self._centre_radii[self.nlp : -1] = radius / 2
            w = np.linspace(1, 0.5, self.nlp)
            # linearly decrease local front radii
            #self._centre_radii[0:self.nlp] = self._centre_radii[0:self.nlp] * w[0:self.nlp]

        # save indices of PO set locations
        self.pareto_set_indices = self.nlp + self.ngp

        

    
    def placeRegions(self, n, r):
        """
        ignoring the time thingy
        """
        effective_bound = 1 - r
        threshold = 4*r
        self._centre_list = np.zeros((n,2)) # maybe correct conversion
        # ignore the time tic whatever

        self._centre_list[0,:] = (np.random.rand(1,2)*2*effective_bound) - effective_bound  #random cordinate pair between -(1-radius) and +(1-radius)
        print('Radius: ', r)

        for i in np.arange(1, n):
            invalid = True
            while invalid:
                rand_coord = (np.random.rand(1, 2)*2*effective_bound) - effective_bound
                t = np.min(np.linalg.norm(self._centre_list[0:i,:] - rand_coord))
                print(t)
                if t > threshold:
                    print("assigned centre", i)
                    invalid = False

            self._centre_list[i,:] = rand_coord


    def assign_attractor_region_rotations(self):
        """
        Set up rotations to be used by each attractor region

        this obj.ParetoAngles and obj.rotations attributes,
        """
        pass

    def place_attractor_points(self):
        """
        Randomly place attractor regions in 2D space

        This placeAttractors.
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

    def place_vertex_constraint_locations(self):
        """
        Place constraints located at attractor points
        """
        pass

    def place_center_constraint_locations(self):
        """
        Place center constraint regions
        """
        pass

    def place_moat_constraint_locations(self):
        """
        Place moat constraint regions
        """
        pass

    def set_projection_vectors(self):
        """
        Lirum lorum lapsum
        """
        pass

    def set_objective_rescaling_variables(self):
        """
        Set offset and multiplier for objectives
        """
        pass


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
        


if __name__=="__main__":
    # global PO set style 0.
    my_instance = DBMOPP(4, 2, 0, 0, 5, 0,1,0,0,False, False, 0)
    my_instance.generate_problem()
    print(my_instance._centre_list)
    print("runs")

