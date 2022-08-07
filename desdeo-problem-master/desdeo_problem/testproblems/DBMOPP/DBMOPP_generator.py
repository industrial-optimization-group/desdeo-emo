import functools
from time import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from desdeo_problem.problem import (
    MOProblem,
    ScalarConstraint,
    VectorObjective,
    variable_builder,
)
from desdeo_problem.testproblems.DBMOPP.Region import Attractor, AttractorRegion, Region
from desdeo_problem.testproblems.DBMOPP.utilities import (
    assign_design_dimension_projection,
    between_lines_rooted_at_pivot,
    euclidean_distance,
    get_2D_version,
    get_random_angles,
)
from matplotlib import cm
from numpy import matlib
from shapely.geometry import MultiPoint, Point, Polygon


class DBMOPP:
    """
    Object that holds the problem state and information
    """

    def __init__(self):
        self.rescaleConstant = 0
        self.rescaleMultiplier = 1
        self.pi1 = None
        self.pi2 = None
        self.pareto_set_indices = None
        self.pareto_angles = None
        self.rotations = None

        self.attractors = []  # Array of attractors
        self.attractor_regions = []  # array of attractorRegions
        self.centre_regions = None
        self.neutral_regions = None
        self.neutral_region_objective_values = np.sqrt(8)

        self.hard_constraint_regions = None
        self.soft_constraint_regions = None
        self.discontinuous_regions = None
        self.discontinuous_region_objective_value_offset = None

        self.pivot_locations = None
        self.bracketing_locations_lower = None
        self.bracketing_locations_upper = None
        self.disconnected_regions = []


class DBMOPP_generator:
    """
        DBMOPP-class has all the necessary functions and methods to create different problems.

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
        msg = self._validate_args(
            k,
            n,
            nlp,
            ndr,
            ngp,
            prop_constraint_checker,
            pareto_set_type,
            constraint_type,
            ndo,
            prop_neutral,
            nm,
        )
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

        self.obj = DBMOPP()

        self.initialize()

    def _print_params(self):
        print("n_obj: ", self.k)
        print("n_var: ", self.n)
        print("n_nlp: ", self.nlp)
        print("n_ndr: ", self.ndr)
        print("n_ngp: ", self.ngp)
        print("potype: ", self.pareto_set_type)
        print("const type: ", self.constraint_type)
        return

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
        nm: int,
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
            msg += f"Proportion of constrained 2D space should be between zero and one, \
            was {prop_constraint_checker}.\n"
        if pareto_set_type not in np.arange(3):
            msg += f"Global pareto set type should be a integer number between 0 and 2, was {pareto_set_type}.\n"
        if pareto_set_type == 1 and ngp <= 1:
            msg += f"Number of global pareto sets needs to be more than one, \
            if using disconnected pareto set type {pareto_set_type, ngp}"
        if ngp > 1 and k < 3:
            msg += f"Number of objectives needs to be more than three, \
            if number of global pareto sets is more than one, {k, ngp}"
        if constraint_type not in np.arange(9):
            msg += f"Constraint type should be a integer number between 0 and 8, was {constraint_type}.\n"
        if prop_constraint_checker == 0.0 and constraint_type in [4, 8]:
            msg += f"Proporortion of constrained space checker should not be 0 if constraint type is 4 or 8, \
                was constraint type {constraint_type} and prop {prop_constraint_checker}"
        if constraint_type not in [4, 8] and prop_constraint_checker != 0.0:
            msg += f"Proporortion of constrained space checker should be 0 if constraint type is not 4 or 8, \
            was constraint type {constraint_type} and prop {prop_constraint_checker}"
        if ndo < 0:
            msg += f"Number of discontinuous objective function regions should be greater than or equal to zero,\
            was {ndo}.\n"
        if not 0 <= prop_neutral <= 1:
            msg += f"Proportion of neutral space should be between zero and one, was {prop_neutral}.\n"
        if nm < 1000:
            msg += f"Number of samples should be at least 1000, was {nm}.\n"
        return msg

    def initialize(self):
        # place attractor centres for regions defining attractor points
        self.set_up_attractor_centres()
        # set up angles for attractors on regin cicumferences and arbitrary rotations for regions
        self.obj.pareto_angles = get_random_angles(self.k)  # arbitrary angles for Pareto set
        self.obj.rotations = get_random_angles(len(self.obj.centre_regions))
        # now place attractors
        self.place_attractors()

        if self.pareto_set_type != 0:
            self.place_disconnected_pareto_elements()
        self.place_discontinunities_neutral_and_checker_constraints()

        # set the neutral value to be the same in all neutral locations
        self.obj.neutral_region_objective_values = np.ones((1, self.k)) * self.obj.neutral_region_objective_values
        # CHECK
        self.place_vertex_constraint_locations()
        self.place_centre_constraint_locations()
        self.place_moat_constraint_locations()
        self.obj.pi1, self.obj.pi2 = assign_design_dimension_projection(self.n, self.vary_sol_density)

    def generate_problem(self) -> MOProblem:
        """
        Generate the test problem to use in DESDEO.

        Returns:
            MOProblem: A test problem
        """
        print("Generating MOProblem")

        obj_names = ["f" + str(i + 1) for i in range(self.k)]
        objectives = [VectorObjective(name=obj_names, evaluator=self.evaluate_objectives)]

        var_names = [f"x{i}" for i in range(self.n)]
        initial_values = (np.random.rand(self.n, 1) * 2) - 1
        lower_bounds = np.ones(self.n) * -1
        upper_bounds = np.ones(self.n)
        variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

        constraints = []

        def eval_wrapper(x, obj, region):
            res = np.zeros_like([])
            if x.shape[0] <= 1:
                return self.evaluate_constraint(x, region)
            for i in range(x.shape[0]):
                res = np.concatenate((res, self.evaluate_constraint(x[i], region)))
            return res

        if self.constraint_type in [1, 2, 3, 4]:
            for i, region in enumerate(self.obj.hard_constraint_regions):
                const_evaluator = functools.partial(eval_wrapper,  region=region)
                constraints.append(
                    ScalarConstraint(
                        f"hard constraint {i}",
                        self.n,
                        self.k,
                        evaluator=const_evaluator,
                    )
                )
        elif self.constraint_type in [5, 6, 7, 8]:
            for i, region in enumerate(self.obj.soft_constraint_regions):
                const_evaluator = functools.partial(eval_wrapper, region=region)
                constraints.append(
                    ScalarConstraint(
                        f"soft constraint {i}",
                        self.n,
                        self.k,
                        evaluator=const_evaluator,
                    )
                )
        else:
            const_evaluator = None

        if const_evaluator is None:
            return MOProblem(objectives, variables)
        return MOProblem(objectives, variables, constraints)

    def is_pareto_set_member(self, z):
        self.check_valid_length(z)
        x = get_2D_version(z, self.obj.pi1, self.obj.pi2)
        return self.is_pareto_2D(x)

    # TODO: make properly with the possible constraints etc. Now just to demo VectorObjective.
    def evaluate_objectives(self, x):
        x = np.atleast_2d(x)
        self.check_valid_length(x)
        ret = []
        for i in range(x.shape[0]):
            y = np.atleast_2d(x[i])
            z = get_2D_version(y, self.obj.pi1, self.obj.pi2)
            ret.append(self.get_objectives(z))
        return ret

    def evaluate_constraint(self, x, region):
        x = np.atleast_2d(x)
        self.check_valid_length(x)
        constr = np.zeros(x.shape[0])
        for i in range(constr.shape[0]):
            y = np.atleast_2d(x[i])
            z = get_2D_version(y, self.obj.pi1, self.obj.pi2)
            constr[i] = self.get_constraint_violation(z, region)
            return constr

    # TODO: check if we need to use this anymore. Check if include_boundary needed.
    def get_constraint_violation(self, x, region):
        return self.distance_from_region(region.centre, x) - region.radius

    def distance_from_region(self, region, x):
        x = np.atleast_2d(x)
        return euclidean_distance(region, x)[0]

    def evaluate(self, x):
        x = np.atleast_2d(x)
        self.check_valid_length(x)
        z = get_2D_version(x, self.obj.pi1, self.obj.pi2)
        return self.evaluate_2D(z)

    # used inside of DBMOPP object.
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
            "hard_constr_viol": self.get_hard_constraint_violation(x),
        }
        if ans["hard_constr_viol"]:
            if self.constraint_type == 3:
                if self.in_convex_hull_of_attractor_region(x):
                    ans["hard_constr_viol"] = False
                    ans["obj_vector"] = self.get_objectives(x)
            return ans

        ans["soft_constr_viol"] = self.get_soft_constraint_violation(x)
        if ans["soft_constr_viol"]:
            if self.constraint_type == 7:
                if self.in_convex_hull_of_attractor_region(x):
                    ans["soft_constr_viol"] = False
                    ans["obj_vector"] = self.get_objectives(x)
            return ans

        # Neither constraint breached
        if self.check_neutral_regions(x):
            ans["obj_vector"] = self.obj.neutral_region_objective_values
        else:
            ans["obj_vector"] = self.get_objectives(x)
        return ans

    def is_pareto_2D(self, x: np.ndarray):
        """ """
        if self.get_hard_constraint_violation(x):
            return False
        if self.get_soft_constraint_violation(x):
            return False
        return self.is_in_limited_region(x)["in_pareto_region"]

    def in_convex_hull_of_attractor_region(self, y: np.ndarray):
        """
        # Attractor region method?
        """
        self.check_valid_length(y)
        x = get_2D_version(y, self.obj.pi1, self.obj.pi2)

        for i, centre_region in enumerate(self.obj.centre_regions):
            if centre_region.is_inside(x):
                return self.obj.attractor_regions[i].in_hull(x)

        return False

    def check_valid_length(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.n:
            msg = f"Number of design variables in the argument does not match that required in the problem instance, \
            was {x.shape[1]}, should be {self.n}"
            raise Exception(msg)

    def set_up_attractor_centres(self):
        """
        Calculate max maximum region radius given problem properties
        """
        # number of local PO sets, global PO sets, dominance resistance regions
        n = self.nlp + self.ngp + self.ndr
        # Create the attractor objects
        self.obj.centre_regions = np.array([Region() for _ in range(n)])  # Different objects
        max_radius = 1 / (2 * np.sqrt(n) + 1) * (1 - (self.prop_neutral + self.prop_contraint_checker))  # prop 0 and 0.
        # Assign centres
        radius = self.place_region_centres(n, max_radius)
        # Assign radius
        self.place_region_radius(n, radius)
        # save indices of PO set locations
        self.obj.pareto_set_indices = np.arange(self.nlp, self.nlp + self.ngp)

    def place_region_radius(self, n, r):
        for i in range(n):
            self.obj.centre_regions[i].radius = r
        # reduce raddius if local fronts used
        if self.nlp > 0:
            for i in range(self.nlp + 1, n):
                self.obj.centre_regions[i].radius = r / 2
            w = np.linspace(1, 0.5, self.nlp + 1)
            # linearly decrease local front radius
            for i in range(self.nlp + 1):
                self.obj.centre_regions[i].radius = self.obj.centre_regions[i].radius * w[i]

    def place_region_centres(self, n: int, r: float):
        effective_bound = 1 - r
        threshold = 4 * r

        time_start = time()
        too_long = False
        max_elapsed = 1
        rand_coord = (np.random.rand(2) * 2 * effective_bound) - effective_bound
        self.obj.centre_regions[0].centre = rand_coord

        for i in range(1, n):  # looping the objects would be nicer
            while True:
                rand_coord = (np.random.rand(2) * 2 * effective_bound) - effective_bound
                distances = np.array([self.obj.centre_regions[i].get_distance(rand_coord) for i in range(i)])
                t = np.min(distances)
                if t > threshold:
                    # print("assigned centre", i)
                    break
                too_long = (time() - time_start) > max_elapsed
                if too_long:
                    break
            self.obj.centre_regions[i].centre = rand_coord

        if too_long:  # Took longer than max_elapsed.
            print("restarting attractor region placement with smaller radius...\n")
            return self.place_region_centres(n, r * 0.95)

        return r

    def place_attractors(self):
        """
        Randomly place attractor regions in 2D space
        """
        num_of_regions = self.nlp + self.ngp
        ini_locs = np.zeros((num_of_regions, 2, self.k))

        self.obj.attractor_regions = np.array([None] * (num_of_regions + self.ndr))

        # assign atractors per region for local and global fronts
        for i in range(num_of_regions):  # used to be arange dunno why.
            B = np.hstack(
                (
                    np.cos(self.obj.pareto_angles + self.obj.rotations[i]),
                    np.sin(self.obj.pareto_angles + self.obj.rotations[i]),
                )
            )

            locs = matlib.repmat(self.obj.centre_regions[i].centre, self.k, 1) + (
                matlib.repmat(self.obj.centre_regions[i].radius, self.k, 2) * B
            )

            convhull_locs = None
            if self.k > 2:
                mpoints = MultiPoint(locs)
                convhull_locs = mpoints.convex_hull

            # create attractor region
            self.obj.attractor_regions[i] = AttractorRegion(
                locations=locs,
                indices=np.arange(self.k),
                centre=self.obj.centre_regions[i].centre,
                radius=self.obj.centre_regions[i].radius,
                convhull=convhull_locs,
            )

            for k in np.arange(self.k):
                ini_locs[i, :, k] = locs[k, :]

        self.obj.attractors = np.array([Attractor() for _ in range(self.k)])

        for i in range(self.k):
            self.obj.attractors[i].locations = ini_locs[:, :, i]

        # now assign dominance resistance regions which have a subset of attractors per region active.
        for i in range(num_of_regions, num_of_regions + self.ndr):
            locs = matlib.repmat(self.obj.centre_regions[i].centre, self.k, 1) + (
                matlib.repmat(self.obj.centre_regions[i].radius, self.k, 2)
                * np.hstack(
                    (
                        np.cos(self.obj.pareto_angles + self.obj.rotations[i]),
                        np.sin(self.obj.pareto_angles + self.obj.rotations[i]),
                    )
                )
            )

            n_include = np.random.permutation(self.k - 1) + 1  # Plus one as we want to include at least one?
            n_include = n_include[0]  # Take the first one
            Idxs = np.argsort(np.random.rand(self.k))
            j = Idxs[:n_include]

            # only calculate convex hull if there are more than two points'
            convehull = None
            if len(locs[:, 1] > 2):
                mpoints = MultiPoint(locs)
                convehull = mpoints.convex_hull

            self.obj.attractor_regions[i] = AttractorRegion(
                locations=locs[j, :],
                indices=j,
                centre=None,
                radius=self.obj.centre_regions[i].radius,
                convhull=convehull,
            )

            for k in range(n_include):
                attractor_loc = self.obj.attractors[k].locations
                self.obj.attractors[k].locations = np.vstack((attractor_loc, locs[Idxs[k], :]))

    def place_disconnected_pareto_elements(self):
        # messy way of appending as many 0 as there is local fronts to start of disconnected_regions
        # to make sure disconnected_regions and attractor_regions match
        for i in range(self.nlp):
            self.obj.disconnected_regions.append(i)
        n = self.ngp - 1  # number of points to use to set up separete subregions
        # first get base angles in region of interest on unrotated Pareto set
        pivot_index = np.random.randint(self.k)  # get attractor at random
        # sort from smallest to largest and get the indices
        indices = np.argsort(self.obj.pareto_angles, axis=0)
        if pivot_index == 0:
            offset_angle_1 = self.obj.pareto_angles[indices[self.k - 1]]
        else:
            offset_angle_1 = self.obj.pareto_angles[indices[pivot_index - 1]]  # check this minus

        offset_angle_1 = offset_angle_1[0]

        if pivot_index == self.k - 1:
            offset_angle_2 = self.obj.pareto_angles[indices[0]]
        else:
            offset_angle_2 = self.obj.pareto_angles[indices[pivot_index + 1]]  # check plus

        offset_angle_2 = offset_angle_2[0]

        pivot_angle = self.obj.pareto_angles[indices[pivot_index]][0]  # dunno if needed to be 2d

        if pivot_angle == (offset_angle_1 or offset_angle_2):
            raise Exception("Angle should not be duplicated!")

        # this should be correct now.
        if offset_angle_1 < offset_angle_2:
            range_covered = offset_angle_1 + 2 * np.pi - offset_angle_2
            p1 = offset_angle_1 / range_covered
            r = np.random.rand(n)  # does this return vals to sum of 1 ??
            p1 = np.sum(r < p1)
            r[:p1] = 2 * np.pi + np.random.rand(p1) * offset_angle_1
            r[p1:n] = np.random.rand(n - p1) * (2 * np.pi - offset_angle_2) + offset_angle_2
            r = np.sort(r)
            r_angles = np.zeros(n + 2)
            r_angles[0] = offset_angle_2
            r_angles[n + 1] = offset_angle_1 + 2 * np.pi  # adding 2*pi as shifted for sorting
            r_angles[1 : n + 1] = r  # cause python's lists [)
        else:
            r = np.random.rand(n) * (offset_angle_1 - offset_angle_2) + offset_angle_2
            r = np.sort(r)
            r_angles = np.zeros(n + 2)
            r_angles[0] = offset_angle_2
            r_angles[n + 1] = offset_angle_1
            r_angles[1 : n + 1] = r

        k = self.nlp + self.ngp
        self.obj.pivot_locations = np.zeros((k, 2))
        self.obj.bracketing_locations_lower = np.zeros((k, 2))
        self.obj.bracketing_locations_upper = np.zeros((k, 2))

        def calc_location(ind, a):
            return self.obj.centre_regions[ind].calc_location(a, self.obj.rotations[ind])

        index = 0
        for i in range(self.nlp, k):  # verify indexing
            self.obj.pivot_locations[i, :] = calc_location(i, pivot_angle)
            self.obj.bracketing_locations_lower[i, :] = calc_location(i, r_angles[index])

            if self.pareto_set_type == 0:
                raise Exception("should not be calling this method with an instance with identical Pareto set regions")

            elif self.pareto_set_type == 2:
                self.obj.bracketing_locations_upper[i, :] = calc_location(i, r_angles[index + 1])
                vertices = np.asarray(self.obj.attractor_regions[i].convhull.exterior.coords)
                self.create_disconnected_po_regions(
                    self.obj.pivot_locations[i],
                    self.obj.bracketing_locations_lower[i],
                    self.obj.bracketing_locations_upper[i],
                    vertices,
                )

            elif self.pareto_set_type == 1:
                # inverted case
                if index == self.ngp - 1:
                    self.obj.bracketing_locations_lower[i, :] = calc_location(i, r_angles[1])
                    self.obj.bracketing_locations_upper[i, :] = calc_location(i, r_angles[n])
                    vertices = np.asarray(self.obj.attractor_regions[i].convhull.exterior.coords)
                    self.create_disconnected_po_regions(
                        self.obj.pivot_locations[i],
                        self.obj.bracketing_locations_lower[i],
                        self.obj.bracketing_locations_upper[i],
                        vertices,
                    )
                else:
                    self.obj.bracketing_locations_upper[i, :] = calc_location(i, r_angles[index + 2])
                    vertices = np.asarray(self.obj.attractor_regions[i].convhull.exterior.coords)
                    self.create_disconnected_po_regions(
                        self.obj.pivot_locations[i],
                        self.obj.bracketing_locations_lower[i],
                        self.obj.bracketing_locations_upper[i],
                        vertices,
                    )

            index += 1

    # creates the disconnected Polygons to plot. Does not change or affect og DBMOPP in any way
    def create_disconnected_po_regions(self, piv, lb, ub, x):
        vertices_between = []
        for ii in range(len(x) - 1):
            # if vertex is between the bracketing lines rooted at pivot, it will be added to the polygon as a corner
            if between_lines_rooted_at_pivot(x[ii], piv, lb, ub):
                vertices_between.append(list(x[ii]))
        coords = None
        # if there is vertices in between, add them to polygon
        if len(vertices_between) > 0:
            up = np.array([piv, lb])
            for j in range(len(vertices_between)):
                up = np.vstack((up, vertices_between[j]))
            bottom = np.array([ub, piv])
            coords = np.vstack((up, bottom))
        else:
            coords = np.array([piv, lb, ub, piv])

        # we only take unique vertexes to draw polygons
        coords = np.unique(coords, axis=0)
        self.obj.disconnected_regions.append(MultiPoint(coords))

    def place_vertex_constraint_locations(self):
        """
        Place constraints located at attractor points
        """
        print("Assigning any vertex soft/hard constraint regions\n")
        if self.constraint_type in [1, 5]:
            to_place = 0
            for i in range(len(self.obj.attractors)):  # or self.k as that should be the same...
                to_place += len(self.obj.attractors[i].locations)

            centres = np.zeros((to_place, 2))
            radii = np.zeros((to_place, 1))
            k = 0

            penalty_radius = np.random.rand(1) / 2
            for i, attractor_region in enumerate(self.obj.attractor_regions):
                for j in range(len(attractor_region.objective_indices)):
                    centres[k, :] = attractor_region.locations[j, :]  # Could make an object here...
                    radii[k] = attractor_region.radius * penalty_radius
                    k += 1

            if self.constraint_type == 1:
                self.obj.hard_constraint_regions = np.array([Region() for _ in range(to_place)])
                for i, hard_constraint_region in enumerate(self.obj.hard_constraint_regions):
                    hard_constraint_region.centre = centres[i, :]
                    hard_constraint_region.radius = radii[i]
            else:
                self.obj.soft_constraint_regions = np.array([Region() for _ in range(to_place)])
                for i, soft_constraint_region in enumerate(self.obj.soft_constraint_regions):
                    soft_constraint_region.centre = centres[i, :]
                    soft_constraint_region.radius = radii[i]

    def place_centre_constraint_locations(self):
        """
        Place center constraint regions
        """
        print("Assigning any centre soft/hard constraint regions.\n")
        if self.constraint_type == 2:
            self.obj.hard_constraint_regions = self.obj.centre_regions
        elif self.constraint_type == 6:
            self.obj.soft_constraint_regions = self.obj.centre_regions

    def place_moat_constraint_locations(self):
        """
        Place moat constraint regions
        """
        print("Assigning any moat soft/hard constraint regions\n")
        r = np.random.rand() + 1
        if self.constraint_type == 3:
            self.obj.hard_constraint_regions = self.obj.centre_regions
            for i in range(len(self.obj.hard_constraint_regions)):
                self.obj.hard_constraint_regions[i].radius = self.obj.hard_constraint_regions[i].radius * r
        elif self.constraint_type == 7:
            self.obj.soft_constraint_regions = self.obj.centre_regions
            for i in range(len(self.obj.soft_constraint_regions)):
                self.obj.soft_constraint_regions[i].radius = self.obj.soft_constraint_regions[i].radius * r

    def place_discontinunities_neutral_and_checker_constraints(self):
        print("Assigning any checker soft/hard constraint regions and neutral regions\n", self.prop_contraint_checker)
        if (self.prop_contraint_checker + self.prop_neutral) > 0:

            S = (np.random.rand(self.nm, 2) * 2) - 1
            print(S.shape)
            for _i, centre_region in enumerate(self.obj.centre_regions):
                to_remove = centre_region.is_inside(S, True)
                not_to_remove = np.logical_not(to_remove)
                S = S[not_to_remove, :]

            if S.shape[0] < self.nm * (self.prop_contraint_checker + self.prop_neutral):
                msg = "Not enough space outside of attractor regions to match requirement of constrained+neural space"
                raise Exception(msg)

            # Now iteratively place a centre, check legality, and update
            # proportion of space covered as estimated using the MC samples
            # falling inside the neutral/penality region
            # Note, by definition, all samples in S are legal centres
            # outside of attractor regions, and are randomly ordered
            # so whill just select from these to speed up the process
            if self.prop_contraint_checker > 0:
                regions, S = self.set_not_attractor_regions_as_proportion_of_space(S, self.prop_contraint_checker, [])
                if self.constraint_type == 4:
                    self.obj.hard_constraint_regions = regions
                elif self.constraint_type == 8:
                    self.obj.soft_constraint_regions = regions
                else:
                    raise Exception(f"constraintType should be 8 or 4 to reach here is {self.constraint_type}")

            # Neutral space
            if self.prop_neutral > 0:
                regions, _ = self.set_not_attractor_regions_as_proportion_of_space(S, self.prop_neutral, regions)
                self.obj.neutral_regions = regions

        # print("TODO check discontinuity, not done in matlab")

    def set_not_attractor_regions_as_proportion_of_space(self, S, proportion_to_attain, other_regions):
        allocation = 0
        regions = []
        while allocation < proportion_to_attain:
            region = Region()
            region.centre = S[-1, :]

            centre_list = np.zeros((len(self.obj.centre_regions), 2))
            centre_radii = np.zeros(len(self.obj.centre_regions))
            for i, centre_region in enumerate(self.obj.centre_regions):
                centre_list[i] = centre_region.centre
                centre_radii[i] = centre_region.radius
            other_centres = np.zeros((len(other_regions), 2))
            other_radii = np.zeros(len(other_regions))

            for i, other_region in enumerate(other_regions):
                other_centres[i] = other_region.centre
                other_radii[i] = other_region.radius

            both_centres = np.vstack((centre_list, other_centres)) if other_centres.shape[0] > 0 else centre_list

            d = euclidean_distance(both_centres, S[-1, :])
            d = d - np.hstack((centre_radii, other_radii))
            d = np.min(d)

            if d <= 0:
                raise Exception("Should not get here")

            c_r = np.sqrt((proportion_to_attain - allocation) / np.pi)
            r = np.random.rand(1) * np.minimum(d, c_r)
            region.radius = r
            regions.append(region)
            S = S[:-1, :]  # remove last row

            d = euclidean_distance(S, region.centre)
            Idx = d > r  # this was d > r before
            S = S[Idx, :]  # Remove covered points
            # flake8 does not like this. TODO: fix but make sure logic won't change, will break constraints
            covered_count = (Idx == False).sum() + 1

            allocation += covered_count / self.nm

        return np.array(regions), S

    # used for moproblem
    def check_region_prob(self, regions, x, include_boundary):
        if regions is None:
            return False
        x = np.atleast_2d(x)
        in_region = np.zeros(regions.size, dtype=bool)
        d = np.zeros(regions.size)
        for i, region in enumerate(regions):
            if region.is_inside(x, include_boundary):
                in_region[i] = True
            else:
                in_region[i] = False
            d[i] = euclidean_distance(region.centre, x)[0]

        return in_region, d

    # used for dbmopp stuff
    def check_region(self, regions, x, include_boundary) -> bool:
        if regions is None:
            return False
        for region in regions:
            if region.is_inside(x, include_boundary):
                return True
        return False

    def check_neutral_regions(self, x):
        return self.check_region(self.obj.neutral_regions, x, True)

    # Matlab code has hard constraints as true or false
    def get_hard_constraint_violation(self, x) -> bool:
        in_hard_constraint_region = self.check_region(self.obj.hard_constraint_regions, x, False)
        return in_hard_constraint_region

    # gets contstraint violations for MOProblem
    def get_constraint_violations(self, x, include_boundary) -> np.ndarray:
        if include_boundary:
            in_constraint_region, d = self.check_region_prob(self.obj.soft_constraint_regions, x, True)
            constraint_regions = self.obj.soft_constraint_regions
        else:
            in_constraint_region, d = self.check_region_prob(self.obj.hard_constraint_regions, x, False)
            constraint_regions = self.obj.hard_constraint_regions

        violations = np.zeros_like(in_constraint_region, dtype=float)

        for i in in_constraint_region:
            if in_constraint_region.size > 0:
                for i in range(violations.shape[0]):
                    violations[i] = d[i] - constraint_regions[i].radius

        return violations

    def get_soft_constraint_violation(self, x) -> bool:
        in_soft_constraint_region = self.check_region(self.obj.soft_constraint_regions, x, True)
        return in_soft_constraint_region

    def get_minimun_distance_to_attractors(self, x: np.ndarray):
        """ """
        y = np.zeros(self.k)
        for i, attractor in enumerate(self.obj.attractors):
            y[i] = attractor.get_minimum_distance(x)
        y *= self.obj.rescaleMultiplier
        y += self.obj.rescaleConstant
        return y

    def get_minimum_distances_to_attractors_overlap_or_discontinuous_form(self, x):
        y = self.get_minimun_distance_to_attractors(x)
        in_pareto_region, in_hull, index = self.is_in_limited_region(x).values()
        if in_hull:
            if not in_pareto_region:
                y += self.obj.centre_regions[index].radius
        return y

    def get_objectives(self, x):
        if self.pareto_set_type == 0:
            y = self.get_minimun_distance_to_attractors(x)
        else:
            y = self.get_minimum_distances_to_attractors_overlap_or_discontinuous_form(x)

        y = self.update_with_discontinuity(x, y)
        y = self.update_with_neutrality(x, y)
        return y

    def is_in_limited_region(self, x, eps=1e-16):
        """ """
        ans = {"in_pareto_region": False, "in_hull": False, "index": -1}

        if x.shape[0] != 2:
            # print(x)
            x = x[0]
        # can still be improved?
        Idx = np.array([i for i in range(len(self.obj.centre_regions)) if self.obj.centre_regions[i].is_close(x, eps)])
        if len(Idx) > 0:  # is not empty
            i = Idx[0]

            if self.nlp <= i < self.nlp + self.ngp:
                if self.constraint_type in [2, 6]:
                    # Smaller of dist
                    dist = self.obj.centre_regions[i].get_distance(x)
                    radius = self.obj.centre_regions[i].radius
                    r = np.min(np.abs(dist), np.abs(radius))
                    if np.abs(dist) - radius < 1e4 * eps * r:
                        ans["in_hull"] = True

                else:
                    xpoint = Point(x)
                    t = self.obj.attractor_regions[i].convhull.contains(xpoint)
                    if t:
                        ans["in_hull"] = True

        if self.pareto_set_type == 0 or self.constraint_type in [2, 6]:
            ans["in_pareto_region"] = ans["in_hull"]
            ans["in_hull"] = False
        else:
            if ans["in_hull"]:
                ans["index"] = Idx[0]
                res = between_lines_rooted_at_pivot(
                    x,
                    self.obj.pivot_locations[Idx[0], :],
                    self.obj.bracketing_locations_lower[Idx[0], :],
                    self.obj.bracketing_locations_upper[Idx[0], :],
                )
                ans["in_pareto_region"] = res
                if self.pareto_set_type == 1:
                    if Idx[0] == self.nlp + self.ngp - 1:
                        ans["in_pareto_region"] = not ans[
                            "in_pareto_region"
                        ]  # special case where last region is split at the two sides, should not get here everytime

        return ans

    def update_with_discontinuity(self, x, y):
        return self.update(
            self.obj.discontinuous_regions,
            self.obj.discontinuous_region_objective_value_offset,
            x,
            y,
        )

    def update_with_neutrality(self, x, y):
        return self.update(self.obj.neutral_regions, self.obj.neutral_region_objective_values, x, y)

    def update(self, regions, offsets, x, y):
        if regions is None:
            return y
        distances = np.zeros(len(regions))
        for i, region in enumerate(regions):
            distances[i] = region.get_distance(x) if region.is_inside(x, include_boundary=True) else 0
        if np.any(distances > 0):
            index = np.argmin(distances)  # molst likely will return the index of the first 0
            y = y + offsets[index, :]
        return y

    # PLOTTING

    def plot_problem_instance(self):
        """ """
        fig, ax = plt.subplots()
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        # Plot local Pareto regions
        for i in range(self.nlp):
            self.obj.attractor_regions[i].plot(ax, "g")  # Green

        # if not type 0, we need to find the intersection between the connected pareto regions polygon and
        # disconnected pareto regions polygon formed from pivot_locs and brackets in place_disconnected_pareto_elements.
        if self.pareto_set_type != 0:
            for i in range(self.nlp, self.nlp + self.ngp):
                poly1 = self.obj.attractor_regions[i].convhull
                poly2 = self.obj.disconnected_regions[i]
                poly2 = Polygon(poly2.convex_hull)

                # for testing
                # patch_poly = PolygonPatch(poly1, facecolor="green")
                # patch_poly2 = PolygonPatch(poly2, facecolor="blue")
                # ax.add_patch(patch_poly)
                # ax.add_patch(patch_poly2)

                # if last region, in inverted case we have splitting
                if self.pareto_set_type == 1 and i == self.nlp + self.ngp - 1:
                    diff = poly1.difference(poly2)
                    difference = PolygonPatch(diff, facecolor="r")
                    ax.add_patch(difference)
                # otherwise just intersection
                else:
                    inter = poly1.intersection(poly2)
                    intersect = PolygonPatch(inter, facecolor="r")
                    ax.add_patch(intersect)

        else:
            # global pareto regions
            for i in range(self.nlp, self.nlp + self.ngp):
                self.obj.attractor_regions[i].plot(ax, "r")

        # dominance resistance set regions
        for i in range(self.nlp + self.ngp, self.nlp + self.ngp + self.ndr):
            # attractor regions should take care of different cases
            self.obj.attractor_regions[i].plot(ax, "b")

        def plot_constraint_regions(constraint_regions, color):
            if constraint_regions is None:
                return
            for constraint_region in constraint_regions:
                constraint_region.plot(color, ax)

        plot_constraint_regions(self.obj.hard_constraint_regions, "black")
        plot_constraint_regions(self.obj.soft_constraint_regions, "grey")
        plot_constraint_regions(self.obj.neutral_regions, "c")

    def plot_landscape_for_single_objective(self, index, res=500):
        if res < 1:
            raise Exception("Cannot grid the space with a resolution less than 1")
        if index not in np.arange(self.k):
            raise Exception(f"Index should be between 0 and {self.k-1}, was {index}.")

        xy = np.linspace(-1, 1, res)
        x, y = np.meshgrid(xy, xy)

        z = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                decision_vector = np.hstack((xy[i], xy[j]))
                obj_vector = self.evaluate_2D(decision_vector)["obj_vector"]
                obj_vector = np.atleast_2d(obj_vector)
                z[i, j] = obj_vector[0, index]

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.view_init(elev=90, azim=-90)

        surf = ax.plot_surface(
            x,
            y,
            z.T,
            cmap=cm.plasma,
            linewidth=0,
            antialiased=False,
            vmin=np.nanmin(z),
            vmax=np.nanmax(z),
        )

        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()

    def plot_pareto_set_members(self, resolution=500):
        if resolution < 1:
            raise Exception("Cannot grid the space with a resolution less than 1")
        fig, ax = plt.subplots()

        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        xy = np.linspace(-1, 1, resolution)
        po_set = np.empty((0, 2), float)

        for x in xy:
            for y in xy:
                z = np.array([x, y])
                if self.is_pareto_2D(z):
                    po_set = np.vstack((po_set, z))
                    ax.scatter(x, y, color="black", s=1)

        plt.show()
        return po_set

    def plot_dominance_landscape(self, res=500, moore_neighbourhood=True):
        print("Plotting dominance landscape not implemented!")
        return

    """
        if res < 1:
            raise Exception("Cannot grid the space with a resolution less than 1")
        xy = np.linspace(-1, 1, res)
        y = np.zeros((self.k, res, res))
        for i in range(res):
            for j in range(res):
                decision_vector = np.hstack((xy[i], xy[j]))
                obj_vector = self.evaluate_2D(decision_vector)["obj_vector"]
                obj_vector = np.atleast_2d(obj_vector)
                y[:, i, j] = obj_vector

        return self.plot_dominance_landscape_from_matrix(y, xy, xy, moore_neighbourhood)
    """

    def plot_dominance_landscape_from_matrix(self, z, x, y, moore_neighbourhood):
        (
            basins,
            neutral_areas,
            dominated,
            destination,
            dominating_neighbours,
            offset,
        ) = self.get_dominance_landscape_basins_from_matrix(z, x, y, moore_neighbourhood)
        # TODO: there must be a way to optimize the code..

    def get_dominance_landscape_basins_from_matrix(self, z, x, y, moore_neighbourhood):
        num_obj, res, r = z.shape

        # some checks
        assert res == r, "Second and third dimension of z must be the same size"
        assert self.k >= 2, "must have atleast two objectives"
        assert x.shape[0] == res, "must be as many x grid labels as elements"
        assert y.shape[0] == res, "must be as many y grid labels as elements"

        if moore_neighbourhood:
            dominating_neighbours = np.zeros((res, res, 8))
            neutral_neighbours = np.zeros((res, res, 8))
        else:
            dominating_neighbours = np.zeros((res, res, 4))
            neutral_neighbours = np.zeros((res, res, 4))
        dominated = np.ones((res, res))

        # array holding neighbourhood directions
        offset = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        # determine those which r not dominated
        for i in range(res):
            for j in range(res):
                (
                    dominating_neighbours[i, j, :],
                    neutral_neighbours[i, j, :],
                    dominated[i, j],
                ) = self.identify_dominating_neighbours(z, i, j, res, moore_neighbourhood, offset)

        # dominating_neighbours now holds location of neigbours which dominate and dominated holds whether a particular
        # location is dominated by any neigbour
        neutral_areas = np.ones((res, res)) * -1
        neutral_idx = 1
        # label contiguos neutral areas
        for i in range(res):
            for j in range(res):
                # if not dominated and not yet assigned to a neutral area
                if dominated[i, j] == 0 and neutral_areas[i, j] < 0:
                    neutral_areas = self.identify_neutral_area_members(
                        i,
                        j,
                        neutral_areas,
                        neutral_idx,
                        dominated,
                        neutral_neighbours,
                        moore_neighbourhood,
                        offset,
                    )
                    neutral_idx = neutral_idx + 1

        # identify basins of attraction
        basins = np.ones((res, res)) * -1
        basins[neutral_areas > 0] = 0
        processed = np.zeros((res, res))
        number_distinct_neutral_regions = np.max(np.max(neutral_areas))  # check if this is correct
        destination = [[] for d in range(res)]
        for i in range(res):
            for j in range(res):
                if processed[i, j] == 0:
                    processed, destination, _ = self.update_destinations(
                        i,
                        j,
                        processed,
                        destination,
                        dominating_neighbours,
                        neutral_areas,
                        offset,
                    )

        # now fill value in basins
        for i in range(res):
            for j in range(res):
                if neutral_areas[i, j] > 0:
                    basins[i, j] = 0
                else:
                    if len(destination[i, j]) == 1:
                        # put between 0.25 and 0.75, graded by which basin it leads to
                        basins[i, j] = 0.25 + destination[i, j] / (2 * number_distinct_neutral_regions)
                    else:
                        basins[i, j] = 1

        return (
            basins,
            neutral_areas,
            dominated,
            destination,
            dominating_neighbours,
            offset,
        )

    def identify_dominating_neighbours(self, z, i, j, res, moore_neighbourhood, offset):
        if moore_neighbourhood:
            n = 8
        else:
            n = 4
        dominating_neighbours = np.zeros((n, 1))
        neutral_neighbours = np.zeros((n, 1))

        for k in range(n):
            if i + offset[k, 1] > 0 and i + offset[k, 1] <= res and j + offset[k, 2] > 0 and j + offset[k, 2] <= res:
                (
                    dominating_neighbours[k],
                    neutral_neighbours[k],
                ) = self.vector_is_dominated_or_neutral(z[:, i, j], z[:, i + offset[k, 1] + offset[k, 2]])

        dominated = np.any(dominating_neighbours)  # check its same as matlabs any
        return dominating_neighbours, neutral_neighbours, dominated

    def identify_neutral_area_members(
        self,
        i,
        j,
        neutral_areas,
        n_idx,
        dominated,
        neutral_neighbours,
        moore_neighbourhood,
        offset,
    ):
        pass

    def update_destinations(self, i, j, processed, destination, dominating_neighbours, neutral_areas, offset):
        pass
        # TODO: make work on 0s and 1s instead of true and falses, since np arrays are using 0s as false and 1s as true

    def vector_is_dominated_or_neutral(x, y):
        # TODO: check if works make properly if not
        def vector_dominates(x1, x2):
            return np.sum(x1 <= x2) == x1.shape[0] and np.sum(x1 < x2) > 0

        # returns is true if y dominates x.
        # n is true if x and y are incomparable under the dominates relation
        d = vector_dominates(y, x)
        if d is False:
            n = not vector_dominates(x, y)
        else:
            n = False

        return d, n  # as booleans below as ints
        # return int(d), int(n)

    def unit_hypercube_simplex_sample(self, dim, sum_value):
        no_points = 1  # TODO: make sure this can be one..
        X = np.random.exponential(np.ones((no_points, dim)))
        S = np.sum(X)  # drop array bracket or not

        if sum_value == 1:
            X = np.divide(X, matlib.repmat(S, 1, dim))
        elif sum_value < 1:
            X = (np.divide(X, matlib.repmat(S, 1, dim))) * sum_value
        else:
            if sum_value < dim / 2:
                # rejection sampling
                X = self.recalibrate(X, no_points, S, sum_value, dim)
            elif sum_value < dim - 1:
                # flipped around dim/2 face for rejection sampling
                X = 1 - self.recalibrate(X, no_points, S, dim - sum_value, dim)
            else:
                # special case when sum value >= dim -1, just flip round the scaled unit simplex no rejection sampling
                X = (np.divide(X, matlib.repmat(S, 1, dim))) * (dim - sum_value)
                X = 1 - X
        return X

    def recalibrate(self, Z, npoints, S, sum_value, dim):
        X = (np.divide(Z, matlib.repmat(S, 1, dim))) * sum_value

        for i in range(npoints):
            while (np.max(X[i])) > 1:  # rejection sampling
                Z[i, :] = np.random.exponential(np.ones((1, dim)))
                S = np.sum(Z[i])
                X[i, :] = Z[i, :] / S * sum_value

        return X

    def get_vectors_mapping_to_location(self, x):
        z = np.zeros(self.n)

        def process_dims(z, x, pi):
            # print("Processing dims")
            # print(z)
            pi_mag = int(np.sum(pi))
            if pi_mag == 1:
                z[pi] = x
            else:
                # map value from [-1, 1] to [0,1]
                x = ((x + 1) / 2) * pi_mag
                s = self.unit_hypercube_simplex_sample(pi_mag, x)[0]
                s = (s * 2) - 1  # map s back to [-1, 1]
                z[pi] = s
            return z

        z = process_dims(z, x[0], self.obj.pi1)
        z = process_dims(z, x[1], self.obj.pi2)
        return z

    # returns random pareto set member uniformly from the Pareto set and the point in 2D it maps to
    # should not be used during optimization
    def get_Pareto_set_member(self) -> Tuple:
        # while not legal point obtained, get random pareto centre
        invalid = True
        x = []
        point = []
        low = np.min(self.obj.pareto_set_indices)
        high = np.max(self.obj.pareto_set_indices)
        centres = self.obj.centre_regions
        iters = 0

        while invalid:

            k = np.random.randint(low, high + 1)  # + self.nlp + self.ndr. +1 bc randint is [)
            angle = np.random.rand() * 2.0 * np.pi

            # 2D case
            if self.constraint_type == 2 or self.constraint_type == 6:
                # if centre constraints used, randomly choose angle and use that radius from
                # Pareto set centre list and project
                x = centres[k].centre + [
                    centres[k].radius * np.cos(angle),
                    centres[k].radius * np.sin(angle),
                ]
                invalid = False
            else:
                # generate random point in Circle
                r = centres[k].radius * np.sqrt(np.random.rand())
                x = centres[k].centre + [r * np.cos(angle), r * np.sin(angle)]
                if self.is_pareto_2D(x):
                    invalid = False

            iters += 1

        # project to higher dims if needed
        if self.n > 2:
            # design spcae is larger than 2d, so need to randomly select a locations
            # in this higher dim space which maps to this Pareto location
            point = x
            x = self.get_vectors_mapping_to_location(x)
        else:
            point = x

        return (x, point)

    # test for get Pareto set..
    def get_Pareto_set(self, points):
        print("getting the pareto set approximation")
        # while not legal point obtained, get random pareto centre
        x = []
        point = []
        low = np.min(self.obj.pareto_set_indices)
        high = np.max(self.obj.pareto_set_indices)
        centres = self.obj.centre_regions

        centre_list = []
        counter = 0

        # results = np.zeros((points, 2))
        results = []
        results2d = []
        k = low

        while len(results2d) < points:
            invalid = True
            while invalid and len(centre_list) <= len(centres):

                if k >= high:
                    k = low
                # k = np.random.randint(low, high) #+ self.nlp + self.ndr
                angle = np.random.rand() * 2.0 * np.pi

                # 2D case
                if self.constraint_type == 2 or self.constraint_type == 6:
                    # if centre constraints used, randomly choose angle and use that radius from
                    # Pareto set centre list and project
                    x = centres[k].centre + [
                        centres[k].radius * np.cos(angle),
                        centres[k].radius * np.sin(angle),
                    ]
                    invalid = False
                else:
                    # generate random point in Circle
                    r = centres[k].radius * np.sqrt(np.random.rand())
                    x = centres[k].centre + [r * np.cos(angle), r * np.sin(angle)]
                    if self.is_pareto_2D(x):
                        if k not in centre_list:
                            centre_list.append(k)
                        invalid = False

                k += 1
            counter += 1
            # project to higher dims if needed
            if self.n > 2:
                # design spcae is larger than 2d, so need to randomly select a locations
                # in this higher dim space which maps to this Pareto location
                point = x
                x = self.get_vectors_mapping_to_location(x)
            else:
                point = x
            results.append(x)
            results2d.append(point)
        return results, results2d


if __name__ == "__main__":

    n_objectives = 5
    n_variables = 5
    n_local_pareto_regions = 2
    n_dominance_res_regions = 0
    n_global_pareto_regions = 4
    const_space = 0.6
    pareto_set_type = 2
    constraint_type = 8
    ndo = 2  # numberOfdiscontinousObjectiveFunctionRegions
    neutral_space = 0.0

    # 0: No constraint, 1-4: Hard vertex, centre, moat, extended checker,
    # 5-8: soft vertex, centre, moat, extended checker.

    problem = DBMOPP_generator(
        n_objectives,
        n_variables,
        n_local_pareto_regions,
        n_dominance_res_regions,
        n_global_pareto_regions,
        const_space,
        pareto_set_type,
        constraint_type,
        ndo,
        False,
        False,
        neutral_space,
        10000,
    )
    print(problem._print_params())
    print("Initializing works!")

    # print(problem.)

    # x, point = problem.get_Pareto_set_member()
    # print("A pareto set member ", x)
    # print("A corresponding 2D point", point)
    # n_of_points = 10
    # po_list = np.zeros((n_of_points, problem.n))
    # po_points = np.zeros((n_of_points, 2))
    # for i in range(n_of_points):
    #    result = problem.get_Pareto_set_member()
    #    po_list[i] = result[0]
    #    po_points[i] = result[1]

    # print(po_points.shape)
    # po_list, po_points = problem.get_Pareto_set(50)
    # print(po_list)
    # print(po_points)

    # plt.scatter(x=po_points[:, 0], y=po_points[:, 1], s=5, c="r", label="Pareto set members")
    # plt.title("Pareto set members
    # plt.xlabel("F1")
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # plt.ylabel("F2")
    # plt.legend()hard_constraint_regions = None

    x = np.array(np.random.rand(3, n_variables))
    print(x)
    # For desdeos MOProblem only
    moproblem = problem.generate_problem()
    print("\nFormed MOProblem: \n\n", moproblem.evaluate(x))
    problem.plot_problem_instance()

    # need to get the population
    # po_set = problem.plot_pareto_set_members(100)
    # print(po_set[:5])
    # problem.plot_landscape_for_single_objective(0, 500)
    # problem.plot_dominance_landscape(10)

    plt.show()
