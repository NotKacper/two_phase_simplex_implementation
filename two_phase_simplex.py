import numpy as np


class TwoPhaseSimplex:
    def __init__(self, A: list[list[float]], b: list[float], c: list[float]) -> None:
        """
        Initializes the TwoPhaseSimplex solver object in terms of it's distinguishing components
        The problem to be solved is of the form: min c*x, s.t. Ax=b.
        Assumes that the problem must be translated to auxillary form.
        """
        valid, message = self.__check_validity_of_arguments(A,b,c)
        if not(valid):
            raise Exception("Invalid linear programming problem described: " + message)
        self.__A: np.ndarray = np.array(A)
        self.__b: np.ndarray = np.array(b)
        self.__c: np.ndarray = np.array(c)
        self.__tableau: np.ndarray = np.zeros((len(A) + 1, len(A)+len(A[0])+1))
        self.__solution: np.ndarray = np.zeros(len(c))

    def __check_validity_of_arguments(self, A :list[list[float]], b : list[float] , c : list[float]) -> tuple[bool, str]:
        """
        Returns a bool referring to whether the input makes any sense as a linear program, if false it contains a
        meaningful error message pertaining as to why the input doesn't make any sense.
        """
        m : int = len(A)
        if (m <= 0):
            return False, "A should have a postive number of rows"
        n : int = len(A[0])
        if (n <= 0):
            return False, "A should have a positive number of columns"
        if (len(b) != m):
            return False, "Vector b does not have the same length as matrix A has rows"
        elif (len(c) != n):
            return False, "Vector c does not have the same length as matrix A has columns"
        return True, ""

    def __construct_tableau(self) -> None:
        """
        Constructs the two phase simplex tableau
        """
        m : int
        n : int
        m, n = self.__A.shape
        e: np.ndarray = np.ones(m)
        # top left entry is -e*b, e = (1,...,1)
        self.__tableau[0, 0] = -np.dot(e, self.__b)
        # tab[0, 1:n+1] (exclusive) entries are -e*A
        self.__tableau[0, 1:n+1] = -np.matmul(e, self.__A)
        # tab[1:m+1,0] = b.
        self.__tableau[1:m+1, 0] = self.__b
        # tab[1:m+1, 1:n+1] = A
        self.__tableau[1:m+1, 1:n+1] = self.__A
        # tab[n+1:, 1:] = Identity Matrix
        self.__tableau[1:m+1, n+1:n+m+1] = np.identity(m)

    def __find_pivot(self) -> tuple[int, int]:
        """
        Find the pivot (r,s) on the tableau using Bland's rule.
        """
        m: int = self.__tableau.shape[0]
        s = 1
        while self.__tableau[0, s] >= 0:
            s += 1
        min_ratio = np.inf
        r = 0
        for i in range(1, m):
            if self.__tableau[i][s] > 0:
                temp = self.__tableau[i, 0]/self.__tableau[i, s]
                if min_ratio > temp:
                    min_ratio = temp
                    r = i
        return (r, s)

    def __pivot(self, r: int, s: int) -> None:
        """
        Induces pivoting operations on the (r,s) entry of the tableau,
        this causes the rth row to be divide by tableau[r,s] and then all entries
        on the sth column to become 0 other than where the row is r.
        """
        m: int = self.__tableau.shape[0]
        self.__tableau[r] /= self.__tableau[r, s]
        for i in range(m):
            if i != r:
                self.__tableau[i] -= self.__tableau[r]*self.__tableau[i, s]

    def __solve_auxillary_problem(self) -> np.ndarray:
        """
        Solves the auxillary problem using the phase 1 simplex method
        """
        m : int
        n : int
        m, n = self.__A.shape
        # basis is always the auxillary variables
        basis: np.ndarray = np.arange(n+1, n+m+1)
        while (self.__tableau[0, 1:].min() < 0):
            r, s = self.__find_pivot()
            if r != 0:
                self.__pivot(r, s)
            basis[r-1] = s  # beware of this line for the time being
        return basis

    def __check_tableau_for_constraint_violation(self, basis: np.ndarray) -> bool:
        """
        Ensures that if auxillary variables are in basis then they are equal to 0
        """
        m : int
        n : int
        m, n = self.__A.shape
        for var in basis:
            if var >= n+1 and var <= m+n and var != 0:
                return False
        return True

    def __drive_auxillary_variables_from_basis(self, basis: np.ndarray) -> np.ndarray:
        """
        Removes auxillary variables from the basis of the simplex tableau, sets up for phase 2
        """
        m : int
        n : int
        m, n = self.__A.shape
        for r, var in enumerate(basis):
            # if auxillary variable is in basis
            if var >= n+1 and var <= m+n:
                # pivot on any non-zero entry on that basis variables row
                for s, var2 in enumerate(self.__tableau[r+1,1:]):
                    if (var2 != 0):
                        self.__pivot(r+1, s+1)
                        # to drive the auxillary variable from basis
                        basis[r] = s+1
                        break
        return basis

    def __get_solution(self, basis: np.ndarray):
        """
        Gets the current solution stored in the tableau, all variables not included in basis are 0 by construction
        Note that the solutions array returned has 0 indexed variable numbers (i.e. 0th index is the 1st variable).
        """
        m: int = self.__A.shape[0]
        n: int = self.__A.shape[1]
        solutions: np.ndarray = np.zeros(n)
        for i in range(len(basis)):
            # basis contains the numbered variable which is in the index
            solutions[basis[i]-1] = self.__tableau[i+1, 0]
        return solutions

    def __change_tableau_to_phase_two(self, basis: np.ndarray) -> None:
        """
        changes the objective function and removes columns of the auxillary variables,
        preparing the tableau for phase 2.
        """
        phase_one_solution: np.ndarray = self.__get_solution(basis)
        self.__tableau[0, 0] = -np.dot(self.__c, phase_one_solution)
        n: int = self.__A.shape[1]
        # take a copy of the tableau and remove the auxillary variables
        temp: np.ndarray = self.__tableau[0:, 0:n+1].copy()
        # make the new tableau this copy
        self.__tableau = temp
        # add in new constraints (of original problem)
        self.__tableau[0, 1:] = self.__c

    def solve_phase_one(self) -> np.ndarray:
        self.__construct_tableau()
        basis: np.ndarray = self.__solve_auxillary_problem()
        return basis

    def __solve_phase_two(self, basis: np.ndarray) -> np.ndarray:
        """
        Solves the phase 2 simplex by applying the phase 1 simplex onto the 
        new tableau.
        """
        while (self.__tableau[0, 1:].min() < 0):
            r, s = self.__find_pivot()
            self.__pivot(r, s)
            basis[r-1] = s  # beware of this line for the time being
        return basis

    def __store_solution(self, basis: np.ndarray) -> None:
        """
        Writes the solution according to the current basis and tableau into the solution property
        """
        self.__solution = self.__get_solution(basis)

    def solve_program(self) -> bool:
        """
        Induces the solving of the phase 2 simplex problem prescribed by the object
        Returns True if the problem is bounded
        Returns False if the problem is unbounded
        """
        basis: np.ndarray = self.solve_phase_one()
        basis = self.__drive_auxillary_variables_from_basis(basis)
        valid: bool = self.__check_tableau_for_constraint_violation(basis)
        # if any optimal auxillary variable is non-zero then throw an error
        # this will only return if and only if it is not possible to drive
        # the variable from the basis (although there always should be a way)
        if (not valid):
            return False
        self.__change_tableau_to_phase_two(basis)
        basis = self.__solve_phase_two(basis)
        self.__store_solution(basis)
        return True

    def get_tableau(self) -> np.ndarray:
        """Returns the tableau current state"""
        return self.__tableau

    def get_solution(self) -> dict:
        """
        Returns dict containing the solution to the given system
        """
        solution_dict = {}
        for i, soln in enumerate(self.__solution):
            # have to translate to 1 index variable names
            solution_dict[f"x{i+1}"] = soln
        return solution_dict
