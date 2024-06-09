import numpy as np

class TwoPhaseSimplex:
    def __init__(self, A : list[int,int], b : list[int], c : list[int]) -> None:
        """
        Initializes the TwoPhaseSimplex solver object in terms of it's distinguishing components
        The problem to be solved is of the form: min c*x, s.t. Ax=b.
        Assumes that the problem must be translated to auxillary form.
        """
        self.__A : np.array[int][int] = np.array(A)
        self.__b : np.array[int] = np.array(b)
        self.__c : np.array[int] = np.array(c)
        self.__tableau : np.array[int] = np.zeros((len(A) + 1, len(A)+len(A[0])+1))

    def __construct_tableau(self) -> None:
        """
        Constructs the two phase simplex tableau
        """
        m : int = self.__A.shape[0]
        n : int = self.__A.shape[1]
        e : np.array[int] = np.ones(m)
        # top left entry is -e*b, e = (1,...,1)
        self.__tableau[0,0] = -np.dot(e,self.__b)
        # tab[0, 1:n+1] (exclusive) entries are -e*A
        self.__tableau[0,1:n+1] = -np.matmul(e, self.__A)
        # tab[1:m+1,0] = b.
        self.__tableau[1:m+1,0] = self.__b
        # tab[1:m+1, 1:n+1] = A
        self.__tableau[1:m+1, 1:n+1] = self.__A
        # tab[n+1:, 1:] = Identity Matrix
        self.__tableau[1:m+1, n+1:n+m+1] = np.identity(m)

    def __find_pivot(self) -> tuple[int,int]:
        """
        Find the pivot (r,s) on the tableau using Bland's rule
        """
        m : int = self.__tableau.shape[0]
        s = 1
        while self.__tableau[0,s] >= 0:
            s+=1
        min_ratio = np.inf
        r = 0
        for i in range(1,m):
            if self.__tableau[i][s] > 0 :
                temp = self.__tableau[i,0]/self.__tableau[i,s]
                if min_ratio > temp:
                    min_ratio = temp
                    r = i
        # problem infeasible if this is true
        if r == 0:
            print("Error : Problem is infeasible, minimum ratio test failed")
            raise Exception()
        return (r,s)
    
    def __pivot(self, r : int, s : int) -> None:
        """
        Induces pivoting operations on the (r,s) entry of the tableau,
        this causes the rth row to be divide by tableau[r,s] and then all entries
        on the sth column to become 0 other than where the row is r.
        """
        m : int = self.__tableau.shape[0]
        self.__tableau[r] /= self.__tableau[r,s]
        for i in range(m):
            if i != r:
                self.__tableau[i] -= self.__tableau[r]*self.__tableau[i,s]

    def __solve_auxillary_problem(self) -> None: 
        """
        Solves the auxillary problem using the phase 1 simplex method
        """
        m : int = self.__tableau.shape[0]
        n : int = self.__tableau.shape[1]
        # basis is always the auxillary variables
        basis : np.array[int] = np.arange(n+1, n+m+1)
        while (self.__tableau[0,1:].min() < 0):
            r,s = self.__find_pivot()
            self.__pivot(r,s)
            basis[r] = s # beware of this line for the time being
    
    def solve_program(self) -> None:
        """Induces the solving of the phase 2 simplex problem prescribed by the object"""
        self.__construct_tableau()
        self.__solve_auxillary_problem()

    def get_tableau(self) -> np.array:
        return self.__tableau