# Two Phase Simplex Method Algorithm

The general overview of the simplex method implemented in the repo is given in the testing.ipynb notebook (note some tests are included in there for curiosity but do not really serve any purpose). <br> <br>
The two_phase_simplex.py file contains the class TwoPhaseSimplex which solves a given linear programming problem from standard form (i.e. all constraints are equalities and the optimisation goal is minimisation). <br> <br>
## Use
In your program include the statement "from two_phase_simplex import TwoPhaseSimplex" which will allow you to construct a TwoPhaseSimplex object, provide the relevant matrix and vectors then use .solve_program() in order to solve the problem. <br>
This method return True if there exists a solution and False if the problem has no solution. To see the final solution (if one exists) then use .get_solution() which will return a dictionary containing the variables and their corresponding values.
## Why?
This program was written as an exercise in formulating programs from a prescribed algorithm, additionally I will be using it for other repositories which I will link here when I have started working on them. <br>
