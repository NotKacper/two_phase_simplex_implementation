from two_phase_simplex import TwoPhaseSimplex

def main() -> None:
    # test the basic tableau construction of the auxillary problem
    A = [[1,0],[1,1]]
    b = [2,3]
    c = [1,1]
    x = TwoPhaseSimplex(A,b,c)
    x.solve_program()
    A = [[3,2,2],[4,5,4],[2,1,1],[0,1,1]]
    b = [1,2,3,4]
    c = [-1,-2,3]
    y = TwoPhaseSimplex(A,b,c)
    y.solve_program()
    pass

if __name__ == '__main__':
    main()