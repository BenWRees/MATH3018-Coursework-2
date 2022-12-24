import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt

def Lagrangian(t,y,dy,alpha=5,beta=5) :
    """
    Description:
        Represents the Lagrangian from coursework
    :param t: a float or integer that represents the time
    :param y: a float or intege which represents the output of y at time t
    :param dy: a float or intege which represnts the derivative of y at time t
    :param alpha: a float or intege that is a constant in the lagrangian
    :param beta: a float or intege that is a constant in the lagrangian
    :return: a float or intege representing the output of the lagrangian at point (t,y,dy) with constants alpha and beta
    """
    t_coords_errors(t,(y,dy,alpha,beta))
    return alpha*dy**2 + beta*((t**2)-1)*dy**3 - y

def t_coords_errors(t,coords) :
    """
    Description:
        checks the errors for Lagrangian and derivatives t,y,dy,alpha and beta values
    :param t: a float for a functions t value
    :param coords: a tuple of length 3 that are the values for (y,dy,alpha,beta)
    :return: None
    """
    assert isinstance(t, (int,float,list, np.ndarray)), "The parameter t is not an integer, list or float"
    assert t != np.NAN, "t cannot be NAN"
    assert len(coords) == 4, "The parameter coords is not a list of length 4"

    y,dy,alpha,beta = coords
    assert isinstance(y, (int, float)), "y is not an integer or float"
    assert not np.isnan(y), "y cannot be NAN"
    assert isinstance(dy, (int, float)), "dy is not an integer or float"
    assert not np.isnan(dy), "dy cannot be NAN"
    assert isinstance(alpha, (int, float)), "alpha is not an integer or float"
    assert not np.isnan(alpha), "alpha cannot be NAN"
    assert isinstance(beta, (int, float)), "beta is not an integer or float"
    assert not np.isnan(beta), "beta cannot be NAN"
    return None

def h_errors(h):
    """
    Description:
        checks the errors for Lagrangian and derivatives t,y,dy,alpha and beta values
    :param h: a tuple of length 3 that are the values for (t_space,y_space,dy_space)
    :return: None
    """
    assert h != 0, "h cannot equal 0, as a divide by zero will occur"
    assert isinstance(h, (int, float)), "The parameter h is not a float or integer"
    assert not np.isnan(h), "h cannot be NAN"
    return None

#Question 1
def part_diff_L_wrt_y(L, t, coords, h):
    """
     Description:
         Differentiates a Lagrange function with respect to y
    :param L: callable function that represents the lagrangian being differentiated
    :param t: an integer or float which is the point at which lagrangian is being differentiated at
    :param coords: tuple of size 4 that are the coordinates of L. These are (y,dy,alpha,beta)
    :param h: an integer or float representing the grid spacing for the domain of L, represented as (t,y,dy)
    :return: a float that is the value of the first order partial derivative of L with respect to y at some point
            (t,y,dy) in the lagrangian
    """
    assert callable(L), "The parameter L in part_diff_L_wrt_y is not a callable function"

    t_coords_errors(t, coords)
    y, dy,alpha,beta = coords

    h_errors(h)

    return (L(t, y + h, dy, alpha, beta) - L(t, y - h, dy, alpha, beta)) / (2 * h)

def part_diff_L_wrt_t_y_prime(L, t, coords, h):
    """
     Description:
         Differentiates a Lagrange function with respect to y prime and t
    :param L: callable function that represents the lagrangian being differentiated
    :param t: an integer or float which is the point at which lagrangian is being differentiated at
    :param coords: tuple of size 4 that are the coordinates of L. These are (y,dy,alpha,beta)
    :param h: an integer or float representing the grid spacing for the domain of L, represented as (t,y,dy)
    :return: a float that is the value of the second order partial derivative of L with respect to t and dy at some
            point (t,y,dy) in the lagrangian
    """
    assert callable(L), "The parameter L in part_diff_L_wrt_t_y_prime is not a callable function"

    t_coords_errors(t, coords)
    y, dy,alpha,beta = coords

    h_errors(h)

    return (L(t + h, y, dy + h, alpha, beta) - L(t + h, y, dy - h, alpha, beta) - L(t - h, y, dy + h, alpha, beta) +
            L(t - h, y, dy - h, alpha, beta)) / (4 * h**2)


def part_diff_L_wrt_y_y_prime(L, t, coords, h):
    """
     Description:
         Differentiates a Lagrange function with respect to y and y prime
    :param L: callable function that represents the lagrangian being differentiated
    :param t: an integer or float which is the point at which lagrangian is being differentiated at
    :param coords: tuple of size 4 that are the coordinates of L. These are (y,dy,alpha,beta)
    :param h: an integer or float representing the grid spacing for the domain of L, represented as (t,y,dy)
    :return: a float that is the value of the second order partial derivative of L with respect to y and dy at
            some point (t,y,dy) in the lagrangian
    """
    assert callable(L), "The parameter L in part_diff_L_wrt_y_y_prime is not a callable function"

    t_coords_errors(t, coords)
    y, dy,alpha,beta = coords

    h_errors(h)

    return (L(t, y + h, dy + h, alpha, beta) - L(t, y + h, dy - h, alpha, beta)
            - L(t, y - h, dy + h, alpha, beta) +
            L(t, y - h, dy - h, alpha, beta))/(4 * h**2)

def part_diff_L_wrt_y_prime_y_prime(L, t, coords, h):
    """
     Description:
         Differentiates a Lagrange function with respect to y prime and y prime
    :param L: callable function that represents the lagrangian being differentiated
    :param t: an integer or float which is the point at which lagrangian is being differentiated at
    :param coords: tuple of size 4 that are the coordinates of L. These are (y,dy,alpha,beta)
    :param h: an integer or float representing the grid spacing for the domain of L, represented as (t,y,dy)
    :return: a float that is the value of the second order partial derivative of L with respect to dy twice
            at some point (t,y,dy) in the lagrangian
    """
    assert callable(L), "The parameter L in part_diff_L_wrt_y_prime_y_prime is not a callable function"
    t_coords_errors(t, coords)
    y, dy, alpha, beta = coords

    h_errors(h)

    return (L(t, y, dy + h, alpha, beta) - (2 * L(t, y, dy, alpha, beta)) + L(t, y, dy - h, alpha, beta)) / (h ** 2)


def IVP_transform(L,h,alpha,beta) :
    """
    Description:
        the transformation of some lagrangian L into an IVP of F(t,y,y') = y''
    :param L: a callable function the lagrangian that is plugged into the coefficients of the BVP
    :param h: An integer or float that represents the grid spacing of the domain of lagrangian
    :param alpha: An integer or float that represents the constant of alpha in the lagrangian
    :param beta: An integer or float that represents the constant of alpha in the lagrangian
    :return: a callable function that takes values of q and t, where t is a value for time and q is a pair for the
            values of y and y'
    """
    #error handling
    assert callable(L), "parameter L in the function IVP_transform is not a callable function"
    assert isinstance(h, (int, float)), "parameter h in the function IVP_transform is not a float or an integer"
    assert not np.isnan(h), "h cannot be NAN"
    assert isinstance(alpha, (int,float)), "parameter alpha in function IVP_transform is not a float or an integer"
    assert not np.isnan(alpha), "alpha cannot be NAN"
    assert isinstance(beta, (int,float)), "parameter beta in function IVP_transform is not a float or an integer"
    assert not np.isnan(beta), "beta cannot be NAN"
    def curry_IVP_transform(q,t) :
        """
        Description:
            the inner function that curries the function so it can be used as a callable function in the scipy function
            odeint
        :param q: a tuple that is a pair of of values of y and y'
        :param t: a value of int that represents the value of t in the lagrangian
        :return: a list of length 2 which contains the value for dy and the value for F(t,y,y')
        """
        assert len(q) == 2, "parameter q in curry_IVP_transform is a list of length 2"
        assert isinstance(t, (int,float)),"Parameter t in curry_IVP_transform is not a float or Int"
        assert not np.isnan(t), "parameter t in curry_IVP_transform cannot be NAN"
        y,dy = q
        F = np.zeros(2)
        F[0] = dy
        try :
            F[1] = (part_diff_L_wrt_y(L,t,(y,dy,alpha,beta),h) - part_diff_L_wrt_t_y_prime(L,t,(y,dy,alpha,beta),h) -
            (part_diff_L_wrt_y_y_prime(L,t,(y,dy,alpha,beta),h)*F[0]))/part_diff_L_wrt_y_prime_y_prime(L,t,(y,dy,alpha,beta),h)
        except ZeroDivisionError :
            #error handling to prevent a divide by zero error
            print("divide by 0 occurred, shifting h")
            IVP_transform(L, (2*h), alpha, beta)

        return F
    return curry_IVP_transform

def shooting_method(L, A, B, a, b, alpha=5,beta=5, tolerance=0.1, error_handling=True) :
    """
    Description:
        Uses the shooting method to calculate the BVP, posed in the coursework,
        where the IVP is y'' = F(t,y,y') and y(a)=A and y(b)=B with a <= t <= b
        Some of the restrictions on the algorithm is that the signature of the lagrangian function, must be with the
        parameters t,y,dy,alpha=5,beta=5. The shooting method solves y(t) only for the Euler-Lagrange defined in
        the coursework.
    :param L: a callable function that represents the lagrangian passed into the BVP calculating the coefficients
    :param A: An integer or float that represents the left hand Boundary condition
    :param B: An integer or float that represents the right hand boundary condition
    :param a: An integer or float that represents the minimum of the range of t values
    :param b: An integer or float that represents the maximum of the range of t values
    :param alpha: An integer or float that represents a constant found in the lagrangian in the coursework. Defaults to 5
    :param beta: An integer or float that represents a constant found in the lagrangian in the coursework. Defaults to 5
    :param tolerance: A non-negative integer or float that represents The allowable error of the zero value. defaults to 0.1
    :return: a tuple of all the t values in the domain of y(t), and all values of y(t) in the range of [a,b]
    """

    #error handling for types
    if error_handling == True :
        assert callable(L), "Parameter L in shooting_method is not a callable function"
        assert isinstance(A, (int, float)), "Parameter A in shooting_method is not a float or Int"
        assert not np.isnan(A), "A cannot be NAN"
        assert isinstance(B, (int, float)), "Parameter B in shooting_method is not a float "
        assert not np.isnan(B), "B cannot be NAN"
        assert isinstance(a, (int, float)), "Parameter a in shooting_method is not a float "
        assert not np.isnan(a), "a cannot be NAN"
        assert isinstance(b, (int, float)), "Parameter b in shooting_method is not a float "
        assert not np.isnan(b), "b cannot be NAN"
        assert isinstance(alpha, (int, float)), "Parameter alpha in shooting_method is not a float "
        assert not np.isnan(alpha), "alpha cannot be NAN"
        assert isinstance(beta, (int, float)), "Parameter beta in shooting_method is not a float "
        assert not np.isnan(beta), "beta cannot be NAN"
        assert isinstance(tolerance, (int, float)), "Parameter tolerance in shooting_method is not a float "
        assert not np.isnan(tolerance), "tolerance cannot be NAN"
        assert a <= b, "parameter a is not less than parameter b"
        assert tolerance >= 0, "tolerance cannot be less than 0"

    #provide the domain of t in [a,b], as well as its grid spacing
    t,dt = np.linspace(a,b, retstep=True)

    def phi_function(z_value) :
        """
        Description:
            Inner function that is used in the optimisation that calculates phi(z) = y(b,z) - B
        :param guess: the float or integer for z at step n-1, that is being plugged into the function
        :return: a value for the most recent guess of phi(z) at step n
        """
        # The initial conditions from the guess and the boundary conditions
        init_y = [A, z_value]
        y = integrate.odeint(IVP_transform(L,dt,alpha,beta), init_y, t)
        assert len(y) != 0, "the output of y cannot be an empty list "

        # Compute the error at the final point - list of all the values of y across the domain t - B
        return y[-1,0] - B

    #find the root of y(b,z) - B, giving us a value of z
    z = optimize.newton(phi_function,(b-a)/2, tol=tolerance)

    # The initial conditions from the boundary, and the value from finding the root of phi(z)
    init_y = [A, z]

    # Solve the IVP
    y = integrate.odeint(IVP_transform(L,dt,alpha,beta), init_y, t)
    return t, y[:,0]
    
"""
JUSTIFICATION OF USING THE SHOOTING METHOD WITH A BLACK BOX ROOT FINDER:
The Shooting Method is an easy to implement algorithm, especially in comparison to the finite differences and 
collocative algorithms, (while maybe not being as so robust), it is also fast and efficient, able to quickly converge 
on the correct solution. In particular for nonlinear differential equations, like the case we are dealing with, shooting
methods have certain advantages for the problem solver. The shooting methods are quite general and are applicable to a
wide variety of differential equations. We do see some problems arise with the shooting method generally, such as they
sometimes fail to converge for problems which are sensitive to the initial conditions, however we see with the case of
the BVP we deal with in the coursework, that this problem does not occur with the boundary conditions we have imposed
between 0 ≤ t ≤ 1. Another serious shortcoming of shooting becomes apparent when the differential equations are so
unstable that they "blow up" before the initial value problem can be completely integrated. We can attempt to explore
if these are really issues for our bvp through testing, by demonstrating the robustness of the algorithm under different
constraints, we can show that the shooting method is robust enough, as well that the IVP calculated from the well-posed
BVP is not unstable. 

While experimenting with methods with a more white box root finding algorithm, it was observable that changes in the work
on the algorithm, tolerances would greatly affect the result moreso than a black box root finding method. 

Essentially, My choice of using the shooting method with a black box root finder, is that it is one of the least 
computationally taxing and fast algorithms, while still affording great accuracy; the problems that usually arise with 
the shooting algorithm are somewhat negated by the choice of bvp problem faced in the coursework. Also the shooting 
method is able to handle non-linear BVP, like the one in the coursework, in an efficient and fast manner.
"""


#QUESTION 2
def exact_sol_from_cw(t,A,B,a,b,alpha) :
    """
    Description:
        the analytic solution for L(t,y,y') = alpha*y'' - y, which gives us the second order ODE 2*alpha*y'' + 1 = 0
        which has the solution y(t) = -t**2/(4*alpha) + C1*t + C2
    :param t: a float or integer that represents the time t of the lagrangian
    :param A: a float or integer that is the right boundary condition
    :param B: a float or integer that is the left boundary condition
    :param a: a float or integer that is the minimum of the domain of t in the bvp
    :param b: a float or integer that is the maximum of the domain of t in the bvp
    :param alpha: a float or integer that represents a constant in the lagrangian
    :return: a float or integer that is of the mapping of the lagrangian at time t
    """
    #error checking
    assert isinstance(t, (int, float, np.ndarray)), "parameter t in exact_sol_from_cw is not a float,list or int"
    assert isinstance(A, (int, float)), "parameter A in exact_sol_from_cw is not a float or int"
    assert not np.isnan(A), "A cannot be NAN"
    assert isinstance(B, (int, float)), "parameter B in exact_sol_from_cw is not a float or int"
    assert not np.isnan(B), "B cannot be NAN"
    assert isinstance(a, (int, float)), "parameter a in exact_sol_from_cw is not a float or int"
    assert not np.isnan(a), "a cannot be NAN"
    assert isinstance(b, (int, float)), "parameter b in exact_sol_from_cw is not a float or int"
    assert not np.isnan(b), "b cannot be NAN"
    assert isinstance(alpha, (int, float)), "parameter alpha in exact_sol_from_cw is not a float or int"
    assert not np.isnan(alpha), "alpha cannot be NAN"
    assert a <= b, "the minimum of the domain of t (a) must be less than or equal to the maximum of the domain (b)"


    c1 = (B-A -((a**2 - b**2)/(4*alpha)))/(b-a)
    c2 = (A*b - B*a -((a*b**2 - b*a**2)/(4*alpha)))/(b-a)
    return (-t**2)/(4*alpha) + c1*t + c2


#TESTS
"""
EXPLANATION OF CHOICE OF TEST:
In this test we are checking the accuracy of the shooting method under different boundary conditions. Thus testing its 
robustness.
We used the same lagrangian in each case. Having analytically solved the lagrangian, so that we have an exact solution. 
We graphically show each case with the exact solution and the numerical solution in the same graph. Furthermore, 
the subtitle displayed, will either display true or false, based on if the numerical solutions are approximately equal 
to the exact solutions - displaying true if they are equal. This checks the overall robustness of our algorithm, 
under essentially different Lagrangians in the BVP, as well as the general accuracy of the algorithm against a known 
analytic solution
"""
A = [1,2,0.8,3,4,5]
B = [0.9,1.5,0,0,1,2]
a = [0,3,0.5,1,2,4]
b = [1,10,3,2.2,4,6]
alpha = [5,10,2,3.4,1.2,6]
y_approx = []
y_exact = []
t = []
check_validity = []
fig,axs = plt.subplots(2,3)
fig.suptitle("Testing the Algorithm against a linear ODE against the exact solution under different B.C.")
fig.tight_layout()
#plots each subplot with unique values for A,B,a,b and alpha against the exact solution
for i in range(len(B)) :
    t_cw, y_cw = shooting_method(Lagrangian,A[i],B[i],a[i],b[i],alpha[i],0)
    y_cw_exact = exact_sol_from_cw(t_cw,A[i],B[i],a[i],b[i],alpha[i])
    check_validity.append(np.allclose(y_cw_exact, y_cw))
    #case for putting subplots on the first row
    if i < 3 :
        cw_approx, = axs[0][i-3].plot(t_cw,y_cw)
        cw_exact, = axs[0][i-3].plot(t_cw,y_cw_exact, '--')
        axs[0][i-3].set_title(r"Case where: y({0})={1}, y({2})={3}, $\alpha$={4}".format(a[i],A[i],b[i],B[i],alpha[i]),fontsize=9)
        axs[0][i - 3].set_xlabel("value of t")
        axs[0][i-3].set_ylabel("value of y(t)")
    #subplots on the second row
    else :
        cw_approx, = axs[1][i-3].plot(t_cw,y_cw)
        cw_exact, = axs[1][i-3].plot(t_cw,y_cw_exact, '--')
        axs[1][i - 3].set_title(r"Case where: y({0})={1}, y({2})={3}, $\alpha$={4}".format(a[i], A[i], b[i], B[i], alpha[i]),fontsize=9)
        axs[1][i-3].set_ylabel("value of y(t)")
        axs[1][i-3].set_xlabel("value of t")

fig.legend((cw_approx, cw_exact), ("approximate solution", "exact solution"), loc="upper right")
fig.set_figheight(8)
fig.set_figwidth(15)
fig.text(x=0.28,y=0.93,s="Result of Test for all Numerical cases being approximate to the exact solution: {0}".format(np.all(check_validity)))
fig.show()
plt.show()

#check convergence against tolerance
"""
EXPLANATION OF CHOICE OF TEST:
We do this test to check to see if our shooting method improves with accuracy as we reduce the tolerance of our 
guesses for the value of z. This is useful to test the accuracy of our method, especially since we are testing the
numerical solution against the exact solution (instead of the numerical solution against some approximately exact 
solution, that was calculated using the algorithm with very low tolerance - which would assume the algorithm was accurate)
If we happen to see the error between the numerical solution and exact solution approach 0 as our tolerance approaches 0
this would imply that our algorithm is able to caulcuate a numerical solution that is increasingly accurate, and will 
converge to the exact solution as the overall accuracy of the algorithm is modified.

We see from this graph, and its line of best fit, that as the tolerance approaches 0, as does the absolute error between
the numerical solution and the exact solution. This would imply that as we decrease the tolerance given to calculating 
our value of z in the shooting method, we get an increasingly more accurate answer, which would imply that varying the 
tolerance improves the accuracy of our algorithm.
"""
#calculate set of tolerances as []
tolerances = np.reciprocal(list(map((lambda x: 10**x),np.array(np.arange(1.0,10.0,1.0)))))
tolerances = np.insert(tolerances,0,1.) #added to show some variation in the values of the error
errors = []

#create an error value at each tolerance point, by taking the 2-norm of the exact solution - numerical solution
for tolerance in tolerances :
    t,y = shooting_method(Lagrangian,1,0.9,0,1,5,0,tolerance=tolerance)
    exact_y = exact_sol_from_cw(t,1,0.9,0,1,5)
    errors.append(np.linalg.norm(exact_y - y, 2))

#plot a line of best fit to demonstrate convergence
line_of_best_fit_gradient, line_of_best_fit_intercept = np.polyfit(tolerances, errors, 1)

plt.plot(tolerances,errors, '--')
plt.plot(tolerances, line_of_best_fit_gradient*tolerances + line_of_best_fit_intercept)
plt.xlabel("tolerances")
plt.ylabel("|Error|")
plt.title("Evidence that the Shooting method improves with \n accuracy as the tolerance decreases")
plt.legend(('errors', "Best fit line"))
plt.show()

#do some unit tests through some assertions in the algorithms and test if these assertions hold
"""
EXPLANATION OF CHOICE OF TEST:
These unit tests, will test the edge cases of the shooting method algorithm, especially the implementation and general
logic of it. We test how it handles incorrect types, as well as how it handles certain values being entered that
the algorithm cannot handle, such as the tolerance being non-positive, or the minimum of the domain of t being 
greater than the maximum of t. 
"""
def shooting_method_tests() :
    """
    Description:
        tests if the shooting method holds errors correctly, checks if each parameter correctly handles incorrect
        types, checks if numerical parameters handle being np.NAN, checks if the shooting method handles parameter
        a being less than parameter b and checks if the shooting method handles the tolerance being less than or
        equal to 0
    :return: a list of tuples telling us if each parameter passed it's type checking test
    """
    test_passed = []
    # TEST FOR CHECKING IF PARAMETER L IS A CALLABLE FUNCTION
    try :
        shooting_method(3,2,0.9,0,1)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due to the parameter L not being callable")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER A IS OF TYPE INT OR FLOAT
    try :
        shooting_method(Lagrangian,'a',0.9,0,1)
        test_passed.append(False)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due to parameter B not being a float or int")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER a IS OF TYPE INT OR FLOAT
    try:
        shooting_method(Lagrangian,1,0.9,'a',1)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due to parameter a not being a float or int")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER b IS OF TYPE INT OR FLOAT
    try:
        shooting_method(Lagrangian,1,0.9,0,'a')
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due to parameter b not being a float or int")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER ALPHA IS OF TYPE INT OR FLOAT
    try:
        shooting_method(Lagrangian ,1,0.9,0,1,'a')
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due to parameter alpha not being a float or int")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER BETA IS OF TYPE INT OR FLOAT
    try:
        shooting_method(Lagrangian,1,0.9,1,0,beta='a')
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due to parameter beta not being a float or int")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER TOLERANCE IS OF TYPE INT OR FLOAT
    try:
        shooting_method(Lagrangian,1,0.9,0,1,tolerance='a')
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due the tolerance not being set to a float or int")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER A IS NOT EQUAL TO NAN
    try:
        shooting_method(Lagrangian,np.NAN,0.9,0,1)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due the A being NAN")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER B IS NOT EQUAL TO NAN
    try:
        shooting_method(Lagrangian,1,np.NAN,0,1)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due the B being NAN")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER a IS NOT EQUAL TO NAN
    try:
        shooting_method(Lagrangian,1,0.9,np.NAN,1)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due the a being NAN")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER b IS NOT EQUAL TO NAN
    try:
        shooting_method(Lagrangian,1,0.9,0,np.NAN)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due the b being NAN")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER ALPHA IS NOT EQUAL TO NAN
    try:
        shooting_method(Lagrangian,1,0.9,0,1,np.NAN)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due the alpha being NAN")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER BETA IS NOT EQUAL TO NAN
    try:
        shooting_method(Lagrangian,1,0.9,0,1,beta=np.NAN)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due the beta being NAN")
        test_passed.append(True)
    # TEST FOR CHECKING IF PARAMETER TOLERANCE IS NOT EQUAL TO NAN
    try:
        shooting_method(Lagrangian,1,0.9,0,1,tolerance=np.NAN)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due the tolerance being NAN")
        test_passed.append(True)
    # TEST FOR CHECKING IF a<=b THROWS AN ERROR
    try:
        shooting_method(Lagrangian,1,0.9,1,0)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due a<=b")
        test_passed.append(True)
    # TEST FOR CHECKING IF THE TOLERANCE =< 0 THROWS AN ERROR
    try:
        shooting_method(Lagrangian,1,0.9,0,1, tolerance=-1)
        test_passed.append(False)
    except AssertionError:
        print("Shooting Method correctly throws an assertion error due tolerance < 0")
        test_passed.append(True)
    #testing if any partial derivative will throw error for h=0
    try :
        part_diff_L_wrt_y(Lagrangian,0,(0,1,2,3),0)
        test_passed.append(False)
    except AssertionError:
        print("partial derivative correctly throws an assertion error due to h being set to 0")
        test_passed.append(True)
    #testing if any partial derivative will throw error for length of coordinates != 4
    try :
        part_diff_L_wrt_y(Lagrangian,0,(0,2,3),2)
        test_passed.append(False)
    except AssertionError:
        print("partial derivative correctly throws an assertion error due to the length of the coords parameter not "
              "being equal to 4")
        test_passed.append(True)

    #EACH TEST RETURNS TRUE IF THE ASSERTION ERROR WAS THROWN AS EXPECTED, AND FALSE IF THE TEST CARRIES ON
    #IGNORING THE INCORRECT ARGUMENT BEING PASSED
    test = ["L type check","A type check","B type check","a type check","b type check","alpha type check",
            "beta type check","tolerance type check", "A NAN", "B NAN", "a NAN", "b NAN", "alpha NAN",
            "beta NAN", "tolerance NAN", "a <= b", "tolerance < 0","h being set to 0 in partial derivatives",
            "length of the coordinates parameter in the partial derivatives is not equal 4"]
    return list(zip(test,test_passed))

print("tests on shooting_method and if they passed: ", shooting_method_tests())


#QUESTION 3
# Graph of alpha=beta=5 with 3 different tolerances
t_1_a, y_1_a = shooting_method(Lagrangian, 1, 0.9, 0, 1, 5, 5, 0.01)
t_1_b, y_1_b = shooting_method(Lagrangian, 1, 0.9, 0, 1, 5, 5, 0.001)
t_1_c, y_1_c = shooting_method(Lagrangian, 1, 0.9, 0, 1, 5, 5, 0.0001)

tol_1, = plt.plot(t_1_a, y_1_a)
tol_2, = plt.plot(t_1_b, y_1_b, '-.')
tol_3, = plt.plot(t_1_c, y_1_c, '--')
plt.ylabel("price curve, y(t)")
plt.xlabel("value of t")
plt.title(r'Price Curve where $\alpha$ = 5 and  $\beta$ = 5')
plt.legend((tol_1, tol_2, tol_3), ('tol=0.01', 'tol=0.001', 'tol=0.0001'))
plt.show()

# Graph of alpha=1.75, beta=5 with 3 different tolerances
t_2_a, y_2_a = shooting_method(Lagrangian, 1, 0.9, 0, 1, 1.75, 5, 0.01)
t_2_b, y_2_b = shooting_method(Lagrangian, 1, 0.9, 0, 1, 1.75, 5, 0.001)
t_2_c, y_2_c = shooting_method(Lagrangian, 1, 0.9, 0, 1, 1.75, 5, 0.0001)

tol_1, = plt.plot(t_2_a, y_2_a, '--')
tol_2, = plt.plot(t_2_b, y_2_b, '-.')
tol_3, = plt.plot(t_2_c, y_2_c, '--')
plt.ylabel("price curve, y(t)")
plt.xlabel("value of t")
plt.title(r'Price Curve where $\alpha$ = 1.75 and $\beta$ = 5')
plt.legend((tol_1, tol_2, tol_3), ('tol=0.01', 'tol=0.001', 'tol=0.0001'))
plt.show()

"""
STATE AND JUSTIFY THE MATHEMATICAL PROBLEM HAS BEEN POSED CORRECTLY TO SOLVE THE ORIGINAL PROBLEM:
Based on the two pricing models produced for the cases of alpha=beta=5 and alpha=1.75, beta=5. One could reasonably argue
this modelling could help the company who is forced to reduce their prices, decide the most appropriate course of action,
based on numerous cases for alpha and beta, they could also use the mathematical model to break down a more minute 
analysis of how they maximise their profit across the year, for as long as possible. 

More concretely, just by observing the graphs, the rate of decrease in the charge, y(t) is much slower when alpha=1.75,
beta=5 suggesting when alpha=1.75, beta=5 is a better pricing model to use than the case when alpha=beta=5. A member of 
the company could then easily use these types of graphs to make a more informed decision on how better to maximise their 
profits, by focusing on a case in which y(t) manages to remain higher for longer throughout the year. This all suggests 
that the mathematical problem for maximising y(t) has helped solve the original problem of making sure a company is able
to retain profits for as long as possible by being able to model their maximum profits across that year under different
cases imposed by the penalty, through alpha and beta.     	

"""

# show that when alpha=1.75, beta=5 then the result is converging
exact_t, exact_y = shooting_method(Lagrangian, 1, 0.9, 0, 1, 1.75, 5, 1e-14)

# calculate absolute error
# array of all the tolerance points
tolerances = np.reciprocal(list(map((lambda x: 10 ** x), np.array(np.arange(1.0, 11.0, 1.0)))))
errors = []

for tolerance in tolerances:
    t, y = shooting_method(Lagrangian, 1, 0.9, 0, 1, 1.75, 5, tolerance)
    errors.append(np.linalg.norm(exact_y - y, 2))

line_of_best_fit_gradient, line_of_best_fit_intercept = np.polyfit(tolerances, errors, 1)
poly = np.polyfit(np.log(tolerances), np.log(errors), 1)

error, = plt.plot(tolerances, errors, 'x')
best_fit, = plt.plot(tolerances, line_of_best_fit_gradient*tolerances + line_of_best_fit_intercept)
#best_fit, = plt.loglog(tolerances, np.exp(poly[1]) * tolerances ** poly[0])
plt.ylabel("computable Error")
plt.xlabel("Tolerances")
plt.title(r"Evidence that the error converges when $\alpha$=1.75, $\beta$=5")
plt.legend((error, best_fit), ('errors', 'Best fit line'))
plt.show()

"""
EVIDENCE THAT THE SOLUTION FOR alpha=1.75, beta=5 IS CONVERGING IN AN APPROPRIATE SENSE:
As we have no known exact solution for the particular BVP we’re working with, with this Lagrange, we are forced to 
approximate our exact solution using our algorithm. We can do this as we’ve see from our tests that the algorithm is 
consistent and robust for a variety and boundary conditions, and we have seen by plotting and the cases for when 
alpha=beta=5 and alpha=1.75 and beta=5 and observing that the results are not divergent that the BVP is stable. 
Thus as we can “approximate” the exact solution, we can then show that as we vary the tolerance imposed on our solution,
we can see that the algorithm will converge to some error. Thus by controlling the work done on the algorithm, which in 
this case is the tolerance, and plotting it against the error for each value of tolerance we have taken, we can observe 
that our algorithm slowly converges to some error value. It is expected that with lower tolerance there will still be 
some error, as we are not comparing to some exact solution, but we can observe graphically, that as the tolerance 
approaches our approximately exact solution, the error slowly decreases, by basic interpolation this would suggest at 
some point our error must hit 0.
"""