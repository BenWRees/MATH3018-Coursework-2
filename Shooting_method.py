import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt

def t_coords_errors(t,coords) :
    """
    Description:
        checks the errors for Lagrangian and derivatives t,y,dy,alpha and beta values
    :param t: a float for a functions t value
    :param coords: a tuple of length 3 that are the values for (y,dy,alpha,beta)
    :return: null
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

def h_errors(h):
    """
    Description:
        checks the errors for Lagrangian and derivatives t,y,dy,alpha and beta values
    :param h: a tuple of length 3 that are the values for (t_space,y_space,dy_space)
    :return: null
    """
    assert h != 0, "h cannot equal 0, as a divide by zero will occur"
    assert isinstance(h, (int, float)), "The parameter h is not a float or integer"
    assert not np.isnan(h), "h cannot be NAN"

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


    #t_space, y_space, dy_space = h
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

    #t_space, y_space, dy_space = h
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

    #t_space, y_space, dy_space = h
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

    #t_space, y_space, dy_space = h
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

def optimise_root_finding(phi_func,a,b,dy,tolerance) :
    """
    Description:
        Optimises the
    :param phi_func:
    :param a:
    :param b:
    :param dy:
    :param tolerance:
    :return:
    """
    z = 0
    flag = True
    while flag :
        try :
            z = optimize.brentq(phi_func,a,b,xtol=tolerance)
            flag = False
        except ValueError:
            a -= dy
            b += dy
    return z

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

    t,dt = np.linspace(a,b, retstep=True)
    #step 2
    def phi_function(z_value) :
        """
        Description:
            Inner function that is used in the optimisation that calculates phi(z) = y(b,z) - B
        :param guess: the float or integer for z at step n-1, that is being plugged into the function
        :return: a value for the most recent guess of phi(z) at step n
        """
        # The initial conditions from the guess and the boundary conditions
        #y0 is an array pair of y and z
        init_y = [A, z_value]
        y = integrate.odeint(IVP_transform(L,dt,alpha,beta), init_y, t)
        assert len(y) != 0, "the output of y cannot be an empty list "
        # Compute the error at the final point - list of all the values of y across the domain t - B
        return y[-1,0] - B
    #find the root of y(b,z) - B, giving us a value of z
    #z = optimize.newton(phi_function,(b-a)/2, tol=tolerance)
    z = optimise_root_finding(phi_function,a,b,dt,tolerance)
    #print(z)


    # The initial conditions from the boundary,
    # and the now "correct" value from the root-find
    init_y = [A, z]
    # Solve the IVP
    y = integrate.odeint(IVP_transform(L,dt,alpha,beta), init_y, t)
    return t, y[:,0]

"""
Reasons for Choosing Shooting Method:
    The Shooting method is a relatively simple and efficient method for solving Boundary Value Problems, it takes 
    advantage of the speed and adaptivity of methods for initial value problems. As the BVP, in the Euler-Lagrange 
    posed in the coursework, is not complicated, we can safely assume that the shooting method will be adequate for 
    providing a numerical solution to the bvp. We see that some of the problems posed usually with the shooting method
    is that it can fail to satisfy boundary conditions, the bvp may have multiple solutions. We see in our testing that
    for the lagrangian in the coursework, that these problems do not occur, and the BVP remains stable under numerous 
    boundary conditions.
    
    Conclusively, The disadvantage of the shooting method is that it is not as robust as finite difference or a 
    collocation method. Some initial value problems with growing modes are inherently unstable even though the 
    BVP itself may be quite well posed and stable, however we observe that the BVP explained in the coursework that the
    issue of growing modes is not a problem with this particular BVP. The disadvantages of the shooting method are 
    negated when dealing with this particular BVP in the coursework.
"""

def f(x) :
    return x**2-1
