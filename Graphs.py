import matplotlib.pyplot as plt
import Shooting_method as solver
import numpy as np

#Graph of alpha=beta=5

t_1_a,y_1_a = solver.shooting_method(solver.Lagrangian,1,0.9,0,1,5,5,0.01)
t_1_b,y_1_b = solver.shooting_method(solver.Lagrangian,1,0.9,0,1,5,5,0.001)
t_1_c,y_1_c = solver.shooting_method(solver.Lagrangian,1,0.9,0,1,5,5,0.0001)

tol_1, = plt.plot(t_1_a,y_1_a)
tol_2, = plt.plot(t_1_b,y_1_b, '-.')
tol_3, = plt.plot(t_1_c,y_1_c, '--')
plt.ylabel("price curve, y(t)")
plt.xlabel("value of t")
plt.title(r'Price Curve where $\alpha$ = 5 and  $\beta$ = 5')
plt.legend((tol_1, tol_2, tol_3), ('tol=0.01', 'tol=0.001', 'tol=0.0001'))
plt.show()

#Graph of alpha=1.75, beta=5
"""
    The Graph with 3 different tolerances for 
"""
t_2_a,y_2_a = solver.shooting_method(solver.Lagrangian,1,0.9,0,1,1.75,5,0.01)
t_2_b,y_2_b = solver.shooting_method(solver.Lagrangian,1,0.9,0,1,1.75,5,0.001)
t_2_c,y_2_c = solver.shooting_method(solver.Lagrangian,1,0.9,0,1,1.75,5,0.0001)

tol_1, = plt.plot(t_2_a,y_2_a, '--')
tol_2, = plt.plot(t_2_b,y_2_b, '-.')
tol_3, = plt.plot(t_2_c,y_2_c, '--')
plt.ylabel("price curve, y(t)")
plt.xlabel("value of t")
plt.title(r'Price Curve where $\alpha$ = 1.75 and $\beta$ = 5')
plt.legend((tol_1, tol_2, tol_3), ('tol=0.01', 'tol=0.001', 'tol=0.0001'))
plt.show()

"""
    State and justify the mathematical problem has been posed correctly to solve the original problem:
    - Argue if the company could politically justify the pricing model. Is the mathematical model a politically
    valiable solution
        
        As we can see from the two graphs where alpha=beta=5 and alpha=1.75 and beta=5, We see that the rate of 
        decrease in the charge, y(t) is much slower when alpha=1.75, beta=5 suggesting it is a better pricing model to 
        use than the case when alpha=beta=5. By adjusting these constants, we can observe the rate of decrease of the 
        charge, and the company can make a more informed decision on how they model their penalty. We also see that 
        with alpha=1.75 and beta=5, that y(t) constantly remains higher than the case for alpha=beta=5 for the entirety
        of the year.
"""

#show that when alpha=1.75, beta=5 then the result is converging
exact_t,exact_y = solver.shooting_method(solver.Lagrangian,1,0.9,0,1,1.75,5,1e-14)

#calculate absolute error
#array of all the tolerance points
tolerances = np.reciprocal(list(map((lambda x: 10**x),np.array(np.arange(1.0,10.0,1.0)))))
errors = []

for tolerance in tolerances :
    t,y = solver.shooting_method(solver.Lagrangian,1,0.9,0,1,1.75,5,tolerance)
    errors.append(np.linalg.norm(exact_y - y, 2))

print(list(zip(tolerances,errors)))
line_of_best_fit_gradient, line_of_best_fit_intercept = np.polyfit(tolerances, errors, 1)
poly = np.polyfit(np.log(tolerances),np.log(errors),1)

error, = plt.loglog(tolerances,errors,'x')
#plt.loglog(tolerances, line_of_best_fit_gradient*tolerances + line_of_best_fit_intercept)
best_fit, = plt.loglog(tolerances, np.exp(poly[1])*tolerances**poly[0])
plt.ylabel("computable Error")
plt.xlabel("Tolerances")
plt.title(r"Evidence that the error converges when $\alpha$=1.75, $\beta$=5")
plt.legend((error,best_fit),('errors', 'Best fit line {0 .3}'.format(poly[1])))
plt.show()

"""
    Explanation of convergence graph:
    - converges to some limit as the tolerances get smaller 
    - suggests that the algorithm is convergent as we improve accuracy
        From the final Graph, we see for the case of alpha=1.75 and beta=5, that as the tolerance gets smaller, so
        does the error between our numerical solution, and our approximately "exact" solution. Exact is in quotation 
        marks in this regard as it is impossible to analytically calculate the Lagrangian given in the coursework and
        so we are forced to give our best approximation to what would be the exact solution by using our algorithm with
        a very low value for tolerance, thus being extremely close to the exact solution. 
        This is acceptable based on the fact that we have previously proved through a battery of tests that are 
        algorithm is robust and accurate at calculating solutions for BVPs. Thus we see error is essentially a calculation
        of how close our numerical solution at any tolerance, is to this approximate exact solution, and by Richardson
        Extrapolation, we can consider this to provide the same information as the error calculated from the exact 
        solution. We can see that as the tolerance value decreases to 0, our numerical solution approaches our exact 
        solution. 
"""


