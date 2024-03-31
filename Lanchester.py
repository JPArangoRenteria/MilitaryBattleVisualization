#Lanchester Law Modeling For Armies
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

def lotka_volterra(x,y,t,alpha,beta,delta,gamma):
    state = x,y
    dxdt = alpha*x-beta*x*y
    dydt = delta*x*y-gamma*y
    return [dxdt,dydt]
def lanchesterlaw(y,beta,alpha,ca,cb,helmbold,case_type):
    state = beta,alpha
    if len(y) == 2:
        a = y[0]
        b = y[1]
        if case_type == 'linear':
            if helmbold:
                dadt = -beta*cb*b
                dbdt = -alpha*ca*a
                return [dadt,dbdt]
            else: 
                dadt = -beta*b
                dbdt = -alpha*a
                return [dadt,dbdt]
        elif case_type == 'square':
            if helmbold:
                dadt = -beta*cb*b^2
                dbdt = -alpha*ca*a^2
                return [dadt,dbdt]
            else: 
                dadt = -beta*b^2
                dbdt = -alpha*a^2
                return [dadt,dbdt]
        elif case_type == 'logarithmic':
            if helmbold:
                dadt = -beta*cb*math.log(b)
                dbdt = -alpha*ca*math.log(a)
                return [dadt,dbdt]
            else: 
                dadt = -beta*math.log(b)
                dbdt = -alpha*math.log(a)
                return [dadt,dbdt]
        elif case_type == 'exponential':
            if helmbold:
                dadt = -beta*cb*math.exp(b)
                dbdt = -alpha*ca*math.exp(a)
                return [dadt,dbdt]
            else: 
                dadt = -beta*math.exp(b)
                dbdt = -alpha*math.exp(a)
                return [dadt,dbdt]
    else:
        n = len(y) // 2  # Number of force types
        A = y[:n]  # Strengths of forces A
        B = y[n:]  # Strengths of forces B
        
        dAdt = np.zeros(n)
        dBdt = np.zeros(n)
        
        for i in range(n):
            dAdt[i] = -np.sum(beta[i,:] * cb * b)
            dBdt[i] = -np.sum(alpha[i,:] * ca * a)
        
        return np.concatenate((dAdt, dBdt))


def solve_ode(ode_func, initial_conditions, t):
    """
    Solve an ordinary differential equation 

    Parameters:
        ode_func : function
            Function that defines the system of ODEs. It should have the signature
            `ode_func(y, t)` where `y` is an array containing the dependent variables
            and `t` is the independent variable.
        initial_conditions : array_like
            Initial values of the dependent variables.
        t : array_like
            Array of time points for which to solve the ODE.

    Returns:
        solution : array_like
            Array containing the solution of the ODE at each time point.
    """
    solution = odeint(ode_func, initial_conditions, t)
    return solution

def simulation_result(choice,parameters,initial_conditions,tn):
    t = np.linspace(0,10,tn)
    if choice == 1:
        beta,alpha,a,b,ca,cb,helmbold,case_type= parameters
        ode_f = lanchesterlaw(y,beta,alpha,ca,cb,helmbold,case_type)
        solution = solve_ode(ode_f, initial_conditions, t)
    elif choice == 2:
        x,y,t,alpha,beta,delta,gamma = parameters
        ode_f = lotka_volterra(x,y,t,alpha,beta,delta,gamma)
        solution = solve_ode(ode_f, initial_conditions, t)

def compare_with_historical_data(simulation_results, historical_data):
    # Extract relevant metrics from simulation results and historical data
    sim_casualties = simulation_results[:, 1]  # Assuming casualties are in the second column of simulation results
    hist_casualties = historical_data['casualties']  # Assuming historical data has a 'casualties' column

    # Plot simulation results and historical data
    plt.plot(simulation_results[:, 0], sim_casualties, label='Simulation Results')
    plt.plot(historical_data['time'], hist_casualties, label='Historical Data')
    plt.xlabel('Time')
    plt.ylabel('Casualties')
    plt.title('Comparison with Historical Data')
    plt.legend()
    plt.show()

def compare_with_historical_data(simulation_results, historical_data):
    # Extract relevant metrics from simulation results and historical data
    sim_casualties = simulation_results[:, 1]  # Assuming casualties are in the second column of simulation results
    hist_casualties = historical_data['casualties']  # Assuming historical data has a 'casualties' column

    # Calculate a score to measure the similarity between simulation and historical data
    score = np.sum(np.abs(sim_casualties - hist_casualties))

    return score

def optimize_parameters(parameters, initial_conditions, tn, historical_data, tolerance=1e-6, max_iterations=100):
    best_parameters = parameters
    best_score = float('inf')

    for _ in range(max_iterations):
        simulation_results = simulation_result(*best_parameters, initial_conditions, tn)
        score = compare_with_historical_data(simulation_results, historical_data)

        if score < best_score:
            best_score = score
            continue
        else:
            # Adjust parameters randomly within a certain range
            # You can implement a more sophisticated optimization algorithm here
            perturbed_parameters = perturb_parameters(best_parameters)
            new_parameters, new_score = optimize_parameters(perturbed_parameters, initial_conditions, tn, historical_data)

            if new_score < best_score:
                best_parameters = new_parameters
                best_score = new_score

        # Check for convergence
        if best_score < tolerance:
            break

    return best_parameters, best_score

def perturb_parameters(parameters):
    # Perturb the parameters randomly within a certain range
    # You can adjust this based on the characteristics of your parameters
    perturbed_parameters = [param + np.random.uniform(-0.1, 0.1) for param in parameters]
    return perturbed_parameters
    

