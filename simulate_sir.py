import numpy as np
import pandas as pd

def gillespie_SIR(S0, I0, R0, beta, gamma, max_time=100, dt=1.0):
    """
    Runs the Gillespie algorithm for the SIR model.

    Parameters:
    - S0, I0, R0: Initial susceptible, infected, and recovered counts
    - beta: Infection rate
    - gamma: Recovery rate
    - max_time: Time until which the simulation runs
    - dt: Interval at which to record data

    Returns:
    - DataFrame containing time series of S, I, R counts
    """
    S, I, R = S0, I0, R0
    t = 0.0
    t_out = [0.0]
    S_out, I_out, R_out = [S], [I], [R]
    next_output = dt

    while t < max_time and I > 0:
        rate_infect = beta * S * I
        rate_recover = gamma * I
        rate_total = rate_infect + rate_recover

        # Avoid division by zero
        if rate_total == 0:
            break

        # Time until next event
        t += np.random.exponential(1 / rate_total)

        # Determine event type
        if np.random.rand() < rate_infect / rate_total:
            S = max(S - 1, 0)
            I += 1
        else:
            I = max(I - 1, 0)
            R += 1

        # Record state at each dt step
        while next_output <= t:
            t_out.append(next_output)
            S_out.append(S)
            I_out.append(I)
            R_out.append(R)
            next_output += dt

    return pd.DataFrame({
        'time': t_out,
        'S': S_out,
        'I': I_out,
        'R': R_out
    })
