from typing import Callable, Sequence, Tuple, List
import numpy as np
import random
from scipy.stats import norm
import pandas as pd
from rl.function_approx import DNNApprox, LinearFunctionApprox, \
    FunctionApprox, DNNSpec, AdamGradient, Weights
from random import randrange
from numpy.polynomial.laguerre import lagval
import copy

TrainingDataType = Tuple[float, float, float, float, float, float, float, float, float, float, float, float]

def fitted_lspi_put_option(
    feature_vals,
    next_feature_vals,
    training_iters: int,
        num_intervals: int
) -> LinearFunctionApprox:

    epsilon: float = 1e-3
    # Index(['Interval', 'CumulativePlayerPointsInterval', 'Home', 'TeamRest',
    #       'OpponentTeamRest', 'CumulativeTeamPointsInterval',
    #       'CumulativeOpponentPointsInterval', 'ScoreMarginInterval',
    #       'ScoreMarginxTimeRemainingInterval',
    #       'ScoreMarginxTimeRemaining2Interval', 'RollingAvgPlayerPoints',
    #       'RollingAverageOpposingTeamAllowedPoints', 'RollingAverageTeamPoints']
    exer: np.ndarray = np.array([max(row[1] - row[10] * (row[0] / num_intervals), 0) for row in feature_vals])
    non_terminal: np.ndarray = np.array([row[0] < num_intervals for row in feature_vals])

    features = list(lambda x: x[i] for i in range(13))

    gamma: float = 1.0
    num_features: int = len(features) # will be hardcoded based on Spencer's code

    wts: np.ndarray = np.zeros(num_features)
    for _ in range(training_iters):
        a_inv: np.ndarray = np.eye(num_features) / epsilon
        b_vec: np.ndarray = np.zeros(num_features)
        cont: np.ndarray = np.dot(next_feature_vals, wts)
        cont_cond: np.ndarray = non_terminal * (cont > exer)
        for i in range(len(feature_vals)):
            phi1: np.ndarray = feature_vals[i]
            phi2: np.ndarray = phi1 - \
                cont_cond[i] * gamma * next_feature_vals[i]
            temp: np.ndarray = a_inv.T.dot(phi2)
            a_inv -= np.outer(a_inv.dot(phi1), temp) / (1 + phi1.dot(temp))
            b_vec += phi1 * (1 - cont_cond[i]) * exer[i] * gamma
        wts = a_inv.dot(b_vec)

    return LinearFunctionApprox.create(
        feature_functions=features,
        weights=Weights.create(wts)
    )


def fitted_dql_put_option(
    expiry: float,
    num_steps: int,
    num_paths: int,
    spot_price: float,
    spot_price_frac: float,
    rate: float,
    vol: float,
    strike: float,
    training_iters: int
) -> DNNApprox[Tuple[float, float]]:

    reg_coeff: float = 1e-2
    neurons: Sequence[int] = [6]

#     features: List[Callable[[Tuple[float, float]], float]] = [
#         lambda t_s: 1.,
#         lambda t_s: t_s[0] / expiry,
#         lambda t_s: t_s[1] / strike,
#         lambda t_s: t_s[0] * t_s[1] / (expiry * strike)
#     ]

    num_laguerre: int = 2
    ident: np.ndarray = np.eye(num_laguerre)
    features: List[Callable[[Tuple[float, float]], float]] = [lambda _: 1.]
    features += [(lambda t_s, i=i: np.exp(-t_s[1] / (2 * strike)) *
                  lagval(t_s[1] / strike, ident[i]))
                 for i in range(num_laguerre)]
    features += [
        lambda t_s: np.cos(-t_s[0] * np.pi / (2 * expiry)),
        lambda t_s: np.log(expiry - t_s[0]) if t_s[0] != expiry else 0.,
        lambda t_s: (t_s[0] / expiry) ** 2
    ]

    ds: DNNSpec = DNNSpec(
        neurons=neurons,
        bias=True,
        hidden_activation=lambda x: np.log(1 + np.exp(-x)),
        hidden_activation_deriv=lambda y: np.exp(-y) - 1,
        output_activation=lambda x: x,
        output_activation_deriv=lambda y: np.ones_like(y)
    )

    fa: DNNApprox[Tuple[float, float]] = DNNApprox.create(
        feature_functions=features,
        dnn_spec=ds,
        adam_gradient=AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        ),
        regularization_coeff=reg_coeff
    )

    dt: float = expiry / num_steps
    gamma: float = np.exp(-rate * dt)
    training_data: Sequence[TrainingDataType] = training_sim_data(
        expiry=expiry,
        num_steps=num_steps,
        num_paths=num_paths,
        spot_price=spot_price,
        spot_price_frac=spot_price_frac,
        rate=rate,
        vol=vol
    )
    for _ in range(training_iters):
        t_ind, s, s1 = training_data[randrange(len(training_data))]
        t = t_ind * dt
        x_val: Tuple[float, float] = (t, s)
        val: float = max(strike - s1, 0)
        if t_ind < num_steps - 1:
            val = max(val, fa.evaluate([(t + dt, s1)])[0])
        y_val: float = gamma * val
        fa = fa.update([(x_val, y_val)])
        # for w in fa.weights:
        #     pprint(w.weights)
    return fa


if __name__ == '__main__':

    random.seed(100)
    np.random.seed(100)

    # read in data from nba csv file
    df = pd.read_csv('NBA_PBP_2018-19_processed_points.csv')
    # sort by Player, GameId, and Interval
    df = df.sort_values(by=['Player', 'GameId', 'Interval'], ascending=True)
    df = df[df['RollingAvgPlayerPoints'] > 15]
    names = df['Player']
    # drop date, player, gameid
    df = df[['Interval', 'CumulativePlayerPointsInterval', 'Home', 'TeamRest',
           'OpponentTeamRest', 'CumulativeTeamPointsInterval',
           'CumulativeOpponentPointsInterval', 'ScoreMarginInterval',
           'ScoreMarginxTimeRemainingInterval',
           'ScoreMarginxTimeRemaining2Interval', 'RollingAvgPlayerPoints',
           'RollingAverageOpposingTeamAllowedPoints', 'RollingAverageTeamPoints']]
    # drop rows with missing values
    df = df.dropna()
    # convert to numpy array
    feature_vals = df.to_numpy()
    next_features = copy.deepcopy(feature_vals)
    # move first row to last row
    next_feature_vals = np.roll(next_features, -1, axis=0)

    flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_put_option(
        feature_vals=feature_vals,
        next_feature_vals=next_feature_vals,
        training_iters=100,
        num_intervals=4,
    )
    names = names.to_numpy()

    print("Fitted LSPI Model")

    # fdql: DNNApprox[Tuple[float, float]] = fitted_dql_put_option(
    #     expiry=expiry_val,
    #     num_steps=num_steps_dql,
    #     num_paths=num_training_paths_dql,
    #     spot_price=spot_price_val,
    #     spot_price_frac=spot_price_frac_dql,
    #     rate=rate_val,
    #     vol=vol_val,
    #     strike=strike_val,
    #     training_iters=training_iters_dql
    # )

    # Index(['Interval', 'CumulativePlayerPointsInterval', 'Home', 'TeamRest',
    #       'OpponentTeamRest', 'CumulativeTeamPointsInterval',
    #       'CumulativeOpponentPointsInterval', 'ScoreMarginInterval',
    #       'ScoreMarginxTimeRemainingInterval',
    #       'ScoreMarginxTimeRemaining2Interval', 'RollingAvgPlayerPoints',
    #       'RollingAverageOpposingTeamAllowedPoints', 'RollingAverageTeamPoints']
    res = np.matmul(feature_vals, flspi.weights.weights)
    res = pd.DataFrame(res)
    df['Player'] = names
    df['res'] = res

    print('x')





