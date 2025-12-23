# src/survival_utils.py

import numpy as np
import tensorflow as tf

def cox_partial_likelihood(time, event, risk):
    order = tf.argsort(time, direction="DESCENDING")
    time = tf.gather(time, order)
    event = tf.gather(event, order)
    risk = tf.gather(risk, order)

    hazard = tf.exp(risk)
    log_risk = tf.math.log(tf.cumsum(hazard))
    likelihood = (risk - log_risk) * event
    return -tf.reduce_mean(likelihood)


def harrell_c_index(time, event, risk):
    n = len(time)
    concordant = permissible = 0

    for i in range(n):
        for j in range(n):
            if time[i] < time[j] and event[i] == 1:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5

    return concordant / permissible if permissible > 0 else np.nan
