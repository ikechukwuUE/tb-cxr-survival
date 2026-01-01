# src/survival_utils.py

import numpy as np
import tensorflow as tf

def cox_ph_loss(y_true, y_pred):
    """
    Custom Cox Proportional Hazards Loss for Deep Learning.
    Mathematically implements the Negative Partial Log Likelihood.
    """
    # 1. Parse Inputs
    # y_true is (Batch, 2) -> [Time, Event]
    time = tf.cast(y_true[:, 0], tf.float32)
    event = tf.cast(y_true[:, 1], tf.float32)
    
    # y_pred is (Batch, 1) -> [Log Hazard Risk]
    risk = y_pred[:, 0]
    
    # 2. Create Risk Set Matrix (Who is at risk at time t_i?)
    # We compare every patient i against every patient j
    # Mask = 1 if Time_j >= Time_i (Patient j lived longer than or equal to i)
    time_i = tf.expand_dims(time, 1)
    time_j = tf.expand_dims(time, 0)
    risk_set_mask = tf.cast(time_j >= time_i, tf.float32)
    
    # 3. Log-Sum-Exp Trick (for Numerical Stability)
    # We need log( sum( exp(risk_j) ) ) over the risk set.
    # Subtracting max(risk) prevents overflow (exp(100) -> inf).
    risk_max = tf.reduce_max(risk)
    exp_risk = tf.exp(risk - risk_max)
    
    # Sum only the patients currently in the risk set
    masked_sum_exp = tf.reduce_sum(exp_risk * risk_set_mask, axis=1)
    
    # Add max back after log
    log_sum_risk = tf.math.log(masked_sum_exp + 1e-8) + risk_max
    
    # 4. Calculate Negative Log Likelihood
    # Loss is calculated ONLY for patients who experienced the event (Death)
    # Censored patients contribute to the risk set denominator but have 0 loss directly.
    loss_per_sample = event * (risk - log_sum_risk)
    
    return -tf.reduce_mean(loss_per_sample)

def cox_partial_likelihood(time, event, risk):
    order = tf.argsort(time, direction="DESCENDING")
    time = tf.gather(time, order)
    event = tf.gather(event, order)
    risk = tf.gather(risk, order)

    hazard = tf.exp(risk)
    log_risk = tf.math.log(tf.cumsum(hazard))
    likelihood = (risk - log_risk) * event
    return -tf.reduce_mean(likelihood)


from numba import jit

# nopython=True forces compilation to machine code
@jit(nopython=True)
def harrell_c_index(y_true, scores, event):
    n = len(y_true)
    
    concordant = 0.0
    permissible = 0.0
    ties = 0.0
    
    # We loop through every unique pair (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            
            # --- optimization: skip if both are censored (useless pair) ---
            if event[i] == 0 and event[j] == 0:
                continue
            
            # Identify who is who
            t_i, t_j = y_true[i], y_true[j]
            e_i, e_j = event[i], event[j]
            s_i, s_j = scores[i], scores[j]
            
            is_permissible = False
            
            # CASE 1: Both had events
            if e_i == 1 and e_j == 1:
                is_permissible = True
                
            # CASE 2: One censored
            elif e_i == 1 and e_j == 0:
                # i died, j was censored. j must have lived at least as long as i
                if t_j >= t_i:
                    is_permissible = True
            elif e_i == 0 and e_j == 1:
                # j died, i was censored. i must have lived at least as long as j
                if t_i >= t_j:
                    is_permissible = True
            
            # If the pair is comparable, check concordance
            if is_permissible:
                permissible += 1
                
                # Check for Ties in Risk Score
                if s_i == s_j:
                    ties += 1
                    continue # Stop here, don't check concordance
                
                # Check Concordance: 
                # Does the person who died earlier (lower time) have higher risk (higher score)?
                if t_i < t_j:
                    if s_i > s_j:
                        concordant += 1
                elif t_i > t_j:
                    if s_i < s_j:
                        concordant += 1
                else:
                    # Times are equal (t_i == t_j)
                    # If both died at the same time but had different scores, 
                    # standard C-index often counts this as a tie or ignores it.
                    pass 

    # Avoid division by zero
    if permissible == 0:
        return 0.0
        
    return (concordant + 0.5 * ties) / permissible