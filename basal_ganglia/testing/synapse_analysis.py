def pos(value):
    if value < 0:
        return 0
    
    return value

equations="""
        tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(pre.trace - mean(pre.trace) - threshold_pre) * (mean(post.r) - post.r  - threshold_post)
        aux = if (trace>0): 1 else: 0
        dopa_mod = if (DA_type*dopa_sum>0): K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
        trace2 = trace
        delta = dopa_mod * trace2 - alpha * pos(trace2)
        tau*dw/dt = delta : min=0
    """
threshold_pre = 0.012
threshold_post = 0.3

#action1
m1_strd1_pre = 0.08492559
m1_strd1_pre_mean = 0.07019753318467481
m1_gpi_post = 0.13756663
m1_gpi_post_mean = 0.5783806589040699

#action2
m2_strd1_pre = 0.05545579
m2_strd1_pre_mean = 0.07019753318467481
m2_gpi_post = 0.0
m2_gpi_post_mean = 0.5783806589040699


m1_pre = pos(m1_strd1_pre - m1_strd1_pre_mean - threshold_pre)
m1_post = m1_gpi_post_mean - m1_gpi_post - threshold_post

m2_pre = pos(m2_strd1_pre - m2_strd1_pre_mean - threshold_pre)
m2_post = m2_gpi_post_mean - m2_gpi_post - threshold_post

print("~~~~~~~~~~~~~~~~~~~~")
print(m1_pre)
print(m1_post)
print(m1_pre*m1_post)
print("~~~~~~~~~~~~~~~~~~~~")
print(m2_pre)
print(m2_post)
print(m2_pre*m2_post)
print("~~~~~~~~~~~~~~~~~~~~")