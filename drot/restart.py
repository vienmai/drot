import numpy.linalg as nla


def fixed_restart(k, milestones=[]):
    return True if k in milestones else False

def adaptive_restart():
    ...
