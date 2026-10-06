import copy
import random
import numpy as np

from deap import gp, creator
from deap import tools

def init_primitives_general(pset):

    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)

    pset.addTerminal(str('NIQ'))
    pset.addTerminal(str('WIQ'))
    pset.addTerminal(str('MWT'))
    pset.addTerminal(str('PT'))
    pset.addTerminal(str('NPT'))
    pset.addTerminal(str('OWT'))
    pset.addTerminal(str('WKR'))
    pset.addTerminal(str('NOR'))

    pset.addTerminal(str('TIS'))

    pset.addTerminal(str('SLACK'))

def init_primitives_factory(pset):
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)

    pset.addTerminal(str('FEC'))  
    pset.addTerminal(str('FEO')) 
    pset.addTerminal(str('FET'))  
    pset.addTerminal(str('FMAR'))  
    pset.addTerminal(str('FAU'))  
    pset.addTerminal(str('ESO'))  
    pset.addTerminal(str('FVR')) 

def lf(x):
    return 1 / (1 + np.exp(-x))

def init_toolbox(toolbox, pset):
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)

    toolbox.register("mate",lim_xmate)
    toolbox.register("mutate",lim_xmut,expr=toolbox.expr_mut)

def init_toolbox_dual(toolbox, pset_rou, pset_seq):

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox.register("expr_rou", gp.genHalfAndHalf, pset=pset_rou, min_=1, max_=6)
    toolbox.register("expr_seq", gp.genHalfAndHalf, pset=pset_seq, min_=1, max_=6)

    toolbox.register("tree_rou", tools.initIterate, gp.PrimitiveTree, toolbox.expr_rou)
    toolbox.register("tree_seq", tools.initIterate, gp.PrimitiveTree, toolbox.expr_seq)

    def _init_ind():
        t_rou = toolbox.tree_rou()
        t_rou.pset = pset_rou
        t_seq = toolbox.tree_seq()
        t_seq.pset = pset_seq
        return creator.Individual([t_rou, t_seq])

    toolbox.register("individual", _init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile_rou", gp.compile, pset=pset_rou)
    toolbox.register("compile_seq", gp.compile, pset=pset_seq)

    toolbox.register("mate", lim_xmate)
    toolbox.register("mutate", lim_xmut)

def init_toolbox_multi(toolbox, pset_factory, pset_general):
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox.register("expr_factory", gp.genHalfAndHalf, pset=pset_factory, min_=1, max_=6)
    toolbox.register("expr_general", gp.genHalfAndHalf, pset=pset_general, min_=1, max_=6)

    toolbox.register("tree_factory", tools.initIterate, gp.PrimitiveTree, toolbox.expr_factory)
    toolbox.register("tree_general", tools.initIterate, gp.PrimitiveTree, toolbox.expr_general)

    def _init_ind():
        t1 = toolbox.tree_factory()
        t1.pset = pset_factory 
        t2 = toolbox.tree_general()
        t2.pset = pset_general
        t3 = toolbox.tree_general()
        t3.pset = pset_general

        return creator.Individual([t1, t2, t3])

    toolbox.register("individual", _init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile_factory", gp.compile, pset=pset_factory)
    toolbox.register("compile_general", gp.compile, pset=pset_general)

    toolbox.register("mate", lim_xmate)
    toolbox.register("mutate", lim_xmut)

def maxheight(v):
    return max(i.height for i in v)

def wrap(func, *args, **kwargs):

    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > MAX_HEIGHT:
            new_inds[i] = random.choice(keep_inds)
    return new_inds

def xmate(ind1, ind2):

    i = random.randrange(0, 2)
    ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2

def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)

def xmut(ind, min_=2, max_=8):

    i = random.randrange(0, 2)
    tree = ind[i]
    pset = getattr(tree, "pset", getattr(ind, "pset", None))
    if pset is None:
        raise ValueError("No pset bound to the selected tree or individual.")
    def _expr(pset, type_=None):
        return gp.genFull(pset=pset, min_=min_, max_=max_, type_=type_)
    mutant, = gp.mutUniform(tree, expr=_expr, pset=pset)
    ind[i] = mutant
    ind[i].pset = pset
    return ind,

def lim_xmut(ind, min_=2, max_=8, **_):

    res = wrap(xmut, ind, min_=min_, max_=max_)
    return res

def add_abs(a, b):
    return np.abs(np.add(a, b))

def sub_abs(a, b):
    return np.abs(np.subtract(a, b))

def mt_if(a, b, c):
    return np.where(a < 0, b, c)

def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

MAX_HEIGHT = 8
N_TREES = 2
