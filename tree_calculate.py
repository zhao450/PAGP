import numpy as np
import math
from deap import gp

def GP_evolve_S(data, tree_S): # genetic programming evolved sequencing rule
    individualvalue,length = treeNode_S(tree_S, 0, data)  # todo: actually, this should be used for sequencing rule
    # if isinstance(individualvalue, (np.int64, np.float64, float, int)):
    #     return 0 #todo: need to check if this is right!!! by mengxu 2022.10.15
    # job_position = individualvalue.argmin()
    return individualvalue

def treeNode_S(tree, index, data):
    if tree[index].arity == 2:
        left,length_left = treeNode_S(tree, index+1, data)
        right,length_right = treeNode_S(tree, index+length_left+1, data)
        if tree[index].name == 'add':
            return left+right,length_left+length_right+1
        elif tree[index].name == 'subtract':
            return left-right,length_left+length_right+1
        elif tree[index].name == 'multiply':
            return left*right,length_left+length_right+1
        elif tree[index].name == 'protected_div':
            return protected_div(left,right),length_left+length_right+1
        elif tree[index].name == 'maximum':
            return np.maximum(left,right),length_left+length_right+1
        elif tree[index].name == 'minimum':
            return np.minimum(left,right),length_left+length_right+1
    elif tree[index].arity == 1:
        if tree[index].name == 'lf': # add by mengxu 2022.11.08
            ref,length_ref = treeNode_S(tree, index + 1, data)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref)),length_ref+1
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                    # print(ref[i])
                return ref,length_ref+1
    elif tree[index].arity == 0:
        if tree[index].name == 'NIQ':
            return data[0],1
        elif tree[index].name == 'WIQ':
            return data[4],1
        elif tree[index].name == 'MWT':
            return data[1],1
        elif tree[index].name == 'PT':
            return data[2],1
        elif tree[index].name == 'NPT':
            return data[3],1
        elif tree[index].name == 'OWT':
            return data[5],1
        elif tree[index].name == 'WKR':
            return data[7],1
        elif tree[index].name == 'NOR':
            return data[8],1
        elif tree[index].name == 'TIS':
            return data[6],1
        elif tree[index].name == 'SLACK':
            return data[9],1
        
def GP_pair_S_test(data, tree_S):
    individualvalue,length = treeNode_S_test(tree_S, 0, data)  # todo: actually, this should be used for sequencing rule
    return individualvalue


def treeNode_S_test(tree, index, data):
    if tree[index] == 'add':
        left,length_left = treeNode_S_test(tree, index+1, data)
        right,length_right = treeNode_S_test(tree, index+length_left+1, data)
        return left+right,length_left+length_right+1
    elif tree[index] == 'subtract':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index+length_left+1, data)
        return left - right, length_left + length_right + 1
    elif tree[index] == 'multiply':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index+length_left+1, data)
        return left * right, length_left + length_right + 1
    elif tree[index] == 'protected_div':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return protected_div(left,right), length_left + length_right + 1
    elif tree[index] == 'maximum':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return np.maximum(left,right), length_left + length_right + 1
    elif tree[index] == 'minimum':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return np.minimum(left,right), length_left + length_right + 1
    elif tree[index] == 'lf': # add by mengxu 2022.11.08
        ref,length_ref = treeNode_S_test(tree, index+1, data)
        if isinstance(ref, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-ref)),length_ref+1
        else:
            for i in range(len(ref)):
                ref[i] = 1 / (1 + np.exp(-ref[i]))
            return ref,length_ref+1
    elif tree[index] == 'NIQ':
        return data[0],1
    elif tree[index] == 'WIQ':
        return data[4],1
    elif tree[index] == 'MWT':
        return data[1],1
    elif tree[index] == 'PT':
        return data[2],1
    elif tree[index] == 'NPT':
        return data[3],1
    elif tree[index] == 'OWT':
        return data[5],1
    elif tree[index] == 'WKR':
        return data[7],1
    elif tree[index] == 'NOR':
        return data[8],1
    elif tree[index] == 'TIS':
        return data[6],1
    elif tree[index] == 'SLACK':
        return data[9],1
    

def GP_evolve_fac(data, tree_S): # genetic programming evolved sequencing rule
    individualvalue,length = treeNode_fac(tree_S, 0, data)  # todo: actually, this should be used for sequencing rule
    # if isinstance(individualvalue, (np.int64, np.float64, float, int)):
    #     return 0 #todo: need to check if this is right!!! by mengxu 2022.10.15
    # job_position = individualvalue.argmin()
    return individualvalue

def treeNode_fac(tree, index, data):
    if tree[index].arity == 2:
        left,length_left = treeNode_fac(tree, index+1, data)
        right,length_right = treeNode_fac(tree, index+length_left+1, data)
        if tree[index].name == 'add':
            return left+right,length_left+length_right+1
        elif tree[index].name == 'subtract':
            return left-right,length_left+length_right+1
        elif tree[index].name == 'multiply':
            return left*right,length_left+length_right+1
        elif tree[index].name == 'protected_div':
            return protected_div(left,right),length_left+length_right+1
        elif tree[index].name == 'maximum':
            return np.maximum(left,right),length_left+length_right+1
        elif tree[index].name == 'minimum':
            return np.minimum(left,right),length_left+length_right+1
    elif tree[index].arity == 1:
        if tree[index].name == 'lf': # add by mengxu 2022.11.08
            ref,length_ref = treeNode_fac(tree, index + 1, data)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref)),length_ref+1
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                    # print(ref[i])
                return ref,length_ref+1
    elif tree[index].arity == 0:
        if tree[index].name == 'FEC':
            return data[0],1
        elif tree[index].name == 'FEO':
            return data[1],1
        elif tree[index].name == 'FET':
            return data[2],1
        elif tree[index].name == 'FMAR':
            return data[3],1
        elif tree[index].name == 'FAU':
            return data[4],1
        elif tree[index].name == 'ESO':
            return data[5],1
        elif tree[index].name == 'FVR':
            return data[6],1

def GP_pair_fac_test(data, tree_S):
    individualvalue,length = treeNode_fac_test(tree_S, 0, data)
    return individualvalue  

def treeNode_fac_test(tree, index, data):
    if tree[index] == 'add':
        left,length_left = treeNode_fac_test(tree, index+1, data)
        right,length_right = treeNode_fac_test(tree, index+length_left+1, data)
        return left+right,length_left+length_right+1
    elif tree[index] == 'subtract':
        left, length_left = treeNode_fac_test(tree, index + 1, data)
        right, length_right = treeNode_fac_test(tree, index+length_left+1, data)
        return left - right, length_left + length_right + 1
    elif tree[index] == 'multiply':
        left, length_left = treeNode_fac_test(tree, index + 1, data)
        right, length_right = treeNode_fac_test(tree, index+length_left+1, data)
        return left * right, length_left + length_right + 1
    elif tree[index] == 'protected_div':
        left, length_left = treeNode_fac_test(tree, index + 1, data)
        right, length_right = treeNode_fac_test(tree, index + length_left + 1, data)
        return protected_div(left,right), length_left + length_right + 1
    elif tree[index] == 'maximum':
        left, length_left = treeNode_fac_test(tree, index + 1, data)
        right, length_right = treeNode_fac_test(tree, index + length_left + 1, data)
        return np.maximum(left,right), length_left + length_right + 1
    elif tree[index] == 'minimum':
        left, length_left = treeNode_fac_test(tree, index + 1, data)
        right, length_right = treeNode_fac_test(tree, index + length_left + 1, data)
        return np.minimum(left,right), length_left + length_right + 1
    elif tree[index] == 'lf': # add by mengxu 2022.11.08
        ref,length_ref = treeNode_fac_test(tree, index+1, data)
        if isinstance(ref, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-ref)),length_ref+1
        else:
            for i in range(len(ref)):
                ref[i] = 1 / (1 + np.exp(-ref[i]))
            return ref,length_ref+1
    elif tree[index] == 'FEC':
        return data[0],1
    elif tree[index] == 'FEO':
        return data[1],1
    elif tree[index] == 'FET':
        return data[2],1
    elif tree[index] == 'FMAR':
        return data[3],1
    elif tree[index] == 'FAU':
        return data[4],1
    elif tree[index] == 'ESO':
        return data[5],1
    elif tree[index] == 'FVR':
        return data[6],1


def GP_evolve_S_PC(data, tree_S): # genetic programming evolved sequencing rule
    individualvalue,length = treeNode_S_PC(tree_S, 0, data)  # todo: actually, this should be used for sequencing rule
    # if isinstance(individualvalue, (np.int64, np.float64, float, int)):
    #     return 0 #todo: need to check if this is right!!! by mengxu 2022.10.15
    # job_position = individualvalue.argmin()
    return individualvalue

def treeNode_S_PC(tree, index, data):
    if tree[index].arity == 2:
        left,length_left = treeNode_S_PC(tree, index+1, data)
        right,length_right = treeNode_S_PC(tree, index+length_left+1, data)
        if tree[index].name == 'add':
            return left+right,length_left+length_right+1
        elif tree[index].name == 'subtract':
            return left-right,length_left+length_right+1
        elif tree[index].name == 'multiply':
            return left*right,length_left+length_right+1
        elif tree[index].name == 'protected_div':
            return protected_div(left,right),length_left+length_right+1
        elif tree[index].name == 'maximum':
            return np.maximum(left,right),length_left+length_right+1
        elif tree[index].name == 'minimum':
            return np.minimum(left,right),length_left+length_right+1
    elif tree[index].arity == 1:
        if tree[index].name == 'lf': # add by mengxu 2022.11.08
            ref,length_ref = treeNode_S_PC(tree, index + 1, data)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref)),length_ref+1
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                    # print(ref[i])
                return ref,length_ref+1
    elif tree[index].arity == 0:
        if tree[index].name == 'NIQ':
            return data[0],1
        elif tree[index].name == 'WIQ':
            return data[4],1
        elif tree[index].name == 'MWT':
            return data[1],1
        elif tree[index].name == 'PT':
            return data[2],1
        elif tree[index].name == 'NPT':
            return data[3],1
        elif tree[index].name == 'OWT':
            return data[5],1
        elif tree[index].name == 'WKR':
            return data[7],1
        elif tree[index].name == 'NOR':
            return data[8],1
        elif tree[index].name == 'TIS':
            return data[6],1
        elif tree[index].name == 'SLACK':
            return data[9],1


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x