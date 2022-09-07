
DEFAULT_B = -1

B_VALS = {
    'Cochlea_L': {
        'clinical': {
            'apl-mm-tol-1': int(-1e2)
        }
    },
    'Cochlea_R': {
        'clinical': {
            'apl-mm-tol-1': int(-1e1)
        }
    },
    'SpinalCord': {
        'clinical': {
            'apl-mm-tol-1': int(-1e1)
        }
    }
}

def get_p_init_b(
    region: str,
    model: str,
    metric: str) -> float:
    # if region in B_VALS.keys():
    #     if model in B_VALS[region].keys():
    #         if metric in B_VALS[region][model]:
    #             return B_VALS[region][model][metric]

    return DEFAULT_B    
