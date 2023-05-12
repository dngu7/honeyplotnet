
CHART_TO_HEAD_MAP = {
    'vertical box': 'boxplot',
    'line': 'point',
    'scatter': 'point',
    'vertical bar': 'categorical', 
    'horizontal bar': 'categorical'
    }
UNIQ_CHART = sorted(list(set(list(CHART_TO_HEAD_MAP.keys()))))
UNIQ_CHART_HEADS = sorted(list(set(list(CHART_TO_HEAD_MAP.values()))))
CHART_TO_HEAD_IDX = {m: UNIQ_CHART_HEADS.index(n) for m,n in CHART_TO_HEAD_MAP.items()}
HEAD_IDX_TO_CHART = {n: m for m,n in CHART_TO_HEAD_IDX.items()}

REG_DIMS = {
    'boxplot': 5,
    'point': 2,
    'categorical': 1
    }

SCALE_DIMS = {
    'minmax': {
        'boxplot': 2,
        'point': 4,
        'categorical': 2
        },
    'offset': {
        'boxplot': 2,
        'point': 4,
        'categorical': 2
    }
}

PAD_ID = -100