
PAD_IDX = -100
REF_REG = ['\[\d+\]'] + ['\[{}\]'.format(','.join(['\d+' for _ in range(count)])) for count in range(2, 5)] + ['\[\d+-\d+\]', '\[\d+-\d+,\d+\]', '\[\d+,\d+-\d+\]', '\[\d+-\d+,\d+-\d+\]'] 

CB_TOKEN_TEMP = '<CB{}-{}>'

CHART_TYPE_MAP = [
          'area',
          'heatmap',
          'horizontal bar',
          'horizontal interval',
          'line',
          'manhattan',
          'map',
          'pie',
          'scatter',
          'scatter-line',
          'surface',
          'venn',
          'vertical bar',
          'vertical box',
          'vertical interval'
          ]
          
TASK2PREPEND = {
        'caption': 'CAPTION: ',
        'series_name': 'SERIES NAME: ',
        'categorical': 'CATEGORICAL: ',
        'axis': 'AXIS: ',
      }

CHART_TO_HEAD_MAP = {
    'vertical box': 'boxplot',
    'line': 'point',
    'scatter': 'point',
    'vertical bar': 'categorical', 
    'horizontal bar': 'categorical'
    }

UNIQ_CHART_HEADS = sorted(list(set(list(CHART_TO_HEAD_MAP.values()))))
