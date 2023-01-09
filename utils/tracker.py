# ---------------------------------------------------------------
# Copyright (c) __________________________ 2022.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------------------------------

class ResultTracker(object):
    def __init__(self, intervals, print_names=None):

        self.intervals = intervals # ['epoch', 'iter', '']
        self.print_names = ['total'] + print_names
        self.metric_names = ['total_loss'] # ['loss', 'count',]
        self.reset_all()

    def reset_all(self):
        self.metrics = {}
        for inter in self.intervals:
            self.metrics[inter] = {}
            # for name in list(self.metrics[inter].keys()):
            #     self.metrics[inter][name] = []

    def reset_interval(self, interval):
        for name in list(self.metrics[interval].keys()):
            self.metrics[interval][name] = []


    def mean(self, interval, metric):

        count_metric = 'count'

        if metric in self.metrics[interval] and sum(self.metrics[interval][count_metric]):
            return sum(self.metrics[interval][metric]) / sum(self.metrics[interval][count_metric])
        else:
            return 0.0

    def get_sum(self, interval, metric):
        return sum(self.metrics[interval][metric])

    def get_len(self, interval, metric):
        if metric not in self.metrics[interval]:
            return 0
        return len(self.metrics[interval][metric])

    def get_loss(self, interval, loss_name):
        if self.get_len(interval, loss_name) > 0:
            return self.get_sum(interval, loss_name) / self.get_len(interval, loss_name)
        return 0

    def loss_str(self, interval):
        string = 'L: '
        for n in sorted(list(set(self.metric_names))):

            if self.get_len(interval, n) and n.startswith('loss'):
                ref = n.split('/')[-1].split('_')[0].upper()
                string += '{}={:2.3f} '.format(
                    ref,
                    self.get_loss(interval, n)
                    )
        return string

    def metric_str(self, interval, ct_name='total', stage='continuous', restrict_text=''):
        if ct_name != 'total':
            string = 'M: {:10}'.format(ct_name)
        else:
            string = 'M: '

        for n in sorted(list(set(self.metric_names))):
            #   print("metric_str", n, self.get_len(interval, n))
            if self.get_len(interval, n) and n.startswith('metric'):
                ref_list = n.split('/')
                m_name = None
                if stage == 'continuous':
                    if ct_name in n:
                        ct_name = ref_list[-1]
                        m_name  = ref_list[-2]
                elif stage in ['caption','chart_text']:
                    m_name = ref_list[-1]
                else:
                    raise
                
                if m_name is not None and restrict_text in m_name:
                    string += '{}={:2.3f} '.format(
                        m_name,
                        self.get_loss(interval, n)
                        )
        return string

    def add(self, interval, metric, val):
        if not isinstance(interval, list):
            interval = [interval]
        
        for inter in interval:
            if metric not in self.metrics[inter]:
                self.metrics[inter][metric] = []
                if metric not in self.metric_names:
                    self.metric_names.append(metric)
                
            if isinstance(val, list):
                self.metrics[inter][metric] += val
            else:
                self.metrics[inter][metric].append(val)
    
    def add_logs(self, split, log=None, total_loss=None):
        if log is not None:
            for l_name, values in log.items():
                k = f"loss/{split}/{l_name}"
                self.add(['epoch', 'iter'], k, values)

        if total_loss is not None:
            self.add(['epoch', 'iter'], f'loss/{split}/total', total_loss)

    def add_metrics(self, metrics, split, metric_name):

        if metric_name == 'continuous':
            for m_name, ct_dict in metrics.items():
                total_container = []
                for ct_name, values in ct_dict.items():
                    k = f"metric/{split}/{m_name}/{ct_name}"
                    v = values.tolist()
                    self.add(['epoch', 'iter'], k, v)
                    total_container += v
                #Save as a total
                k = f"metric/{split}/{m_name}/total"
                self.add(['epoch', 'iter'], k, total_container)
        elif metric_name in ['ksm','rouge']:
            for k, scores in metrics.items():
                k_name = f"metric/{split}/{k}"
                self.add(['epoch', 'iter'], k_name, scores)            

        else:
            print("Metric name not implemented: {}".format(metric_name))
            raise

