import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from .constant import CHART_TO_HEAD_MAP, UNIQ_CHART_HEADS

SINGLE_PLOT_TIGHT_LAYOUT_REC = [0, 0.2, 1.0, 1.0]
SINGLE_PLOT_TEXT_LOC = [0.5, 0.05]
MAX_SENT_CAPTION = 2
MAX_TEXT_CAPTION = 400

def unscale_conts(cont_preds, scale_preds, tab_shape=None, scale_eps=1.00001, scale_exponent=10):
    if tab_shape is not None:
      row_counts = tab_shape['row']
      col_counts = tab_shape['col']
    
    cont_values = []
    for bidx, (all_cont_pred) in enumerate(cont_preds):

      #if bidx == 0:
        #print(bidx, all_cont_pred[0][:10])

      #Can only generate as far as we have predictions available.
      if tab_shape is not None:
        rows = min(row_counts[bidx], all_cont_pred.size(0))
        cols = min(col_counts[bidx], all_cont_pred.size(1))
      else:
        rows = all_cont_pred.size(0)
        cols = all_cont_pred.size(1)

      cont_pred = all_cont_pred[:rows, :cols, :].clone()
      scale_pred = scale_preds[bidx].clone()
      if len(scale_pred.shape) == 1:
        scale_pred = scale_pred.view(1, -1)

      cont_dims = cont_pred.size(-1)
      
      #Reverse offseting. Increment from first point
      if cont_dims * 2 == scale_pred.size(-1):

        for dim in range(cont_dims):
          mindim = dim * 2
          rngdim = mindim + 1
          
          min_scale = scale_pred[:rows, mindim]
          rng_scale = scale_pred[:rows, rngdim]

          #scale_min_rng = (bsz, 2)
          min_scale = min_scale[:,None]
          rng_scale = rng_scale[:,None]

          #Perform scaling
          cont_pred[:,:,dim] = cont_pred[:,:,dim] * rng_scale + min_scale
      else:

        # for cidx in range(1, cont_pred.size(1)):
        #   cont_pred[:, cidx, :] += cont_pred[:, cidx - 1, :]
        for cidx in range(1, cont_pred.size(2)):
          cont_pred[:, :, cidx] += cont_pred[:, :, cidx - 1]

        min_scale = scale_pred[:rows, 0]
        rng_scale = scale_pred[:rows, 1]

        #scale_min_rng = (bsz, 2)
        min_scale = min_scale.view(-1,1,1)
        rng_scale = rng_scale.view(-1,1,1)

        cont_pred = cont_pred * rng_scale + min_scale

      #print("cont_pred", bidx, cont_pred.shape)
      cont_values.append(cont_pred)
    return cont_values
    

def unscale_scales(scale_logits, tab_shape=None,  scale_eps_min=1.100001, scale_eps_rng=1.100001, scale_exponent=10):
    if tab_shape is not None:
        row_counts = tab_shape['row']
    
    scale_values = []
    for bidx, logits in enumerate(scale_logits):

      if tab_shape is not None:
        rows = int(row_counts[bidx])
        logits = logits[:rows, :]

      base = torch.ones_like(logits) * scale_exponent
      scale_pred = torch.pow(base, logits)

      dims = scale_pred.size(-1)
      min_dims = [i for i in range(dims) if (i % 2) == 0]
      rng_dims  = [i for i in range(dims) if (i % 2) == 1]

      scale_pred[:,min_dims] -= scale_eps_min
      scale_pred[:,rng_dims] -= scale_eps_rng

      scale_pred = torch.abs(scale_pred)
      
      scale_values.append(scale_pred)

    return scale_values

def prepare_mpl(x):
    chart_types      = x.get('chart_type')
    chart_idx        = x.get('chart_idx')
    if chart_idx is None and chart_types is None:
        raise ValueError("Provide one")

    scale_tens_list  = x['scale']['inputs_embeds']
    cont_tens_list   = x['continuous']['inputs_embeds']

    #if tab_shape not available, it must be from dataset
    tab_shape = x.get('shape')
    if tab_shape is not None:
        tab_shape = tab_shape['counts']
    else:
        attn_mask = x['continuous']['attention_mask'].sum(-1)
        num_cols = torch.max(attn_mask, dim=-1).values.cpu().tolist()
        num_rows = torch.where(attn_mask > 0, 1, 0).sum(-1).cpu().tolist()
        
        tab_shape = {}
        tab_shape['row'] = num_rows
        tab_shape['col'] = num_cols

    scale_values = unscale_scales(scale_tens_list, tab_shape=tab_shape)
    cont_values = unscale_conts(cont_tens_list, scale_values, tab_shape=tab_shape)

    #Convert into a list
    chart_list = []
    for idx, values in enumerate(cont_values):
        chart_dict = {}

        if chart_types is not None:
            ct = chart_types[idx]
            chart_dict['chart_type'] = ct
            chart_dict['head_type']  = CHART_TO_HEAD_MAP[ct]
        elif chart_idx is not None:
            ct_idx = chart_idx[idx]
            chart_dict['head_type'] = UNIQ_CHART_HEADS[ct_idx]
        else:
            raise

        if chart_dict['head_type'] == 'categorical':
            assert values.size(-1) == 1, f"invalid shape for categorical data, {values.shape}"
            values = torch.reshape(values, [values.size(0), values.size(1)])
            chart_dict['values'] = values.cpu().tolist()
        elif chart_dict['head_type'] == 'point':
            assert values.size(-1) == 2, f"invalid shape for point data, {values.shape}"
            x_val = values[:,:,0].cpu().tolist()
            y_val = values[:,:,1].cpu().tolist()
            chart_dict['values'] = {'x': x_val, 'y': y_val}
        elif chart_dict['head_type'] == 'boxplot':
            assert values.size(-1) == 5, f"invalid shape for boxplot data, {values.shape}"
            chart_dict['values'] = values.cpu().tolist()
            
        chart_list.append(chart_dict)
    return chart_list

def create_bar_chart(x_data, text, f_name, xhat_data=None, bar_width=0.75):

    if xhat_data is not None:
        _create_bar_chart_double(x_data=x_data, xhat_data=xhat_data, text=text, f_name=f_name, bar_width=bar_width)
    else:
        _create_bar_chart_single(x_data=x_data, text=text, f_name=f_name, base_width=bar_width)


def _create_bar_chart_single(x_data, text, f_name, base_width=0.25, flip_thres=4):
    recon_values = x_data['values']
    
    _ = plt.figure()
    plt.figure(facecolor='white')

    #Remove repeats between categorical and series names
    text['categorical'] = [s for s in text['categorical'] if 'unnamed' not in s]
    text['series_name'] = [s for s in text['series_name'] if s not in text['categorical'] and 'unnamed'  not in s]
    text['axis_titles'] = [s for s in text['axis_titles'] if s not in text['categorical'] and 'unnamed'  not in s]

    #clip by number of available categorical values
    max_cat = min(len(text['categorical']), len(recon_values[0]))
    max_series = min(len(text['series_name']), len(recon_values))

    #Manually remove
    if max_cat <= 1 or max_series <= 1:
        return

    if max_series == 1:
        bar_width = base_width
        ticks = [r for r in range(max_cat)]
    else:
        bar_width = base_width / max_series
        #ticks = [r + bar_width for r in range(max_cat)]
        ticks = [r + base_width / 3 for r in range(max_cat)]
    
    label_count = 0
    max_label_len = 0
    for idx, (y, label)in enumerate(zip(recon_values, text['series_name'])):
        if idx > max_series:
            break

        y = y[:max_cat]
        if idx == 0:
            br1 = np.arange(len(y))
        else:
            br1 = [x + bar_width for x in br1]

        label = text['series_name'][idx]
        if 'unnamed' in label:
            label = None
        else:
            label_count += 1
            if len(label) > max_label_len:
                max_label_len = len(label)
        
        if max_cat > flip_thres:
            plt.barh(br1, y, height=bar_width, label=label)
        else:
            plt.bar(br1, y, width=bar_width, label=label)

        #plt.bar(br1, y, width=bar_width, label=label)
    
    rotation = 0
    fontsize = 'small'
    wrap = True
    if max_cat > flip_thres:
        rotation = 30
        wrap = False
        fontsize=8

    categorical_labels = text['categorical'][:max_cat]
    max_cat_label_len = max(len(c) for c in categorical_labels)
    if max_cat > flip_thres:
        plt.yticks(ticks, categorical_labels, wrap=wrap, fontsize=fontsize)
    else:
        plt.xticks(ticks, categorical_labels, wrap=wrap, fontsize=fontsize, rotation=rotation)

    if max_series > 1 and label_count > 0:
        if max_label_len <= 5:
            #Short legend box on the right
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        elif max_label_len >= 40:
            #Long legend box should be put on the top
            plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=1)
        elif max_cat_label_len >= 40:
            #Long categorical labels mean it must be put on the top
            if max_label_len < 10 and label_count < 4:
                plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=label_count)
            else:
                plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=1)
        elif label_count >= 4:
            #Lots of labels, put on the right
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")

        else:
            if max_label_len > 20 or max_cat_label_len > 40:
                ncol = 1
            else:
                ncol = label_count
            
            plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=ncol)

    if len(text['axis_titles']) > 0:
        axis_title1 = text['axis_titles'][0].split('.')[0]
        if max_cat > flip_thres:
            plt.xlabel(axis_title1)
        else:
            plt.ylabel(axis_title1)

    #Only keep first 3 sentences
    caption = text['caption'].replace('\n','')
    caption = '.'.join(caption.split('.')[:MAX_SENT_CAPTION])[:MAX_TEXT_CAPTION]
    s = SINGLE_PLOT_TEXT_LOC
    plt.figtext(s[0], s[1], caption, wrap=True, horizontalalignment='center')

    plt.tight_layout(rect=SINGLE_PLOT_TIGHT_LAYOUT_REC)
    plt.savefig(fname=f_name)
    plt.close()

def _create_bar_chart_double(x_data, xhat_data, text, f_name, bar_width=0.25):

    # Create original plot
    original_values = x_data['values']
    plt.subplot(2, 1, 1)  # row 1, column 2, count 1

    #Pick horizontal or horizontal based on xticks

    #plt.title('Original')

    #plt.xlabel('x-axis')
    if len(text['axis_titles']) > 0:
        axis_title1 = text['axis_titles'][0].split('.')[0]
        if len(text['categorical']) > 5:
            plt.xlabel(axis_title1)
        else:
            plt.ylabel(axis_title1)
    
    if len(original_values) == 1:
        bar_width = 0.75
        ticks = [r for r in range(len(original_values[0]))]
    else:
        bar_width = 0.75 / len(original_values)
        ticks = [r + bar_width for r in range(len(original_values[0]))]
        
    for idx, y in enumerate(original_values):
        if idx == 0:
            br1 = np.arange(len(y))
        else:
            br1 = [x + bar_width for x in br1]
        
        label = text['series_name'][idx]
        if 'unnamed' in label:
            label = None
        
        if len(text['categorical']) > 5:
            plt.barh(br1, y, height=bar_width, label=label)
        else:
            plt.bar(br1, y, width=bar_width, label=label)

    rotation = 0
    wrap = True
    fontsize = 'small'
    if len(text['categorical']) > 5:
        rotation = 30
        wrap = True
        fontsize = 8    
    
    cat_labels = [c for c in text['categorical'] if 'unnamed' not in c]
    if len(cat_labels):
        if len(text['categorical']) > 5:
            plt.yticks(ticks[:len(cat_labels)], cat_labels, wrap=wrap, fontsize=fontsize)
        else:
            plt.xticks(ticks[:len(cat_labels)], cat_labels, wrap=wrap, fontsize=fontsize, rotation=rotation)


    if len(original_values) > 1:
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        
    # Create reconstruction
    plt.subplot(2, 1, 2)
    #plt.title('Reconstruction')

    if len(text['axis_titles']) > 0:
        axis_title1 = text['axis_titles'][0].split('.')[0]
        if len(text['categorical']) > 5:
            plt.xlabel(axis_title1)
        else:
            plt.ylabel(axis_title1)
    
    recon_values = xhat_data['values']

    #clip by number of available categorical values
    clip_val = min(len(text['categorical']), len(recon_values[0]))

    if len(recon_values) == 1:
        bar_width = 0.75
        ticks = [r for r in range(len(recon_values[0]))]
    else:
        bar_width = 0.75 / clip_val
        ticks = [r + bar_width for r in range(len(recon_values[0]))]

    for idx, (y, label)in enumerate(zip(recon_values, text['series_name'])):

        #label = [l for l in label if 'unnamed' not in l]
        if 'unnamed' in label:
            label = None

        y = y[:clip_val]
        if idx == 0:
            br1 = np.arange(len(y))
        else:
            br1 = [x + bar_width for x in br1][:clip_val]

        if len(text['categorical']) > 5:
            plt.barh(br1, y, height=bar_width, label=label)
        else:
            plt.bar(br1, y, width=bar_width, label=label)

    rotation = 0
    wrap = True
    fontsize = 'small'
    if clip_val > 5:
        rotation = 30
        wrap = True
        fontsize = 8

    if len(text['categorical']) > 5:
        plt.yticks(ticks[:clip_val], text['categorical'][:clip_val], wrap=wrap, fontsize=fontsize)
    else:
        plt.xticks(ticks[:clip_val], text['categorical'][:clip_val], wrap=wrap, fontsize=fontsize, rotation=rotation)

    if len(recon_values) > 1:
        # if len(recon_values) <= 2:
        #     plt.legend()
        # else:
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")

    #Add caption
    #plt.subplots_adjust(bottom=0.2) 
    plt.tight_layout(rect=[0, 0.0, 1.0, 0.8])

    #Only keep first 3 sentences
    caption = text['caption'].replace('\n','')
    caption = '.'.join(caption.split('.')[:MAX_SENT_CAPTION])[:MAX_TEXT_CAPTION]
    plt.figtext(0.5, 0.9, caption, wrap=True, horizontalalignment='center')

    # space between the plots
    plt.savefig(fname=f_name)
    plt.close()

def create_scatter(x_data, text, f_name, xhat_data=None, bar_width=0.25):

    if xhat_data is not None:
        _create_scatter_double(x_data=x_data, xhat_data=xhat_data, text=text, f_name=f_name)
    else:
        _create_scatter_single(x_data=x_data, text=text, f_name=f_name)


def _create_scatter_single(x_data, text, f_name):
    x_values = x_data['values']['x']
    y_values = x_data['values']['y']
    fig = plt.figure()
    plt.figure(facecolor='white')

    #Remove repeats between categorical and series names
    text['series_name'] = [s for s in text['series_name'] if 'unnamed'  not in s]

    label_count = 0
    max_label_len = 0
    #print(f_name, len(x_values[0]), len(x_values), len(y_values[0]), len(y_values))
    if len(x_values[0]) == 1:
        return

    for idx, (x, y, label) in enumerate(zip(x_values, y_values, text['series_name'])):
        if 'unnamed' in label:
            label = None
        else:
            label_count += 1
            if len(label) > max_label_len:
                max_label_len = len(label)

        plt.scatter(x, y, cmap = 'viridis', label=label)



    if len(x_values) > 1 and label_count > 0:
        if max_label_len <= 5:
            #Short legend box on the right
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        elif max_label_len >= 40:
            #Long legend box should be put on the top
            plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=1)

        elif label_count >= 4:
            #Lots of labels, put on the right
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")

        else:
            if max_label_len > 20:
                ncol = 1
            else:
                ncol = label_count
            
            plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=ncol)

    if len(text['axis_titles']) > 0:
        axis_title1 = text['axis_titles'][0].split('.')[0]
        plt.xlabel(axis_title1)
    if len(text['axis_titles']) > 1:
        axis_title2 = text['axis_titles'][1].split('.')[0]
        plt.ylabel(axis_title2)

    #Only keep first 3 sentences
    caption = text['caption'].replace('\n','')
    caption = '.'.join(caption.split('.')[:MAX_SENT_CAPTION])[:MAX_TEXT_CAPTION]

    s = SINGLE_PLOT_TEXT_LOC
    plt.figtext(s[0], s[1], caption, wrap=True, horizontalalignment='center')

    plt.tight_layout(rect=SINGLE_PLOT_TIGHT_LAYOUT_REC)
    plt.savefig(fname=f_name)
    plt.close()


def _create_scatter_double(x_data, xhat_data, text, f_name):

    x_values = x_data['values']['x']
    y_values = x_data['values']['y']
    plt.subplot(2, 1, 1)  

    if len(text['axis_titles']) > 0:
        axis_title1 = text['axis_titles'][0].split('.')[0]
        plt.xlabel(axis_title1)
    if len(text['axis_titles']) > 1:
        axis_title2 = text['axis_titles'][1].split('.')[0]
        plt.ylabel(axis_title2)

    for idx, (x,y, label) in enumerate(zip(x_values, y_values, text['series_name'])):
        if 'unnamed' in label:
            label = None
        plt.scatter(x, y, cmap = 'viridis', label=label)

    if len(x_values) > 1:
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")

    x_values = xhat_data['values']['x']
    y_values = xhat_data['values']['y']

    plt.subplot(2, 1, 2)
    #plt.title('Reconstruction')
    if len(text['axis_titles']) > 0:
        axis_title1 = text['axis_titles'][0].split('.')[0]
        plt.xlabel(axis_title1)
    if len(text['axis_titles']) > 1:
        axis_title2 = text['axis_titles'][1].split('.')[0]
        plt.ylabel(axis_title2)
    
    for idx, (x,y, label) in enumerate(zip(x_values, y_values, text['series_name'])):
        if 'unnamed' in label:
            label = None
        plt.scatter(x, y, cmap = 'viridis', label=label)

    if len(x_values) > 1:
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    
    #Add caption
    #plt.subplots_adjust(bottom=0.2) 
    plt.tight_layout(rect=[0, 0.0, 1.0, 0.8])

    #Only keep first 3 sentences
    caption = text['caption'].replace('\n','')
    caption = '.'.join(caption.split('.')[:MAX_SENT_CAPTION])[:MAX_TEXT_CAPTION]
    plt.figtext(0.5, 0.9, caption, wrap=True, horizontalalignment='center')

    # space between the plots
    plt.savefig(fname=f_name)
    plt.close()

def create_boxplot(x_data, text, f_name, xhat_data=None):

    if xhat_data is not None:
        create_boxplot_double(x_data=x_data, xhat_data=xhat_data, text=text, f_name=f_name)
    else:
        _create_boxplot_single(x_data=x_data, text=text, f_name=f_name)


def _create_boxplot_single(x_data, text, f_name):

    fig = plt.figure()
    plt.figure(facecolor='white')

    text['categorical'] = [s for s in text['categorical'] if 'unnamed'  not in s]

    recon_values = x_data['values']
    clip_val = min(len(text['categorical']), len(recon_values[0]))

    #Manually remove
    if clip_val <= 1:
        return

    plt.boxplot(recon_values[0][:clip_val], labels=text['categorical'][:clip_val])

    if len(text['axis_titles']) > 0:
        axis_title = text['axis_titles'][0].split('.')[0]
        plt.ylabel(axis_title)


    #Only keep first 3 sentences
    caption = text['caption'].replace('\n','')
    caption = '.'.join(caption.split('.')[:MAX_SENT_CAPTION])[:MAX_TEXT_CAPTION]

    s = SINGLE_PLOT_TEXT_LOC
    plt.figtext(s[0], s[1], caption, wrap=True, horizontalalignment='center')
    plt.tight_layout(rect=SINGLE_PLOT_TIGHT_LAYOUT_REC)

    # space between the plots
    plt.savefig(fname=f_name)
    plt.close()

def create_boxplot_double(x_data, xhat_data, text, f_name):
    
    # Create original plot
    original_values = x_data['values']
    plt.subplot(2, 1, 1)  # row 1, column 2, count 1
    #plt.title('Original')

    if len(text['axis_titles']) > 0:
        plt.ylabel(text['axis_titles'][0])
    plt.boxplot(original_values[0], labels=text['categorical'])

    # Create reconstruction
    plt.subplot(2, 1, 2)
    #plt.title('Reconstruction')
    if len(text['axis_titles']) > 0:
        axis_title = text['axis_titles'][0].split('.')[0]
        plt.ylabel(axis_title)
        plt.ylabel(axis_title)


    recon_values = xhat_data['values']
    clip_val = min(len(text['categorical']), len(recon_values[0]))

    plt.boxplot(recon_values[0][:clip_val], labels=text['categorical'][:clip_val])

    #Add caption
    #plt.subplots_adjust(bottom=0.2) 
    plt.tight_layout(rect=[0, 0.0, 1.0, 0.8])

    #Only keep first 3 sentences
    caption = text['caption'].replace('\n','')
    caption = '.'.join(caption.split('.')[:MAX_SENT_CAPTION])[:MAX_TEXT_CAPTION]
    plt.figtext(0.5, 0.9, caption, wrap=True, horizontalalignment='center')

    # space between the plots
    plt.savefig(fname=f_name)
    plt.close()

def create_recon_plots(x, x_hat, text_data, step, epoch_dir):

    if 'chart_data' in x:
        x = x['chart_data']
    
    if 'chart_data' in x_hat:
        x_hat  = x_hat['chart_data']

    x_chart_data    = prepare_mpl(x)
    xhat_chart_data = prepare_mpl(x_hat)

    for idx, (x_data, xhat_data, text) in enumerate(zip(x_chart_data, xhat_chart_data, text_data)):
        assert x_data['head_type'] == xhat_data['head_type']

        f_name = os.path.join(epoch_dir, f"{step}-{idx}-{xhat_data['head_type']}.png")
        if x_data['head_type'] == 'categorical':
            create_bar_chart(x_data=x_data, xhat_data=xhat_data, text=text, f_name=f_name)
        elif x_data['head_type'] == 'point':
            create_scatter(x_data=x_data, xhat_data=xhat_data, text=text, f_name=f_name)
        else:
            create_boxplot(x_data=x_data, xhat_data=xhat_data, text=text, f_name=f_name)
            

def create_single_plot(x, text_data, save_dir, step):

    if 'chart_data' in x:
      x = x['chart_data']
    
    x_data_mpl = prepare_mpl(x)

    for idx, x_data in enumerate(x_data_mpl):
      f_name = os.path.join(save_dir, f"{step}-{idx}-{x_data['head_type']}.png")

      text = text_data[idx]

      if x_data['head_type'] == 'categorical':
          create_bar_chart(x_data, text, f_name)
      elif x_data['head_type'] == 'point':
          create_scatter(x_data, text, f_name)
      elif x_data['head_type'] == 'boxplot':
          create_boxplot(x_data, text, f_name)
      else:
        raise ValueError(f"Invalid chart type given: {x_data['head_type']}")
      plt.close()