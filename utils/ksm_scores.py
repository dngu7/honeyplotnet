import torch
import yake


SIM_PAIRS = [
    ('all_text_kw', 'captions_kw'),
    ('all_text_kw', 'axis_words'),
    ('all_text_kw', 'categorical_words'),
    ('all_text_kw', 'series_words'),
    ('all_text_kw', 'categ_and_series_words'),
    ('captions_kw', 'axis_words'),
    ('captions_kw', 'categorical_words'),
    ('captions_kw', 'series_words'),
    ('captions_kw', 'categ_and_series_words'),
    ('series_words', 'categorical_words')
]

def yake_text(text_str, max_ngram_size=3, dedup_thres=0.9, num_kw=20):
    kw_extractor = yake.KeywordExtractor(
        n=max_ngram_size, dedupLim=dedup_thres, top=num_kw)
    return kw_extractor.extract_keywords(text_str)
    
def tokenize(tokenizer, text, max_source_len=1024):
    inputs = tokenizer(
        text, max_length=max_source_len, 
        padding="max_length", truncation=True, return_tensors="pt")
    return inputs

def embed_and_encode(tokens, model, device):
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(tokens.input_ids.to(device))
        encoder = model.get_encoder()
        output = encoder(inputs_embeds=embeddings, attention_mask=tokens.attention_mask.to(device)).last_hidden_state
    return output

def tokenize_keywords(d, tokenizer, model, max_source_len, device):
    extract_list = ['captions_kw','all_text_kw','categorical_words','series_words','axis_words']
    embeddings = {}
    for name in extract_list:
        if len(d[name]):
            tok = tokenize(tokenizer, d[name], max_source_len=max_source_len)
            emb = embed_and_encode(tok, model, device)
            #Average through middle dimension as per https://arxiv.org/pdf/2108.08877.pdf
            emb = emb.mean(1)
            embeddings[name] = emb
    
    #Create combine cateogircal and series if exist
    if len(d['categorical_words']) and len(d['series_words']):
        combined = d['categorical_words'] + d['series_words']
        tok = tokenize(tokenizer,combined, max_source_len=max_source_len)
        emb = embed_and_encode(tok, model, device)
        
        #Average through middle dimension as per https://arxiv.org/pdf/2108.08877.pdf
        emb = emb.mean(1)
        embeddings['categ_and_series_words'] = emb

    return embeddings

def calc_similarity(emb1, emb2, dim=-1, eps=1e-6):
    '''Calculates between all combinations of pairs'''
    cos = torch.nn.CosineSimilarity(dim=dim, eps=eps)
    
    if emb1.size(0) > emb2.size(0):
        repeat_emb, base_emb = emb1, emb2
        bsz = emb2.size(0)
    else:
        repeat_emb, base_emb = emb2, emb1
        bsz = emb1.size(0)

    similarity_matrix = []
    for bidx in range(bsz):
        repeated_emb = repeat_emb[bidx,:].unsqueeze(0).repeat(bsz, 1)
        sim = cos(repeated_emb, base_emb)
        similarity_matrix.append(sim)
    similarity_matrix = torch.stack(similarity_matrix, dim=0)
    return similarity_matrix


def calc_similarity_between_all_pairs(embeddings, sim_pairs, to_cpu=True):
    '''
    Computes similarity between pairs of text
    '''
    
    sim_pair_dict = dict()
    for pair1, pair2 in sim_pairs:
        if embeddings.get(pair1) is not None and embeddings.get(pair2) is not None:
            emb1 = embeddings[pair1]
            emb2 = embeddings[pair2]
            sim_score = calc_similarity(emb1, emb2)
            sim_score = sim_score.mean()
            if to_cpu:
                sim_score = sim_score.detach().cpu().item()

            sim_pair_dict[(pair1, pair2)] = sim_score

    return sim_pair_dict

def compute_similarity_scores_for_dataset(sim_pairs, dataset, tokenizer, model, device, max_source_len=128):
    # Max source len can be low because its only keywords
    model = model.to(device)

    sim_container = {}
    for d in dataset:
        embeddings = tokenize_keywords(d, tokenizer, model, max_source_len, device)
        similarity_scores = calc_similarity_between_all_pairs(embeddings, sim_pairs)

        for sim_pair, sim_score in similarity_scores.items():
            if sim_pair not in sim_container:
                sim_container[sim_pair] = []
            sim_container[sim_pair] += [sim_score]
    
    #Take average across each pair
    for sim_pair in list(sim_container.keys()):
        sim_container[sim_pair] = (sum(sim_container[sim_pair]) / len(sim_container[sim_pair]), len(sim_container[sim_pair]))

    return sim_container
