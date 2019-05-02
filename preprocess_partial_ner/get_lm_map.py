import pickle
ct_dataset = pickle.load(open('./lm/ner_dataset.pk', 'rb'))
flm_map = ct_dataset['flm_map']
blm_map = ct_dataset['blm_map']
with open("./lm/lm_maps.pk", "wb") as f:
	pickle.dump({'flm_map': flm_map, 'blm_map': blm_map}, f)