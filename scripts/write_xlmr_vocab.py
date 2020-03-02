from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Currently, get_vocab() is broken for xlm-roberta so have to do a little workaround

vocab = {tokenizer.convert_ids_to_tokens(i): i for i in range(250001)}
# for ii in range(4):
#     vocab[f'[unused{ii}]'] = ii
vocab[tokenizer.convert_ids_to_tokens(250004)] = 250002

tokens = list(vocab.keys())

with open('./config/archive/xlm-roberta-base/vocab.txt', 'w') as file:
    for tt in tokens:
        file.write(tt+'\n')

# now write the config
config = AutoConfig.from_pretrained('xlm-roberta-base')
config.to_json_file('./config/archive/xlm-roberta-base/bert_config.json')
