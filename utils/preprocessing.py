
import pandas as pd
import read_and_save as rs
import yaml
with open('../params.yaml', 'r') as f:
    params = yaml.safe_load(f)

TRAIN_FILE_PATH = params['TRAIN_FILE_PATH']
NUM_WORDS = params['NUM_WORDS']

import re

# class FilePreprossing(object):
#     def __init__(self, n):
#         # 保留前n个高频字
#         self.__n = n

#     def _read_train_file(self):
#         train_pd = pd.read_csv(TRAIN_FILE_PATH)
#         label_list = train_pd['Label'].unique().tolist()
#         # 统计文字频数
#         character_dict = defaultdict(int)
#         for comment in train_pd['Comment']:
#             comment = clean_text(comment)
#             for key, value in Counter(comment).items():
#                 character_dict[key] += value
#         # 不排序
#         sort_char_list = [(k, v) for k, v in character_dict.items()]
#         sort_char_list = sorted(
#             character_dict.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         # 排序
#         # sort_char_list = sorted(character_dict.items(), key=itemgetter(1), reverse=True)
#         print(f'total {len(character_dict)} characters.')
#         print('top 10 chars: ', sort_char_list[:10])
#         # 保留前n个文字
#         top_n_chars = [_[0] for _ in sort_char_list[:self.__n]]

#         return label_list, top_n_chars

#     def run(self):
#         label_list, top_n_chars = self._read_train_file()
#         rs.save_pickle(data=label_list, file_path='../tmp/labels.pk')
#         rs.save_pickle(data=top_n_chars, file_path='../tmp/chars.pk')


def clean_text(text: str):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[\t\r]', '', text)
    text = text.lower()
    text = re.sub(r' +', ' ', text)

    return text.strip()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
def encode(text):
    text = clean_text(text)

    return tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors="pt"
    )


class FilePreprossing(object):
    def __init__(self):
        pass

    def _read_train_file(self):
        train_pd = pd.read_csv(TRAIN_FILE_PATH)
        label_list = train_pd['Label'].unique().tolist()

        return label_list

    def run(self):
        label_list = self._read_train_file()
        rs.save_pickle(data=label_list, file_path='../tmp/labels.pk')
        

if __name__ == '__main__':
    fp = FilePreprossing()
    fp.run()
    print('Preprocessing done.')
    print(rs.read_pickle('../tmp/labels.pk'))