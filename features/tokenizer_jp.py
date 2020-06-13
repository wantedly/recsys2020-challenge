import re
import urllib3
import MeCab
import unicodedata
from itertools import product


def urlopen(url):
    http = urllib3.PoolManager()
    res = http.request('GET', url)
    words = res.data.decode('utf-8').split()
    return words


def make_stop_words():
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/'
    url_ja = url + 'Japanese.txt'
    url_en = url + 'English.txt'

    stop_words_ja = urlopen(url_ja)
    stop_words_en = urlopen(url_en)
    stop_words = stop_words_ja + stop_words_en
    stop_words += [chr(i) for i in range(12353, 12436)]  # ひらがな1文字
    stop_words += [chr(i) + chr(j) for i, j in product(range(12353, 12436), range(12353, 12436))]  # ひらがな2文字

    return set(stop_words)


class Tokenizer:
    def __init__(self):
        mecab = MeCab.Tagger(' -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
        mecab.parse('')
        self.mecab = mecab
        self.parts = {'名詞', '動詞', '形容詞'}
        self.stop_words = make_stop_words()

    def extract_parts(self, text):
        mecab = self.mecab
        parts = self.parts
        stop_words = self.stop_words
        keywords = []
        node = mecab.parseToNode(text).next
        while node:
            if node.feature.split(',')[0] in parts and node.surface not in stop_words:
                keywords.append(node.surface)
            node = node.next
        return keywords

    def tokenize(self, text):
        text = unicodedata.normalize('NFKC', text)  # カナ全角、アルファベット半角にする
        text = re.sub(r'https?://[\w/:%#$&?()~.=+\-…]+', '', text)  # URLを削る
        text = re.sub(r'[0-9]+年|[0-9]+月|[0-9]+日|[0-9]+時|[0-9]+分', '', text)  # 日時を削る
        text = re.sub(r'[!-@[-`{-~]', '', text)  # 記号 + 半角数字 を削る
        text = text.lower()  # アルファベットを小文字にする

        return self.extract_parts(text)
