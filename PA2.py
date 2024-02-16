import re
import math
from functools import reduce
import os
from itertools import chain
from operator import itemgetter

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_texts_from_directory(directory_path):
    get_full_path = lambda filename: os.path.join(directory_path, filename)
    files = map(get_full_path, filter(lambda f: f.endswith('.txt'), os.listdir(directory_path)))

    texts = map(read_file, files)

    return tuple(texts)

def process_text(text):
    def valid_char(c):
        return c if c.isalnum() or c.isspace() else ' '

    mapped_chars = map(valid_char, text)

    cleaned_text = reduce(lambda a, b: a + b, mapped_chars)

    cleaned_text = cleaned_text.lower()

    def extract_words(acc, char):
        word, words = acc
        if char.isalnum():
            return word + char, words
        elif word:
            return '', words + [word]
        else:
            return '', words

    word, words = reduce(extract_words, cleaned_text, ('', []))
    if word:
        words.append(word)

    words = filter(lambda word: reduce(lambda length, _: length + 1, word, 0) >= 3, words)

    return tuple(words)


def calculate_tf(words):

    word_count = reduce(lambda count, word: count.update({word: count.get(word, 0) + 1}) or count, words, {})

    total_words = reduce(lambda total, _: total + 1, words, 0)

    tf_values = {}
    tf_values.update(map(lambda word_count_pair: (word_count_pair[0], word_count_pair[1] / total_words), word_count.items()))
    return tf_values

def calculate_tf_for_texts(texts):

    tf_values_list = map(lambda text: calculate_tf(process_text(text)), texts)


    combined_tf_values = chain.from_iterable(map(lambda tf: tf.items(), tf_values_list))

    return combined_tf_values

def calculate_idf(texts):

    processed_texts = map(lambda text: set(process_text(text)), texts)


    document_count = reduce(
        lambda count, text_set: reduce(
            lambda inner_count, word: inner_count.update({word: inner_count.get(word, 0) + 1}) or inner_count,
            text_set,
            count
        ),
        processed_texts,
        {}
    )


    total_documents = reduce(lambda count, _: count + 1, texts, 0)

    def accumulate_idf_values(accumulated_dict, item):
        word, count = item
        accumulated_dict[word] = math.log(total_documents / count)
        return accumulated_dict


    idf_values = reduce(accumulate_idf_values, document_count.items(), {})

    return idf_values

def calculate_tf_idf(tf_values, idf_values):
    def calculate_single_tf_idf(tf_value_pair):
        word, tf_value = tf_value_pair
        idf_value = idf_values.get(word, 0)
        return word, tf_value * idf_value

    def accumulate_tf_idf_values(accumulated_list, tf_idf_value):
        accumulated_list.append(tf_idf_value)
        return accumulated_list

    return reduce(accumulate_tf_idf_values, map(calculate_single_tf_idf, tf_values), [])

def main():
    directory_path = 'directory'


    texts = load_texts_from_directory(directory_path)


    processed_texts = tuple(map(process_text, texts))
    tf_values_for_texts = tuple(map(calculate_tf, processed_texts))

    idf_values = calculate_idf(texts)


    tf_idf_values = [calculate_tf_idf(tf.items(), idf_values) for tf in tf_values_for_texts]

    all_tf_idf_tuples = []


    for i, doc_tf_idf in enumerate(tf_idf_values):

        all_tf_idf_tuples.extend(map(lambda item: (item[0], i + 1, round(item[1], 2)), doc_tf_idf))

        print(f"Dokument {i + 1}:")


        sorted_tf_values = sorted(map(lambda item: (item[0], round(item[1], 2)), tf_values_for_texts[i].items()), key=itemgetter(1), reverse=True)
        print("TF vrednosti:", dict(sorted_tf_values))


        current_document_words = set(tf_values_for_texts[i].keys())
        sorted_idf_values = sorted(map(lambda word: (word, round(idf_values[word], 2)), current_document_words), key=itemgetter(1), reverse=True)
        print("IDF vrednosti:", dict(sorted_idf_values))


        sorted_tf_idf_values = sorted(map(lambda item: (item[0], round(item[1], 2)), doc_tf_idf), key=itemgetter(1), reverse=True)
        print("TF-IDF vrednosti:", sorted_tf_idf_values)

        print("\n")


    print("Sortirana lista torki (reč, identifikator fajla, vrednost):")
    all_tf_idf_tuples.sort(key=lambda x: (x[1], -x[2]))
    for t in all_tf_idf_tuples:
        print(t)



if __name__ == '__main__':
    main()
