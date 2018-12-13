# -*- coding: utf-8 -*-
import operator
import time
start_time = time.time()

# ALL DATA SEBANYAK 10,000 RECORDS
# DATA TRAIN DIPAKE 8,000 RECORDS
# DATA TEST DIPKAE 2,000 RECORD

# ======= TRAIN DATA =========
def read_file_init_table(fname):
    tag_count = {}
    tag_count['<start>'] = 0
    word_tag = {}
    tag_trans = {}
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip().lower() for x in content]
    idx_line = 0
    is_first_word = 0
    while idx_line < len(content):
        prev_tag = '<start>'
        while not content[idx_line].startswith('</kalimat'):
            if  not content[idx_line].startswith('<kalimat'):
                content_part = content[idx_line].split('\t')
                if content_part[1] in tag_count:
                    tag_count[content_part[1]] += 1
                else:
                    tag_count[content_part[1]] = 1
                    
                current_word_tag = content_part[0]+'|'+content_part[1]
                if current_word_tag in word_tag:
                    word_tag[current_word_tag] += 1
                else:    
                    word_tag[current_word_tag] = 1
                    
                if is_first_word == 1:
                    current_tag_trans = '<start>,'+content_part[1]
                    is_first_word = 0
                else:
                    current_tag_trans = prev_tag+','+content_part[1]
                    
                if current_tag_trans in tag_trans:
                    tag_trans[current_tag_trans] += 1
                else:
                    tag_trans[current_tag_trans] = 1                    
                prev_tag = content_part[1]   
            else:
                tag_count['<start>'] += 1
                is_first_word = 1
            idx_line = idx_line + 1
        idx_line = idx_line+1
    return tag_count, word_tag, tag_trans

tag_count, word_tag, tag_trans = read_file_init_table('./code/Indonesian_tag.txt')

#======= TEST DATA =========
def create_data_test(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip().lower() for x in content]
    idx_line = 0
    arr_tags = []
    arr_words = []
    while idx_line < len(content):
        while not content[idx_line].startswith('</kalimat'):
            if  not content[idx_line].startswith('<kalimat'):
                content_part = content[idx_line].split('\t')
                arr_tags.append(content_part[1])
                arr_words.append(content_part[0])
            idx_line = idx_line + 1
        idx_line = idx_line+1
    return arr_words, arr_tags

arr_words, arr_tags = create_data_test('./code/Indonesian_test.txt')
#print(arr_words)

#======== TABLE EMISSION =========
def create_emission_prob_table(word_tag, tag_count):
    emission_prob = {}
    for word_tag_entry in word_tag.keys():
        word_tag_split = word_tag_entry.split('|')
        current_word = word_tag_split[0]
        current_tag = word_tag_split[1]
        emission_key = current_word+','+current_tag
        emission_prob[emission_key] = word_tag[word_tag_entry]/tag_count[current_tag]    
    return emission_prob

emission_prob = create_emission_prob_table(word_tag, tag_count)

#======== BASELINE METHOD =========
print('============================================')
def baseline(arr_words, emission_prob, tag_count, word_tag):
    all_word = []
    tag_max = max(tag_count.items(), key=operator.itemgetter(1))[0]
    count_max = max(zip(tag_count.values()))
    sentence_words = arr_words
    tag_sequence = []
    for o in word_tag.keys():
        all_word.append(o.split('|')[0].lower())
    for i, token in enumerate(sentence_words):
        token = token.lower()
        max_prob = {}
        for tag in tag_count:
            if token+','+tag in emission_prob:
                max_prob[token+','+tag] = emission_prob[token+','+tag]
        if token.lower() not in all_word:
            max_prob[token+','+tag_max] = 1/count_max[0]
        key_max = max(max_prob.items(), key=operator.itemgetter(1))[0]
        split_key = key_max.split(',')
        tag_sequence.append(split_key[1])
    return tag_sequence

#sentence = 'Tiga orang yang berdiri di dekat kejadian , termasuk dua orang anak , di antara yang menderita cedera , ujar -nya menambahkan . '
sequence_tag = baseline(arr_words, emission_prob, tag_count, word_tag)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(arr_tags, sequence_tag)

print("Hasil Accuracy : ",accuracy)
#print("-- True Labels --")
#print(arr_tags)
#print("-- Predict Labels --")
#print(sequence_tag)

print("Running program selama ", "--- %s seconds ---" % (time.time() - start_time))
