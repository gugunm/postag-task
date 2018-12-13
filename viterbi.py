#import numpy
#import io
#import re
#from itertools import permutations
import operator
import time
start_time = time.time()

# ALL DATA SEBANYAK 10,000 RECORDS
# DATA TRAIN DIPAKE 8,000 RECORDS
# DATA TEST DIPKAE 2,000 RECORD

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
                    current_tag_trans = prev_tag+'|'+content_part[1]
                    
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
# print(tag_count)
# print(word_tag)
# print(tag_trans)

#======= TEST DATA =========
def create_data_test(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip().lower() for x in content]
    idx_line = 0
    arr_tags = []
    arr_words = []
    while idx_line < len(content):
        arr_word = [] # baru
        arr_tag = [] # baru
        while not content[idx_line].startswith('</kalimat'):
            if  not content[idx_line].startswith('<kalimat'):
                content_part = content[idx_line].split('\t')
#                print(content_part)
                arr_tag.append(content_part[1])
                arr_word.append(content_part[0])
            idx_line = idx_line + 1
        idx_line = idx_line+1
        arr_words.append(arr_word)
        arr_tags.append(arr_tag)
    return arr_words, arr_tags

arr_words, arr_tags = create_data_test('./code/Indonesian_test.txt')
#print(len(arr_tags))

def create_trans_prob_table(tag_trans, tag_count):
    # print(tag_trans)
    trans_prob = {}
    for tag1 in tag_count.keys():
        for tag2 in tag_count.keys():
            #print('tag1 = ')
            #print(tag1)
            trans_idx = tag1+'|'+tag2
            #print('trans_idx = ')
            #print(trans_idx)
            if trans_idx in tag_trans:
                #print(trans_idx)
                trans_prob[trans_idx] = tag_trans[trans_idx]/tag_count[tag1]
    return trans_prob

trans_prob = create_trans_prob_table(tag_trans, tag_count)
# print(trans_prob)

def create_emission_prob_table(word_tag, tag_count):
    emission_prob = {}
    for word_tag_entry in word_tag.keys():
        # clean = re.sub(',','.',word_tag_entry) #untuk menghapus spasi yang lebih dari 1
        word_tag_split = word_tag_entry.split('|')
        current_word = word_tag_split[0]
        current_tag = word_tag_split[1]
        emission_key = current_word+'|'+current_tag
        emission_prob[emission_key] = word_tag[word_tag_entry]/tag_count[current_tag]    
    return emission_prob

emission_prob = create_emission_prob_table(word_tag, tag_count)
#print(emission_prob)

print('=======')
def viterbi(word_tag, trans_prob, emission_prob, tag_count, sentence_words):
    # untuk menampung seluruh kata yang berbeda dari dokumen latih
    all_words = []
    # looping untuk mengambil kata unik di dict word_tag
    for o in emission_prob.keys():
        katta = o.split('|')
        all_words.append(katta[0])
    # menampung tag dengan jumlah paling banyak untuk di sematkan pada kata yang tdk ada pada data latih
    tag_max = max(tag_count.items(), key=operator.itemgetter(1))[0]
    # menampung jumlah dari tag terbanyak
    count_max = max(zip(tag_count.values()))
    # menampung keseluruhan tag sequence dari beberapa kalimat test
    tag_sequences = [] 
    # untuk setiap kalimat yang ada pada data test
    for kalimat in arr_words:
        # untuk menampung setiap kata pada current kalimat
        arr_word = []
        # untuk menampung tag dari kata pada current kalimat
        tag_sequence = []
        # menambahkan <start> pada awal kalimat untuk surrent kalimat
        arr_word.append("<start>")
        # menyatukan seluruh kata dari current kalimat setelah kata <start>
        arr_word.extend(kalimat)
        # untuk setiap token pada current kalimat termasuk didalamnya ada kata <start>
        for i, word in enumerate(arr_word):
            # untuk menampung kata beserta nilai viterbinya dengan lebih dari 1 tag, agar bisa dicari tag dengan nilai maximum
            maxx = {}
            # untuk menampung tag dari kata yang memiliki tag labih dari 1
            emm_word = []
            # untuk awal kata karena diawali dengan kata <start>
            max_score = 1
            # selama i kurang dari (panjang kata -1) karena kita akan mengakses kata selanjutnya,
            # jadi posisi terakhir i ada disebelum kata terakhir dari current kalimat
            if i < len(arr_word)-1:
                # if next word ada pada data latih lakukan berikut
                if arr_word[i+1] in all_words:
                    for kunci in emission_prob.keys():
                        katt = kunci.split('|')
                        if arr_word[i+1] == katt[0]:
                            emm_word.append(katt[1])
                # if next word not exist in train document
                elif arr_word[i+1] not in all_words:
                    # diberi tag 'nn' karena tag tsb memiliki frekuensi terbanyak
                    emm_word.append(tag_max)
                # looping untuk setiap tag pada current kata jika kata memiliki lebih dari 1 tag
                for key in emm_word:
                    results = []
                    # looping untuk setiap tag pada tag_count
                    for tag in tag_count:
                        # if next word ada pada data latih lakukan berikut
                        if arr_word[i+1] in all_words:
                            if (tag+'|'+key in trans_prob) and (arr_word[i+1]+'|'+key in emission_prob):
                                # untuk menampung nilai viterbi pada current kata dan current tagnya
                                result = max_score*trans_prob[tag+'|'+key]*emission_prob[arr_word[i+1]+'|'+key]
                                results.append(result)
                        # if next word not exist in train document
                        elif arr_word[i+1] not in all_words:
                            if (tag+'|'+key in trans_prob):
                                result = max_score*trans_prob[tag+'|'+key]*(1/count_max[0])
                                results.append(result)
                    # menampung nilai maximum dari current tag dari kata yg punya lebih dari 1 tag
                    max_result = max(results)
                    maxx[key] = max_result
                #print(maxx)
                key_max = max(maxx.items(), key=operator.itemgetter(1))[0]
                tag_sequence.append(key_max)
#        print(tag_sequence)
        tag_sequences.append(tag_sequence)
    return tag_sequences

tag_squence = viterbi(word_tag, trans_prob, emission_prob, tag_count, arr_words)

from sklearn.metrics import accuracy_score

accuracy = 0
for x, predict in enumerate(tag_squence):
    accuracy += accuracy_score(arr_tags[x], tag_squence[x])
    
result = accuracy/len(arr_tags)
print("Hasil Accuracy Prediksi : ", result)
#print("-- True Labels --")
#print(arr_tags)
#print("-- Predict Labels --")
#print(tag_squence)

print("Running program selama ", "--- %s seconds ---" % (time.time() - start_time))
#print(len(tag_squence))
#'''