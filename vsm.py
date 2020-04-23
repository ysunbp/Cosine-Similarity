import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import math, glob
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
import csv
import os

'''Map document titles to document ids'''
documents = {}
'''A running counter for assigning numerical IDs to documents'''
docid_counter = 1
'''The document-term frequencies'''
the_index = dict()
'''get the stop words set, if want to remove stop words from the documents, uncomment the line below'''
stop_words = set(stopwords.words('english'))
'''if do not want to remove stop words from the documents, uncomment the line below'''


# stop_words = []

def initialize_globs():
    global documents, docid_counter, the_index
    '''Map document titles to document ids'''
    documents = {}
    '''A running counter for assigning numerical IDs to documents'''
    docid_counter = 1
    '''The document-term frequencies'''
    the_index = dict()


def add_document(doc):
    '''
    Add a document to the inverted index. Returns the document's ID
    '''
    global documents, docid_counter, the_index
    # do not re-add the same document.
    if doc in documents.values():
        return
    docid = docid_counter
    documents[docid] = doc
    docid_counter += 1
    # print("Adding document %s to inverted index with document ID %d" % (doc, docid))

    stemmer = PorterStemmer()
    doc_content = open(doc, encoding='gbk', errors='ignore')
    terms = word_tokenize(doc_content.read())
    for term in terms:
        if term not in stop_words:
            term = stemmer.stem(term)
            # Collect term frequencies 'tf' in a dict of dicts, 'df_t' is implicitly
            # stored as len(the_index[t].keys())
            # the_index.setdefault(term,defaultdict(int))[docid] += 1
            if term not in the_index.keys():
                the_index[term] = {docid: 1}
            elif docid not in the_index[term]:
                the_index[term][docid] = 1
            else:
                the_index[term][docid] += 1


def tf(term, docid):
    '''
    Calculate term frequency for term in docid. Return 0 if term not in index,
    or term does not appear in document.
    '''

    if term not in list(the_index.keys()):
        return 0
    if docid not in the_index[term]:
        return 0
    return the_index[term][docid]


def df(term):
    '''
    Extract frequency of term for document with id docid from index
    '''

    if term not in list(the_index.keys()):
        return 0
    return len(the_index[term].keys())


def idf(term):
    '''
    Compute idf_t for a term
    '''
    if term not in list(the_index.keys()):
        return 0
    return math.log10(len(documents.keys()) / df(term))


def tf_idf(term, docid):
    '''
    Compute tf-idf for term and docid
    '''
    return tf(term, docid) * idf(term)


def norm_cosine(docid):
    '''
    Compute cosine normalization for docid
    '''
    ninv = 0
    for term, freqs in the_index.items():
        if docid in freqs.keys():
            ninv += tf_idf(term, docid) ** 2
    return math.sqrt(ninv)


def compute_vector(d_id):
    vector_array = np.zeros((1, len(the_index.keys())))
    counter = 0
    length_d = norm_cosine(d_id)

    '''if length_d is too small, we set it to 10^-10'''
    if length_d == 0:
        length_d = 10 ** (-10)
    for word in the_index.keys():
        vector_array[0][counter] = tf_idf(word, d_id) / length_d
        counter += 1
    return vector_array


def formulate_result_matrix():
    matrix_array = []
    for doc_id in documents.keys():
        vector = compute_vector(doc_id)
        matrix_array.append(vector)
    matrix = np.concatenate(matrix_array[0:], axis=0)
    result_matrix = np.dot(matrix, matrix.transpose())
    return result_matrix


def length_of_doc(doc_id):
    length = 0
    for word, freqs in the_index.items():
        if doc_id in freqs.keys():
            length += freqs[doc_id]
    return length


def result():
    with open('../shared/BaseFileResultcsv.csv', 'w') as csv_File:
        writer = csv.writer(csv_File)
        writer.writerow(["gvkey", "fyear", "length", "Similarity"])
        row = []
        for folder in glob.glob('../shared/renamed_files/*/'):
            initialize_globs()
            dyears = []
            dlength = []
            counter = 0
            for document in glob.glob(folder + '*.txt'):
                counter += 1
                add_document(document)
                dyears.append(document[-8:-4])
                dlength.append(length_of_doc(counter))
            result_matrix = formulate_result_matrix()
            form_score = " "
            if counter > 1:
                for i in range(counter - 1):
                    row.append(folder[-7:-1])
                    row.append(dyears[i])
                    row.append(dlength[i])
                    row.append(form_score)
                    writer.writerow(row)
                    row = []
                    # print("The length of "+str(dyears[i])+" is: "+str(dlength[i]))
                    # for i in range(counter-1):
                    if int(dyears[i]) + 1 == int(dyears[i + 1]) and i < (counter - 1):
                        form_score = result_matrix[i][i + 1]
                    else:
                        form_score = " "
                row.append(folder[-7:-1])
                row.append(dyears[counter - 1])
                row.append(dlength[counter - 1])
                row.append(form_score)
                writer.writerow(row)
                form_score = " "
                row = []
                # print("The score for "+str(dyears[i])+" to "+str(dyears[i+1])+" is: "+str(result_matrix[i][i+1]))
            elif counter == 1:
                row.append(folder[-7:-1])
                row.append(dyears[0])
                row.append(dlength[0])
                row.append(form_score)
                writer.writerow(row)
                row = []
                # print("The length of "+str(dyears[0])+" is: "+str(dlength[0]))
            else:
                print("Error, no document in this folder")


