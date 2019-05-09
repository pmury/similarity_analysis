# -*- coding: utf-8 -*-

# import os
# import sys
# import pprint
import random
import unittest

import urllib.request
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import numpy as np
import nltk

import tensorflow as tf
import tensorflow_hub as hub

class TestSimilarity(unittest.TestCase):
    """ This is just an example to play around with, inspired by a StackOverflow answer post. """
    # https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents/55732255#comment98596607_55732255
    # For professional applications using TF, please refer to the sources above; also you might want to use another similarty algorithm.
    # https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html """
    # https://cloud.google.com/solutions/machine-learning/analyzing-text-semantic-similarity-using-tensorflow-and-cloud-dataflow


    def test_similarity(self):
        """ Parse given sources and create a heat map as PDF file"""

        docs = {}
        docs['shk_romeo_juliet'] = self.url2words('http://shakespeare.mit.edu/romeo_juliet/full.html')
        docs['shk_macbeth'] = self.url2words('http://shakespeare.mit.edu/macbeth/full.html')
        docs['shk_hamlet'] = self.url2words('http://shakespeare.mit.edu/hamlet/full.html')
        docs['shk_othello'] = self.url2words('http://shakespeare.mit.edu/othello/full.html')
        docs['shk_midsummer'] = self.url2words('http://shakespeare.mit.edu/midsummer/full.html')
        #docs['hmr_odyssey'] = self.url2words('http://classics.mit.edu/Homer/odyssey.1.i.html')
        docs['shk_hamlet_shuffled'] = self.shuffle(self.url2words('http://shakespeare.mit.edu/hamlet/full.html'))
        #docs['caesae_gallic_war'] = self.url2words('http://www.perseus.tufts.edu/hopper/text?doc=Perseus:text:1999.02.0001')
        docs['tolstoy_war_and_peace'] = self.url2words('https://archive.org/stream/warandpeace030164mbp/warandpeace030164mbp_djvu.txt')
        docs['dickens_oliver_twist'] = self.url2words('https://ia802807.us.archive.org/18/items/olivertwist56586gut/56586-0.txt')

        tf_input = list(docs.values())
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed"

        # Import the Universal Sentence Encoder's TF Hub module
        embed = hub.Module(module_url)

        similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
        similarity_message_encodings = embed(similarity_input_placeholder)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            message_embeddings_ = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: tf_input})

            corr = np.inner(message_embeddings_, message_embeddings_)
            print(corr)
            self.heatmap(docs.keys(), docs.keys(), corr)




    def url2words(self, url):
        """Skip words consisting of 1 or 2 characters and take only first N words"""
        text = self.url2text(url)
        n = 10000

        tokens = nltk.tokenize.WordPunctTokenizer().tokenize(text)
        tokens2 = []
        for token in tokens:
            if len(tokens2) == n:
                break
            if len(token)>2:
                tokens2.append(token)

        print("%s words in url %s." % (len(tokens2), url))
        text = " ".join(tokens2)

        return text

    def shuffle(self,text):
        tokens = text.split(" ")
        random.seed(4)
        random.shuffle(tokens)
        random.shuffle(tokens)

        return " ".join(tokens)

    def url2text(self, url):
        """Strip plain text from a given URL"""
        fp = urllib.request.urlopen(url)
        mybytes = fp.read()
        html = mybytes.decode("utf8")
        fp.close()

        soup = BeautifulSoup(html, "html.parser")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    def heatmap(self,x_labels, y_labels, values):
        """ Create a heatmap of correlation matrix"""
        fig, ax = plt.subplots()


        # heatmap = ax.pcolor(values)
        # cbar = plt.colorbar(heatmap)

        im = ax.imshow(values)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=4,
             rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), fontsize=4)

        # Loop over data dimensions and create text annotations.
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, "%.2f"%values[i, j], ha="center", va="center", color="w", fontsize=6)

        fig.tight_layout()

        plt.title("Similarity matrix")
        filename="literature_similarity.png"
        fig.savefig(filename)
        print("Report created: %s" % filename)
