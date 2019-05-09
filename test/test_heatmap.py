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

class TestHeatmap(unittest.TestCase):


    def test_heatmap(self):
        """ Parse given sources and create a heat map as PDF file"""

        docs = {}
        docs['a'] = "ab ba bc ff"
        docs['b'] = "ab ba gg bc"
        docs['c'] = "ab dd bc oo"

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
        filename="heatmap_test.png"
        fig.savefig(filename)
        print("Report created: %s" % filename)
