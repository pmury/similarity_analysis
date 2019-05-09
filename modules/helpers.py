

def url2words(self, url):
    """Skip words consisting of 1 or 2 characters"""
    text = self.url2text(url)
    tokens = nltk.tokenize.WordPunctTokenizer().tokenize(text)
    tokens2 = []
    for t in tokens:
        if len(t)>2:
            tokens2.append(t)
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

    plt.title("Similarity matrix of literary works")
    fig.savefig("literature_similarity.pdf")
