file = open("train.txt", "a+", encoding="utf-8")

with open("example.train", "r", encoding="utf-8") as f:
    sentences = f.read().strip().split("\n\n")
    for i in range(len(sentences)):
        sentence = sentences[i].split("\n")
        contents = []
        labels = []
        for sent in sentence:
            contents.extend(sent.strip().split()[0])
            labels.append(sent.strip().split()[1])
        data = " ".join(contents) + "\t" + " ".join(labels) + "\n"
        file.write(data)
file.close()