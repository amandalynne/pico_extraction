import os

from glob import glob

DATA_DIR = ('io_tags')

DEV_DOCS = glob(os.path.join(DATA_DIR, 'dev', '*.labels'))
TEST_DOCS = glob(os.path.join(DATA_DIR, 'test', '*.labels'))
DEV_OUT = "dev_sents"
TEST_OUT = "test_sents"

def main(docs, outpath):
    """
    This is just a helper script to collect full sentences and label sequences 
    from the dev/ and test/ data and print them to files that may later be used
    for prediction and error analysis.
    """
    for doc in docs: 
        # Iterate over CoNLL-style documents to collect tokens and labels in 
        # parallel to produce complete sentences / tag sequences.
        with open(doc) as inf:
            lines = inf.readlines()
        end_of_sent = False 
        tokens = [] 
        labels = []
        for line in lines:
            token, label = line.split()[0], line.split()[1]
            if not end_of_sent:
                tokens.append(token)
                labels.append(label)
                if token == '.':
                    end_of_sent = True
                    sentence = " ".join(tokens)
                    label_seq = " ".join(labels)
                    with open(outpath, 'a+') as outf:
                        # Append new instance to the file
                        outf.write('{')
                        outf.write('"sentence": "{0}", "labels": "{1}"'.format(sentence, label_seq))
                        outf.write('}\n')
                    tokens = []
                    labels = []
            else:
                end_of_sent = False
                tokens.append(token)
                labels.append(label)

if __name__ == "__main__":
    # Overwrite the files if they already exist
    if os.path.exists(DEV_OUT):
        os.remove(DEV_OUT)
    if os.path.exists(TEST_OUT):
        os.remove(TEST_OUT)
    main(DEV_DOCS, DEV_OUT)
    main(TEST_DOCS, TEST_OUT) 
