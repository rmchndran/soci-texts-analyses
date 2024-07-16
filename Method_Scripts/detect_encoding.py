import chardet

def detect_encoding(filepath):
    with open(filepath, 'rb') as input:
        D = chardet.universaldetector.UniversalDetector()
        for i in input:
            D.feed(i)
            if D.done:
                break
        D.close()
        print(D.result)
        return D.result['encoding']

