import argparse
from typing import Dict, List
import tqdm
from collections import defaultdict
import math
import json
import re
import os

from dependency_parsing import *

RULE = r'[\u4E00-\u9FA5]'


# def calculate_mean_variance(data: List[List[Dict]]) -> Dict:
#     """ For each bigrams in data, calculate the mean and variance
#         with relevant distance of token position.
#     """
#     rtStat = {} # result

#     for sent in tqdm.tqdm(data, desc="Calculating the mean&variance"):
#         if len(sent) > 1:
#             prevToken, prevPos = sent[0]["form"], sent[0]["upos"]
#         for i in range(1, len(sent)):
#             curToken, curPos = sent[i]["form"], sent[i]["upos"]




def calculate_frequencies(data: List[List[Dict]]) -> Dict:
    """
      Return: A dict for each element
              with (((token1, pos1), (token2, pos2)) : frequency)
    """
    rtStat = defaultdict(int)

    for sent in tqdm.tqdm(data, desc="Calculating the frequencies"):
        if len(sent) > 2:
            prevToken, prevPos = sent[0]["form"], sent[0]["upos"]
            for i in range(1, len(sent)):
                curToken, curPos = sent[i]["form"], sent[i]["upos"]
                if re.match(RULE, curToken):
                    # check
                    if re.match(RULE, prevToken):
                        k = ((prevToken,prevPos), (curToken,curPos))
                        rtStat[k] += 1
                        prevToken, prevPos = curToken, curPos
                prevToken, prevPos = curToken, curPos

    return rtStat

def calculate_t_test(data: List[List[Dict]]) -> Dict:
    """
      Return: A dict for each element with
              (((token1, pos1), (token2, pos2)) : chi-square)
    """
    startTokenBigramsFreq = defaultdict(int)
    endTokenBigramsFreq = defaultdict(int)
    bigramsFreq = defaultdict(int)
    rtTScores = {}

    for sent in tqdm.tqdm(data, desc="Couting for the data bigrams"):
        if len(sent) > 2:
            prevToken, prevPos = sent[0]["form"], sent[0]["upos"]
            for i in range(1, len(sent)):
                curToken, curPos = sent[i]["form"], sent[i]["upos"]
                if  re.match(RULE, curToken):
                    # check
                    if re.match(RULE, prevToken):
                        w1 = (prevToken,prevPos)
                        w2 = (curToken,curPos)
                        w1_w2 = ((prevToken,prevPos), (curToken,curPos))
                        # couting
                        startTokenBigramsFreq[w1] += 1
                        endTokenBigramsFreq[w2] += 1
                        bigramsFreq[w1_w2] += 1
                prevToken, prevPos = curToken, curPos
    N = sum(bigramsFreq.values())
    for bigram in tqdm.tqdm(bigramsFreq, desc="Calculating for t-test"):
        w1, w2 = bigram[0], bigram[1]
        c_w1 = startTokenBigramsFreq[w1]
        c_w2 = endTokenBigramsFreq[w2]
        c_w1_w2 = bigramsFreq[bigram]
        p_w1_w2, p_w1, p_w2 = c_w1_w2/N, c_w1/N, c_w2/N
        t = (p_w1_w2 - (p_w1 * p_w2)) / \
            math.sqrt(p_w1_w2/N)
        rtTScores[bigram] = (t, p_w1, p_w2, p_w1_w2)
    return rtTScores


def calcullate_chi_square(data: List[List[Dict]]) -> Dict:
    """
      Return: A dict for each element with
              (((token1, pos1), (token2, pos2)) : chi-square)
    """
    startTokenBigramsFreq = defaultdict(int)
    endTokenBigramsFreq = defaultdict(int)
    bigramsFreq = defaultdict(int)
    rtChiSquares = {}

    for sent in tqdm.tqdm(data, desc="Couting for the data bigrams"):
        if len(sent) > 2:
            prevToken, prevPos = sent[0]["form"], sent[0]["upos"]
            for i in range(1, len(sent)):
                curToken, curPos = sent[i]["form"], sent[i]["upos"]
                if  re.match(RULE, curToken) and curToken!="的":
                    # check
                    if re.match(RULE, prevToken) and prevToken!="的":
                        w1 = (prevToken,prevPos)
                        w2 = (curToken,curPos)
                        w1_w2 = ((prevToken,prevPos), (curToken,curPos))
                        # couting
                        startTokenBigramsFreq[w1] += 1
                        endTokenBigramsFreq[w2] += 1
                        bigramsFreq[w1_w2] += 1
                prevToken, prevPos = curToken, curPos

    N = sum(bigramsFreq.values())
    for bigram in tqdm.tqdm(bigramsFreq, desc="Calculating for chi-square"):
        w1, w2 = bigram[0], bigram[1]
        c_w1 = startTokenBigramsFreq[w1]
        c_w2 = endTokenBigramsFreq[w2]
        c_w1_w2 = bigramsFreq[bigram]
        c_w1_Nw2 = c_w1 - c_w1_w2
        c_Nw1_w2 = c_w2 - c_w1_w2
        c_Nw1_Nw2 = N - c_w1 - c_w2 - c_w1_w2
        p = c_w1 / N
        # change
        c_w1_w2_P = c_w2 * p
        c_Nw1_w2_P = c_w2 * (1-p)
        c_w1_Nw2_P = (N-c_w2) * p
        c_Nw1_Nw2_P = (N-c_w2) * (1-p)
        # calculate
        # rt = math.pow((c_w1_w2_P - c_w1_w2), 2)/c_w1_w2_P \
        #     + math.pow((c_Nw1_w2_P - c_Nw1_w2), 2)/c_Nw1_w2_P \
        #     + math.pow((c_w1_Nw2_P - c_w1_Nw2), 2)/c_w1_Nw2_P \
        #     + math.pow((c_Nw1_Nw2_P - c_Nw1_Nw2), 2)/c_Nw1_Nw2_P

        rt = N*math.pow(c_w1_w2*c_Nw1_Nw2 - c_Nw1_w2*c_w1_Nw2, 2)/ \
            (c_w1_w2+c_Nw1_w2) * (c_w1_w2+c_w1_Nw2) * (c_Nw1_w2+c_Nw1_Nw2) * (c_w1_Nw2+c_Nw1_Nw2)

        rtChiSquares[bigram] = rt
    return rtChiSquares





def main(alg:str, corpusPath:str, valPath:str, topK:int, savePath:str):
    """
      Args:
        - alg: The algorithm which will run in this task.
    """
    with open(corpusPath, "r") as f:
        trainData = json.load(f)

    if alg == "frequency":
        rt = calculate_frequencies(trainData)
        # sort
        sortedRt = sorted(rt.items(), key=lambda x:x[1], reverse=True)
    elif alg == "chi-square":
        rt = calcullate_chi_square(trainData)
        sortedRt = sorted(rt.items(), key=lambda x:x[1], reverse=True)
    elif alg == "t-test":
        rt = calculate_t_test(trainData)
        sortedRt = sorted(rt.items(), key=lambda x:x[1][0], reverse=True)
    else:
        raise KeyError

    

    # check with dependency parsing
    valBigrams = conll_dataloader(valPath)

    # save couting results
    with open(savePath, "w") as f:
        f.write("The selected algorithm is {}\n\n".format(alg))
        k = 0
        for bigram, value in sortedRt:
            if k > topK: # found
                break
            if bigram in valBigrams:
                k += 1
                relCounter = valBigrams[bigram]
                dependFreqs = [(k,v) for k,v in relCounter.items()]
                f.write("{}\t\t\t{}\t\t{}\n".format(bigram, value, dependFreqs))
            # else:
            #     f.write("{}\t\t\t{}\n".format(bigram, value))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="../corpus.json")
    parser.add_argument("--val_path", type=str, default="../ud.conllu")
    # parser.add_argument("--algorithm", type=str, default="chi-square")
    parser.add_argument("--algorithm", type=str, default="frequency")
    # parser.add_argument("--algorithm", type=str, default="t-test")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="")
    args = parser.parse_args()

    savePath = os.path.join(args.save_dir, args.algorithm + ".txt")

    main(args.algorithm, args.corpus_path, args.val_path, args.top_k, savePath)




