from collections import defaultdict, Counter
import argparse
import re



def conll_dataloader(dataPath:str):
    """Couting data for getting frequency with dependency of any bigrams.

      Return
        - bigramsDependFreq: A dict with element `((w1,pos1),(w2,pos2)):(depency, frequency)`
    """
    bigramsDependFreq = defaultdict(Counter)

    with open(dataPath, "r") as f:
        for line in f:
            if re.match(r"^#\s(sent_id|text|newdoc)", line):
                tokens = [("ROOT", "", "", "")]
                continue
            items = line.strip().split()
            if len(items) > 0:
                tokId, tok, _, pos, _, _, dependTokId, dependType, _, _ = items
                tokens.append((tok, pos, dependTokId, dependType))
            else:
                # update
                for i in range(1, len(tokens)):
                    cur_tok, cur_pos, cur_dependTokId, cur_dependType = tokens[i]
                    _idx = int(cur_dependTokId)
                    depend_tok, depend_pos = tokens[_idx][0], tokens[_idx][1]
                    key = ((cur_tok, cur_pos), (depend_tok, depend_pos))
                    bigramsDependFreq[key].update([dependType])
    return bigramsDependFreq







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../ud.conllu")
    args = parser.parse_args()
    

    rt = conll_dataloader(args.data_path)
    import ipdb
    ipdb.set_trace()