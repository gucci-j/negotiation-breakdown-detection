import pandas as pd
from pathlib import Path
import sys
import numpy as np
import re

# negotiation units
nego_units = ('<sep>', '<end>', '<greet>', '<agree>', '<disagree>', '<inquire>', '<propose>', '<inform>', '<unk>')

# converter  
index_to_phrases = {index : unit for index, unit in enumerate(nego_units)}


def convert_to_units(text: str, player_tag=None, prev_tag=None):
    greet_flag = False
    agree_disagree_flag = False
    inquire_flag = False
    propose_flag = False
    queue = []
    ret_text = []

    # ====
    # Regular expression lists
    # ====
    # greet
    greet_result = re.search(r'\bhi\b|\bhello\b|\bhey\b|\bhiya\b|\bhowdy\b| (how are you) | (good day) | (good afternoon) | (good morning) |\byo\b', text)
    # disagree
    disagree_result = re.search(r"\bisn\'t\b|\bworse\b|\bbad\b|\bsorry\b|\bno\b|\bnot\b|\bnothing\b|\bdon\'t\b|\bcan`t\b|\bcan\'t\b|\bcant\b|\bcannot\b|\bafraid\b| (a lot lower) | (too much) | (too high) | (too low)", text)
    # agree
    agree_result = re.search(r"\bok\b| (no problem) |\bokay\b|\byes\b|\bgreat\b|\bperfect\b|\bthanks\b|\bgracias\b|\bthx\b| (thank you) |\bpleasure\b|\bfine\b|\bdeal\b|\bcool\b| (sounds good) | (very good) | (looks good) | (that works) | (that will work) | (it will work) | (i can do)", text)
    # inquire
    inquire_result = re.search(r"\bwhat\b|\bwhen\b|\bwhich\b|\bwhere\b| (how about) |\b(how\'s)\b| (how does) |\?| (do you) | (did you) | (did we) | (do we) | (do i) | (are you) | (would you) | (will you) | (could we) |  (could you) | (let me know) ", text)
    # propose
    propose_result = re.search(r"(\$\d{1,}) | (\d{1,}\$) | \d{1,} | (come down) | (highest) | (lowest) | (go hiher) | (go lower) | (i would like)", text) 
    print('> ', greet_result, disagree_result, agree_result, inquire_result, propose_result)

    # ====
    # extract matching indices
    # ====
    if greet_result is not None:
        greet_index = greet_result.start()
        queue.append((greet_index, 'greet'))

    if disagree_result is not None:
        agree_disagree_index = (disagree_result.start(), 'disagree')
        queue.append((agree_disagree_index[0], 'disagree'))
    elif agree_result is not None:
        agree_disagree_index = (agree_result.start(), 'agree')
        queue.append((agree_disagree_index[0], 'agree'))

    if inquire_result is not None:
        inquire_index = inquire_result.start()
        queue.append((inquire_index, 'inquire'))
    
    if propose_result is not None:
        propose_index = propose_result.start()
        queue.append((propose_index, 'propose'))
    
    if queue != []:
        print(queue)
        queue = sorted(queue, key=lambda x: x[0])
        print(queue)
    
    # ====
    # matching negotiation patterns
    # ====
    if player_tag == 'SELLER:':
        ret_text.append(nego_units[0])
    else: # buyer
        # ret_text.append(nego_units[1])
        ret_text.append(nego_units[0])

    # queue is null but prev_inquire is True
    if prev_tag is True and queue == []:
        ret_text.append(nego_units[7])
        return ret_text, False
    
    # queue is simply null
    elif queue == []:
        ret_text.append(nego_units[8])
        return ret_text, False
        
    for list_index, (start_index, tag) in enumerate(queue):
        if list_index == 0:
            # start with greet
            if tag == 'greet':
                ret_text.append(nego_units[2])
                greet_flag = True
                continue
            
            # start with any other tags
            if tag == 'agree':
                ret_text.append(nego_units[3])
                agree_disagree_flag = True
            elif tag == 'disagree':
                ret_text.append(nego_units[4])
                agree_disagree_flag = True
            elif tag == 'inquire':
                ret_text.append(nego_units[5])
                inquire_flag = True
                break
            elif tag == 'propose':
                ret_text.append(nego_units[6])
                propose_flag = True
            else:
                raise NotImplementedError()
        
        elif greet_flag is True:
            # comes with either propose or inquire
            if tag == 'inquire':
                ret_text.append(nego_units[5])
                inquire_flag = True
                break
            elif tag == 'propose':
                ret_text.append(nego_units[6])
                propose_flag = True
            # comes with an illegal tag
            else:
                break
        
        elif agree_disagree_flag is True:
            # only comes with inquire or propose
            if tag == 'inquire':
                ret_text.append(nego_units[5])
                inquire_flag = True
                break
            elif tag == 'propose':
                ret_text.append(nego_units[6])
                propose_flag = True
            # comes with an illegal tag
            else:
                break
        
        elif propose_flag is True:
            # only comes with inquire
            if tag == 'inquire':
                ret_text.append(nego_units[5])
                inquire_flag = True
            break

        else:
            raise NotImplementedError()
        
    if inquire_flag is True:
        return ret_text, True
    else:
        return ret_text, False        


def divide_utterance(text: str):
    comments = np.array([])
    comment = ""
    body = " "
    prev_player_tag = None
    inquire_flag = False

    for token in text.split(' '):
        # player tag
        if token == "BUYER:" or token == "SELLER:":
            if prev_player_tag is None: # init
                comment = token
                prev_player_tag = token

            elif prev_player_tag != token: # player changed
                # convert to nego units
                body += ' '
                print(prev_player_tag, body)
                ret_text, inquire_flag = convert_to_units(body, prev_player_tag, inquire_flag)
                print(f'> {ret_text}')
                comments = np.concatenate((comments, np.array(ret_text))) if comments.size else np.array(ret_text)
                body = " "
                comment = token
                prev_player_tag = token

            else: # player not changed -> discard
                continue

        # not a player tag
        else:
            body = body + ' ' + token
            comment = comment + ' ' + token

    else:
        body += ' '
        print(body)
        ret_text, inquire_flag = convert_to_units(body, prev_player_tag, inquire_flag)
        comments = np.concatenate((comments, np.array(ret_text))) if comments.size else np.array(ret_text)
        print(f'> {ret_text}')
    
    return np.array(comments).flatten().tolist()


def main(filename: str):
    new_dataset = []
    df = pd.read_csv(filename)

    for index, data in df.iterrows():
        print('=' * 20)
        print(f'Negotiation No. {index}')
        comments = divide_utterance(data['raw_text'])
        comments.append(nego_units[1])
        new_dataset.append([data['text'], comments, data['flag']])
        print(f'> {data["flag"]}')
    
    df2 = pd.DataFrame(new_dataset, columns=['text', 'meta_text', 'flag'])
    df2.to_csv(sys.argv[2], index=False)


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        raise Exception("Please give a valid dataset path!")
    path = sys.argv[1]
    main(path)