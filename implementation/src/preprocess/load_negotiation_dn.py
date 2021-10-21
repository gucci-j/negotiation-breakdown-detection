from pathlib import Path
import pandas as pd
import numpy as np
import torch
from decimal import Decimal, ROUND_HALF_UP
import sys


def length_regulator(negos: list, ratio: float):
    filtered_dataset = []

    # calc # of turns & format the dataset
    for nego in negos:
        comments = []
        raw_comments = []
        comment = ""
        raw_comment = ""
        prev_mover_tag = None

        for token in nego[0].split(' '):
            if token == "YOU:" or token == "THEM:":
                if prev_mover_tag != token and prev_mover_tag is not None:
                    comments.append(comment)
                    raw_comments.append(raw_comment)
                    comment = '<sep>'
                    raw_comment = token
                    prev_mover_tag = token
                else:
                    comment = '<sep>'
                    raw_comment = token
                    prev_mover_tag = token
            else:
                if comment == "":
                    comment = token
                    raw_comment = token
                else:
                    comment = comment + ' ' + token
                    raw_comment = raw_comment + ' ' + token
        
        # add an end tag
        raw_comments.append('<end>')
        comment

        # for early detection
        turn_count = 0
        num_turns = len(comments) * ratio
        num_turns = int(Decimal(str(num_turns)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
        if num_turns < 1:
            num_turns = 1
        
        dialogue = ""
        raw_dialogue = ""
        for raw_comment, comment in zip(raw_comments, comments):
            # for early detection
            if turn_count == num_turns:
                break
            turn_count += 1

            if dialogue != "":
                dialogue = dialogue + ' ' + comment
                if raw_comment != '<end>':
                    raw_dialogue = raw_dialogue + ' ' + raw_comment
            else:
                dialogue = comment
                if raw_comment != '<end>':
                    raw_dialogue = raw_comment
        
        dialogue += ' ' + '<end>'
        filtered_dataset.append([raw_dialogue, dialogue, nego[1]])
    
    return filtered_dataset
    

def load_negotiation(path: Path, is_like_dn: bool, ratio: float):
    """Load dataset and make csv files."""

    def get_tag(tokens, tag):
        return tokens[tokens.index('<'+ tag + '>') + 1: tokens.index('</' + tag + '>')]


    def remove_duplication(dataset: list):
        """Remove duplicated logs."""
        duplicate_flag = False
        nego_b = []
        filtered_dataset = []

        for index, nego in enumerate(dataset):
            if index >= 1:
                # check duplicate? index, breakdown flag, duplicate flag
                if (nego[0] == nego_b[0] + 1) and nego[2] == nego_b[2] and duplicate_flag is False:
                    nego_b = nego
                    duplicate_flag = True
                    continue
            
            filtered_dataset.append([nego[1], nego[2], nego[3], nego[4], nego[5], nego[-1]])
            nego_b = nego
            duplicate_flag = False

        return filtered_dataset


    def preprocessing_dataset(index: int, scenario, is_like_dn: bool):
        dialogue = ' '.join([token for token in scenario[3] if token != '<eos>'])

        value = [int(val) for val in scenario[0][1::2]]
        counter_value = [int(val) for val in scenario[1][1::2]]

        normed_value = np.array(value) / np.sum(value)
        normed_counter_value = np.array(counter_value) / np.sum(counter_value)
        values = np.concatenate((normed_value, normed_counter_value)).tolist()

        flag = 0
        if ('<disagree>' in scenario[2]) or ('<no_agreement>' in scenario[2]):
            flag = 1
            score_you, score_them = 0., 0.

            if is_like_dn is False:
                return [index, dialogue, flag, score_you, score_them, score_them + score_you, values]
            else:
                return [dialogue, flag, score_you, score_them, score_them + score_you, values]

        item_count = [(int(num_item.split('=')[1])) for num_item in scenario[2]]
        score_you = np.dot(value, item_count[:3])
        score_them = np.dot(counter_value, item_count[3:])
        # debug:
        # print(f'{score_you} : {score_them}')
        
        if is_like_dn is False:
            return [index, dialogue, flag, score_you, score_them, score_them + score_you, values]
        else:
            return [dialogue, flag, score_you, score_them, score_them + score_you, values]

    dataset = []

    text = path.read_text('utf-8').split('\n')
    for index, line in enumerate(text):
        tokens = line.strip().split() # split into elements
        scenario = []

        # for empty list
        if tokens == []:
            continue
        
        for tag in ['input', 'partner_input', 'output', 'dialogue']: 
            scenario.append(get_tag(tokens, tag))
        
        # discard unreached an agreement dialogue
        if '<disconnect>' in scenario[2]:
            continue
        scenario = preprocessing_dataset(index, scenario, is_like_dn)
        dataset.append(scenario)
    
    if is_like_dn is False:
        dataset = remove_duplication(dataset)
    
    dataset = length_regulator(dataset, ratio=ratio)
    print(f'{path.name}: {len(dataset)} scenarios.')
    df = pd.DataFrame(dataset, columns=['raw_text', 'text', 'flag'])
    df.to_csv(sys.argv[2], index=False)


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        raise Exception("Please give a valid dataset path!")
    path = sys.argv[1]
    r_path = Path(path)
    load_negotiation(r_path, is_like_dn=False, ratio=1.0)