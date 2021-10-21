import pandas as pd
from util import read_cb_negotiations
import re
from decimal import Decimal, ROUND_HALF_UP
import sys

def preprocess_comment(raw_comment: str):
    comment_list = re.split('([,.!?])', raw_comment)
    comment = ""

    for token in comment_list:
        token = token.strip(' ')
        token = token.replace('  ', ' ')
        token = token.replace('\\', '')
        token = token.replace('`', '')
        if token == '':
            continue
        if comment != "":
            comment = comment + ' ' + token
        else:
            comment = token
    
    return comment


def load_negotiation(path:str, include_breakdown: bool, ratio: float):
    """Load json dataset and make the corresponding csv file."""

    negos = read_cb_negotiations(path, include_breakdown)
    print("There are {} completed negotiations.".format(len(negos)))

    dataset = []
    bad_count = 0
    breakdown_count = 0
    for nego in negos:
        # remove short negos
        if len(nego.comments) < 3:
            bad_count += 1
            continue
        
        id_to_role = {}
        for user in nego.users:
            id_to_role[user.user_id] = user.role
        
        # for early detection
        turn_count = 0
        num_turns = len(nego.comments) * ratio
        num_turns = int(Decimal(str(num_turns)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
        if num_turns < 1:
            num_turns = 1
        
        # concatenate & insert player tags
        dialogue = ""
        raw_dialogue = ""
        for raw_comment in nego.comments:
            # for early detection
            if turn_count == num_turns:
                break
            turn_count += 1

            role = ""
            if id_to_role[raw_comment.user_id] == "seller":
                role = "SELLER:"
            else:
                role = "BUYER:"
            
            comment = raw_comment.body.replace('\n', ' ').lower()
            comment = preprocess_comment(comment)
            raw_comment = role + ' ' + comment
            comment = '<sep>' + ' ' + comment
            
            if dialogue != "":
                dialogue = dialogue + ' ' + comment
                raw_dialogue = raw_dialogue + ' ' + raw_comment
            else:
                dialogue = comment
                raw_dialogue = raw_comment
        
        flag = 0
        if nego.status != "completed":
            flag = 1
            breakdown_count += 1
        
        dialogue += ' ' + '<end>'
        dataset.append([raw_dialogue, dialogue, flag])

    df = pd.DataFrame(dataset, columns=['raw_text', 'text', 'flag'])
    df.to_csv(sys.argv[2], index=False)
    print(f'excluding: {bad_count}, breakdowns: {breakdown_count}')

    # There are 6682 completed negotiations.
    # excluding: 695, breakdowns: 1133

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        raise Exception("Please give a valid dataset path!")
    path = sys.argv[1]
    load_negotiation(path, include_breakdown=True, ratio=1.0)