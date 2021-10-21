import pandas as pd
from util import read_ji_negotiations
import re
from decimal import Decimal, ROUND_HALF_UP
import random
import sys

def preprocess_comment(raw_comment: str):
    comment_list = re.split('([,.!?])', raw_comment)
    comment = ""

    for token in comment_list:
        token = token.strip(' ')
        token = token.replace('  ', ' ')
        if token == '':
            continue
        if comment != "":
            comment = comment + ' ' + token
        else:
            comment = token
    
    return comment


def get_keys_from_value(d, prev_val, current_val=None):
    if current_val is not None:
        for k, v in d.items():
            if prev_val < v and v < current_val:
                return k
    else:
        ret = []
        for k, v in d.items():
            if prev_val < v:
                ret.append((v, k))
        ret = sorted(ret, key=lambda x: x[0])
        ret = [k for v, k in ret]
        return ret
    return None


def load_negotiation(include_breakdown: bool, ratio: float):
    """Load json dataset and make the corresponding csv file."""
    negos = read_ji_negotiations(sys.argv[1], include_breakdown)
    print("There are {} completed negotiations.".format(len(negos)))
    dataset = []
    bad_count = 0
    breakdown_count = 0

    for nego in negos:
        # remove short negos
        if len(nego.comments) < 3:
            bad_count += 1
            continue
        
        # for early detection
        turn_count = 0
        num_turns = len(nego.comments) * ratio
        num_turns = int(Decimal(str(num_turns)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
        if num_turns < 1:
            num_turns = 1
        
        # for obtaining bids
        bids = {}
        prev_created_at = 0.0
        for bid in nego.bids:
            bids[bid] = bid.created_at
        
        # concatenate & insert player tags
        dialogue = ""
        raw_dialogue = ""
        role_dialogue = ""
        agree_flag = False
        for raw_comment in nego.comments:
            if turn_count == num_turns:
                break
            
            # extract bid info
            # -> We include *all* intermediate bid information (agree, disagree, 
            #    and propose) for breakdown detection to alert negotiators 
            #    when a negotiation has a possibility to be broken down before being finalised.
            #    This is different from the Table description in the paper in that 
            #    we also consider agree bids.
            bid = get_keys_from_value(bids, prev_created_at, raw_comment.created_at)
            if bid is not None:
                accepter = ""
                if bid.user.context["role"] == "worker":
                    role = "WORKER:"
                    accepter = "RECRUITER:"
                else:
                    role = "RECRUITER:"
                    accepter = "WORKER:"
                
                comment = '<sep>' + ' ' + '<propose>'
                role_comment = role + ' ' + '<propose>'
                if dialogue != "":
                    dialogue = dialogue + ' ' + comment
                    role_dialogue = role_dialogue + ' ' + role_comment
                else:
                    dialogue = comment
                    role_dialogue = role_comment

                if bid.accepted is True:
                    dialogue = dialogue + ' ' + accepter + ' ' + '<agree>'
                    role_dialogue = role_dialogue + ' ' + accepter + ' ' + '<agree>'
                    agree_flag = True
                else:
                    dialogue = dialogue + ' ' + accepter + ' ' + '<disagree>'
                    role_dialogue = role_dialogue + ' ' + accepter + ' ' + '<disagree>'
                    
            # extract body info
            role = ""
            if raw_comment.user.context["role"] == "worker":
                role = "WORKER:"
            else:
                role = "RECRUITER:"
            comment = raw_comment.body.replace('\n', ' ').lower()
            comment = preprocess_comment(comment)
            role_comment = role + ' ' + comment
            comment = '<sep>' + ' ' + comment
            
            # comnbine
            if dialogue != "":
                dialogue = dialogue + ' ' + comment
                raw_dialogue = raw_dialogue + ' ' + comment
                role_dialogue = role_dialogue + ' ' + role_comment
            else:
                dialogue = comment
                raw_dialogue = comment
                role_dialogue = role_comment
            
            # for next iter
            turn_count += 1
        
        # add proposals
        if agree_flag is False:
            ret_bids = get_keys_from_value(bids, raw_comment.created_at)
            if ret_bids != []:
                for bid in ret_bids:
                    accepter = ""
                    if bid.user.context["role"] == "worker":
                        role = "WORKER:"
                        accepter = "RECRUITER:"
                    else:
                        role = "RECRUITER:"
                        accepter = "WORKER:"
                    comment = '<sep>' + ' ' + '<propose>'
                    role_comment = role + ' ' + '<propose>'
                    if dialogue != "":
                        dialogue = dialogue + ' ' + comment
                        role_dialogue = role_dialogue + ' ' + role_comment
                    else:
                        dialogue = comment
                        role_dialogue = role_comment

                    if bid.accepted is True:
                        dialogue = dialogue + ' ' + '<sep>' + ' ' + '<agree>'
                        role_dialogue = role_dialogue + ' ' + accepter + ' ' + '<agree>'
                        agree_flag = True
                    else:
                        dialogue = dialogue + ' ' + '<sep>' + ' ' + '<disagree>'
                        role_dialogue = role_dialogue + ' ' + accepter + ' ' + '<disagree>'

        flag = 0
        score_you, score_them = 0.0, 0.0
        if nego.status != "completed" or nego.solution is None:
            flag = 1
            breakdown_count += 1
        else:
            score_you = nego.users[0].calc_score(nego.solution)
            score_them = nego.users[1].calc_score(nego.solution)   
        raw_dialogue += ' ' + '<end>'
        dialogue += ' ' + '<end>'
        dataset.append([raw_dialogue, role_dialogue, flag])

    df = pd.DataFrame(dataset, columns=['raw_text', 'role_text', 'flag'])
    df.to_csv(sys.argv[2], index=False)
    print(f'excluding: {bad_count}, breakdowns: {breakdown_count}')
    # There are 2639 completed negotiations.
    # excluding: 62, breakdowns: 127

if __name__ == '__main__':
    if len(sys.argv) <= 2:
        raise Exception("Please give a valid dataset path!")
    load_negotiation(include_breakdown=True, ratio=1.0)