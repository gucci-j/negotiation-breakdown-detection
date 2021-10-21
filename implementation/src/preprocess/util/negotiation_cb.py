import json
from pathlib import Path
from typing import List

class User():
    """
    The instance of user.

    `user_id`: 0 or 1  
    `role`: seller or buyer  
    `target_price`: target price  
    `item_info`: bargaining item information given in "kbs"  
    """

    def __init__(self, user_id: int, role: str, target_price: float, item_info: dict):
        self.user_id = user_id
        self.role = role
        self.target_price = target_price
        self.item_info = item_info


class Comment():
    """
    The instance of comment.

    `user_id`: 0 or 1  
    `body`: dialogue history  
    `created_at`: time  
    """

    def __init__(self, user_id: int, body: str, created_at: float):
        self.user_id = user_id
        self.body = body
        self.created_at = created_at


class Bid():
    """
    The instance of Offer.

    `user_id`: 0 or 1  
    `proposed_price`: float  
    `created_at`: float  
    """

    def __init__(self, user_id: int, proposed_price: float, created_at: float):
        self.user_id = user_id
        self.proposed_price = proposed_price
        self.created_at = created_at


class Negotiation():
    """
    The class instance of each negotiation dialogue.

    `uuid`: unique id for the log  
    `scenario_uuid`: unique id for the scenario  
    `param status`: show quit/rejected/completed/aborted  
    `users`: list of User instance  
    `comments`: list of Comment instance  
    `bids`: list of Bid instance  
    `solution`: agreed bid info  
    """

    def __init__(self, uuid: str, scenario_uuid: str, status: str, 
        users: List[User], comments: List[Comment], bids: List[Bid],
        reward: float, list_price: float, agreed_price: float):

        self.uuid = uuid
        self.scenario_uuid = scenario_uuid
        self.status = status
        self.users = users
        self.comments = comments
        self.bids = bids

        if self.status == "quit" or self.status == "rejected" or self.status == "aborted":
            self.solution = None
        elif self.status == "completed":
            self.solution = {"reward": 1.0, "list_price": list_price, "agreed_price": agreed_price}
        else:
            raise Exception("Illegal reward info!")


def read_cb_negotiations(file_name: str, breakdown_flag=False) -> List[Negotiation]:
    """
    Read negotiations from file.

    `file_name`: file path which contains negotiations with json format  
    `breakdown_flag`: whether to include failed dialogues or not  
    """
    negotiations = []
    ac = 0
    qc = 0
    rc = 0

    for path in ['train.json', 'test.json', 'dev.json']:
        negos_raw = json.loads(Path(f'{file_name}{path}').read_text()) 
        # debug
        # print(negos_raw[0]["scenario"]["kbs"][0])
        
        for nego_raw in negos_raw:
            # Create `User` instances
            users_raw = nego_raw["scenario"]["kbs"]
            users = []
            for index, user_raw in enumerate(users_raw):
                users.append(User(index, user_raw["personal"]["Role"], float(user_raw["personal"]["Target"]), user_raw["item"]))
            
            # debug
            # print(f'{users[1].user_id}, {users[1].role}, {users[1].target_price}, {users[1].item_info}')

            # Create `Comment` & `Bid` instances
            events = nego_raw["events"]
            comments = []
            bids = []
            for event in events:
                if event["action"] == "message":
                    comments.append(Comment(event["agent"], event["data"], event["time"]))
                elif event["action"] == "offer":
                    bids.append(Bid(event["agent"], event["data"]["price"], event["time"]))

            # debug
            # print(f'{comments[0].body}, {bids[0].proposed}')

            # Create an `Negotiation` instance
            status = ""
            agreed_price = None
            if ("offer" in nego_raw["outcome"]) is False:
                status = "aborted"
                ac += 1
            elif nego_raw["outcome"]["offer"] is None:
                status = "quit"
                qc += 1
            elif nego_raw["outcome"]["reward"] == 0:
                status = "rejected"
                rc += 1
            else:
                status = "completed"
                agreed_price = nego_raw["outcome"]["offer"]["price"]
            list_price = nego_raw["scenario"]["kbs"][0]["item"]["Price"]
            negotiations.append(Negotiation(nego_raw["uuid"], nego_raw["scenario_uuid"], status, users, comments, bids, 
                                            float(nego_raw["outcome"]["reward"]), float(list_price), agreed_price))

    # print(f'{ac}, {qc}, {rc}')
    if breakdown_flag is False:
        negotiations = [n for n in negotiations if n.status == "completed"]
        # train: 5247, 18, 956, 357, 3916
        # test: 838, 2, 142, 56, 638
        # dev: 597, 2, 102, 42, 451
        # all: 6682, 22, 1200, 455, 5005

    return negotiations


def main():
    negos = read_negotiations("./data/")
    print("There are {} completed negotiations.".format(len(negos)))


if __name__ == '__main__':
    main()