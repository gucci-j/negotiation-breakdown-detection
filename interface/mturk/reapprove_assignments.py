import json
from pathlib import Path
from client import mturk, sql
from client.sql import Room, calc_score
from collections import namedtuple


def read_assignments():
    assignments_text = Path('./assignments.json').read_text()
    return json.loads(assignments_text)


def main():
    Assignment = namedtuple('Assignment', ('assignment_id', 'worker_id', 'hit_id'))
    assignments = [
        Assignment(
            "3QAVNHZ3EOLXYRO2MXMHL9WSGE6LAW",
            "A1VGSOTBG8O7YA",
            "39O6Z4JLX3D93PAVL9BZ36A41ROXVY"
        )
    ]
    mturk_client = mturk.get_client()
    for assignment in assignments:
        mturk_client.approve_assignment(
            OverrideRejection=True,
            AssignmentId=assignment.assignment_id
        )
        # mturk_client.send_bonus(
        #     WorkerId=assignment.worker_id,
        #     AssignmentId=assignment.assignment_id,
        #     BonusAmount="1",
        #     Reason="Because of apologize for a bug in our system"
        # )

    # room_records = cur.fetchall()
    # rooms = [Room(cur, room_record.id) for room_record in room_records]


if __name__ == "__main__":
    main()
