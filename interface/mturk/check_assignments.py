import json
from pathlib import Path
from client import mturk, sql
from client.sql import Room, calc_score


def read_assignments():
    assignments_text = Path('./assignments.json').read_text()
    return json.loads(assignments_text)


def main():
    assignments = read_assignments()
    mturk_client = mturk.get_client()
    conn, cur = sql.get_conn_and_cursor()
    for assignment in assignments:
        query = "SELECT * FROM users WHERE assignment_id = '{}' ORDER BY id DESC LIMIT 1".format(assignment['AssignmentId'])
        cur.execute(query)
        user_record = cur.fetchone()
        if user_record is None:
            # print(assignment)
            continue

        room = sql.Room(cur, user_record.room_id)
        user = None
        for room_user in room.users:
            if room_user.id == user_record.id:
                user = room_user

        if room.users[0].worker_id == room.users[1].worker_id:
            continue
        # try:
        #     if user.score < 0.50:
        #         mturk_client.reject_assignment(
        #             AssignmentId=user.assignment_id,
        #             RequesterFeedback="Your score does not reach the acceptance condition. To accept the HIT, the score must be above 50pts. Your score is {} pts.".format(user.score * 100)
        #         )
        #     else:
        #         mturk_client.approve_assignment(
        #             AssignmentId=user.assignment_id
        #         )
        # except Exception as e:
        #     print(e)
        if user.score > 0.70 and user.bonus >= 0.01:
            if not user_record.is_bonus_paid:
                mturk_client.send_bonus(
                    WorkerId=user.worker_id,
                    AssignmentId=user.assignment_id,
                    BonusAmount="{:.2f}".format(user.bonus),
                    Reason="You got good score"
                )
                query = "UPDATE users SET is_bonus_paid=true WHERE id={}".format(user.id)
                cur.execute(query)
                conn.commit()

            

        # print(user)
        # print(room)

    # room_records = cur.fetchall()
    # rooms = [Room(cur, room_record.id) for room_record in room_records]


if __name__ == "__main__":
    main()
