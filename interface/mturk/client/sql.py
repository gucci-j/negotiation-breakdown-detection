import psycopg2
import psycopg2.extras
from collections import namedtuple


def get_conn():
    return psycopg2.connect(
        host="localhost",
        database="mturk_dev",
        user="test",
        password="test"
    )


def get_cursor():
    return get_conn().cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)


def get_conn_and_cursor():
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
    return conn, cursor


class Room(object):
    def __init__(self, cur: psycopg2.extras.NamedTupleCursor, room_id: int):
        cur.execute("SELECT * FROM rooms WHERE id = {} LIMIT 1".format(room_id))
        room = cur.fetchone()
        cur.execute("SELECT * FROM users WHERE room_id = {}".format(room.id))
        user_records = cur.fetchall()
        cur.execute("SELECT * FROM comments WHERE user_id in ({}) ORDER BY created_at".format(
            ", ".join(map(lambda ru: str(ru.id), user_records))))
        comments = cur.fetchall()
        cur.execute("SELECT * FROM solutions WHERE user_id in ({}) ORDER BY created_at".format(
            ", ".join(map(lambda ru: str(ru.id), user_records))))
        solutions = cur.fetchall()

        solution = None
        if room.status == 'completed':
            cur.execute(
                "SELECT * FROM solutions WHERE user_id in ({}) AND accepted = TRUE LIMIT 1".format(
                    ", ".join(map(lambda ru: str(ru.id), user_records))))
            fetched = cur.fetchone()
            if fetched:
                solution = fetched.body

        users = []
        User = namedtuple('User', ('id', 'role', 'score',
                                   'assignment_id', 'worker_id', 'reward',
                                   'bonus'))
        for user_record in user_records:
            if solution:
                score = calc_score(user_record, solution)
            else:
                score = 0
            if score < 0.5:
                reward = 0
                bonus = 0
            elif score < 0.7:
                reward = 0.5
                bonus = 0
            else:
                bonus_percentage = (score - 0.7) / 0.3
                bonus = bonus_percentage * 1.0
                reward = 0.5 + bonus
            users.append(User(user_record.id,
                              user_record.role,
                              score,
                              user_record.assignment_id,
                              user_record.worker_id,
                              reward,
                              bonus))

        self.id = room.id
        self.status = room.status
        self.created_at = room.created_at
        self.users = users
        self.solution = solution
        self.comments = comments
        self.solution = solution
        self.solutions = solutions
        self.messages = sorted(comments + solutions,
                               key=lambda x: x.created_at)


def calc_score(user, solution):
    score = 0
    for utility in user.utilities:
        if utility['type'] == 'INTEGER':
            option = solution[utility['name']]
            if user.role == 'recruiter':
                score += utility['weight'] * \
                    (utility['max'] - option) / \
                    (utility['max'] - utility['min'])
            else:
                score += utility['weight'] * \
                    (option - utility['min']) / \
                    (utility['max'] - utility['min'])
        elif 'relatedTo' in utility:
            related_option = solution[utility['relatedTo']]
            option = solution[utility['name']]
            biasedWeights = [o['biasedWeights']
                             for o in utility['options'] if o['name'] == option]
            if len(biasedWeights) == 0:
                continue

            weight = biasedWeights[0][related_option]
            score += weight * utility['weight']
        else:
            option = solution[utility['name']]
            weights = [o['weight']
                       for o in utility['options'] if o['name'] == option]
            if len(weights) == 0:
                continue
            weight = weights[0]
            score += weight * utility['weight']
    return score