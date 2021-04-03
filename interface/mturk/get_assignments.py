import sys
import boto3
import json
from pathlib import Path
from datetime import datetime
from pprint import pprint as print
from client import mturk


def support_datetime_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(repr(o) + " is not JSON serializable")


def main():
    hits = []
    assignments = []
    next_token = None
    client = mturk.get_client()

    while True:
        if next_token:
            response = client.list_hits(MaxResults=100, NextToken=next_token)  # type: dict
        else:
            response = client.list_hits(MaxResults=100)  # type: dict
        hits += response['HITs']  # type: list

        if 'NextToken' not in response:
            break
        next_token = response['NextToken']

    for h, hit in enumerate(hits):
        sys.stderr.write("{}/{} {}\n".format(h + 1, len(hits), hit['HITId']))
        response = client.list_assignments_for_hit(
            HITId=hit['HITId'],
            AssignmentStatuses=['Submitted', 'Approved', 'Rejected'],
        )
        assignments += response['Assignments']

    assignments_json_text = json.dumps(
        assignments,
        ensure_ascii=True,
        indent=True,
        default=support_datetime_default,
    )
    Path('./assignments.json').write_text(assignments_json_text)

if __name__ == "__main__":
    main()
