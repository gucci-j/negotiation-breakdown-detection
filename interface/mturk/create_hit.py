import boto3
import json
from pathlib import Path
# from pprint import pprint as print

# By default, we use the free-to-use Sandbox
create_hits_in_live = True

environments = {
        "live": {
            "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
            "preview": "https://www.mturk.com/mturk/preview",
            "manage": "https://requester.mturk.com/mturk/manageHITs",
            "reward": "0.00"
        },
        "sandbox": {
            "endpoint": "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
            "preview": "https://workersandbox.mturk.com/mturk/preview",
            "manage": "https://requestersandbox.mturk.com/mturk/manageHITs",
            "reward": "0.11"
        },
}
mturk_environment = environments["live"] if create_hits_in_live else environments["sandbox"]

MINUTES = 60
HOURS = 60
DAYS = 24

def main():
    question = Path("./question.xml").read_text()
    profile_name = "default"
    session = boto3.Session(profile_name=profile_name)
    client = session.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url=mturk_environment['endpoint'],
    )
    print(client.create_hit(
        MaxAssignments=20,
        AutoApprovalDelayInSeconds=MINUTES * HOURS * DAYS * 7,
        LifetimeInSeconds=MINUTES * HOURS * 3,
        AssignmentDurationInSeconds=MINUTES * HOURS * 1,
        Reward="0.5",
        QualificationRequirements=[{'ActionsGuarded': 'DiscoverPreviewAndAccept',
                                        'Comparator': 'EqualTo',
                                        'LocaleValues': [{'Country': 'US'}],
                                        'QualificationTypeId': '00000000000000000071',
                                        'RequiredToPreview': True},
                                    {'ActionsGuarded': 'DiscoverPreviewAndAccept',
                                        'Comparator': 'GreaterThan',
                                        'IntegerValues': [95],
                                        'QualificationTypeId': '000000000000000000L0',
                                        'RequiredToPreview': True},
                                    {'ActionsGuarded': 'DiscoverPreviewAndAccept',
                                        'Comparator': 'GreaterThan',
                                        'IntegerValues': [1000],
                                        'QualificationTypeId': '00000000000000000040',
                                        'RequiredToPreview': True}],
        Title="Play a negotiation chat game (5min ~ 20min, BONUS $1)",
        Keywords="chat, negotiation, conversation, english",
        Description="Negotiate as a job recruiter or worker, and get a bonus. If you do good, we will pay you a bonus (~$1).",
        Question=question,
    ))

if __name__ == "__main__":
    main()
