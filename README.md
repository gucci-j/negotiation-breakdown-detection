Dialogue Act-based Breakdown Detection in Negotiation Dialogues
===

This repository contains the proposed dataset and the negotiation interface used to collect our data. It also contains the official implementation.


## Dataset
Please unzip `data.zip`. The unzipped data: `data.json` is our proposed JI dataset.


## Negotiation Interface  
Implementations regarding our negotiation interface are available in the `interface` folder.


## How to extract samples
We provide a helper script: `helper/negotiation_ji.py` that easily enables users to extract the attributes of each dialogue in the JI dataset. 

Any utterances in a dialogue can be extracted with the following procedures:
1. Load dialogues using `read_ji_negotiations(filename=/path/to/data.json/)`. This will return the list of the `Negotiation` object.

2. Get the list of `Comment` object from each `Negotiation` object.


3. `Comment.body` has an utterance from a certain user. 


## Citation  
```
@inproceedings{yamaguchi-etal-2021-dialogue,
    title = "Dialogue Act-based Breakdown Detection in Negotiation Dialogues",
    author = "Yamaguchi, Atsuki  and
      Iwasa, Kosui  and
      Fujita, Katsuhide",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.63",
    pages = "745--757",
    abstract = "Thanks to the success of goal-oriented negotiation dialogue systems, studies of negotiation dialogue have gained momentum in terms of both human-human negotiation support and dialogue systems. However, the field suffers from a paucity of available negotiation corpora, which hinders further development and makes it difficult to test new methodologies in novel negotiation settings. Here, we share a human-human negotiation dialogue dataset in a job interview scenario that features increased complexities in terms of the number of possible solutions and a utility function. We test the proposed corpus using a breakdown detection task for human-human negotiation support. We also introduce a dialogue act-based breakdown detection method, focusing on dialogue flow that is applicable to various corpora. Our results show that our proposed method features comparable detection performance to text-based approaches in existing corpora and better results in the proposed dataset.",
}
```

## License
[MIT](./LICENSE) License
