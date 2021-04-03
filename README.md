Dialogue Act-based Breakdown Detection in Negotiation Dialogues
===

This repository contains the proposed dataset and the negotiation interface used to collect our data.  


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
@inproceedings{Yamaguchi-Iwasa-Fujita-2021,
    title = "Dialogue Act-based Breakdown Detection in Negotiation Dialogues",
    author = "Yamaguchi, Atsuki  and
      Iwasa, Kosui  and
      Fujita, Katsuhide",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics",
    year = "2021",
    publisher = "Association for Computational Linguistics"
}
```

## License
[MIT](./LICENSE) License