# Web Data Processing Systems - Assignment #2

## Description
Link prediction with various Link Prediction models using LibKGE

Note Alex: Good luck on the initial download, it takes like +- 20 minutes.

## Todo
- [x]  Download LibKG
- [x]  Train DBPedia (consider other models TransE, conflicts etc)
- [ ]  Link prediction using test dataset
- [ ]  Determine whether there are relations
- [ ]  Simple method to filter out noise
- [ ]  Provide high quality output
- [ ]  Which links should be kept and which should not

## Fun commands

To start the container:
```
docker-compose up
```

To find the container's name:
```
docker container ls
```

To train a model:
```
docker exec -it <container_name> /bin/sh
kge start examples/train.yaml --folder="/kge/local/experiments/main/new" --job.device cpu 
```


To train a Rescal Model
```
kge start examples/rescal.yaml --folder="/kge/local/experiments/main/rescal" --job.device cpu 
```

To clean an output file
```
python3 clean.py results/rescal/test-result.csv > output.csv
```

To see the predictions, you need to modify the KGE model located here:
```
/kge/mode/kge_model.py
```

Add this code *above* the return statement [Line 702]

```py
        tensors = self._scorer.score_emb(s, p, o, combine="sp_")
        resultList = []

        for tensorOuter in tensors:
            resultOuter = []
            for tensorInner in tensorOuter:
                resultOuter.append(tensorInner.item())
            resultList.append(resultOuter)

        print(resultList)
```

And run this commend to test

```
kge test local/experiments/main/[DIR NAME] > test-result.csv
```
        
