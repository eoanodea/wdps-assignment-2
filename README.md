Note Alex: Good luck on the initial download, it takes like +- 20 minutes.

Fun commands:

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
cd /kge
kge start examples/toy-complex-train.yaml --job.device cpu
```
