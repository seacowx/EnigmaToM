# Run the Docker Image

The docker image is stored at https://hub.docker.com/repository/docker/seacow/masktom/general

First, pull the docker image with the following command:

```bash
docker pull seacow/masktom:latest
```

Next, create a container with the following command:

```bash
docker container create --name doccano \
  -e "ADMIN_USERNAME=masktom" \
  -e "ADMIN_PASSWORD=123456" \
  -v doccano-db:/data \
  -p 8000:8000 seacow/masktom
```

Start the container with the following command:

```bash
docker container start doccano
```

You can now access the Doccano instance at http://localhost:8000. Use the username __`seacow`__ and password __`Aa1234567`__ to log in.

Find the task with you name on it and start labeling the data.

Once finished, download the labeled data as JSON.