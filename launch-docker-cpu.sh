# USAGE - ./launch-docker-cpu.sh {abs-path-to-WorldModels-code}
# - eg. to run from current directory:
#     ./launch-docker-cpu.sh $(pwd)
if [[ "$OSTYPE" == "darwin"* ]]; then
    docker run --rm -p 8888:8888 -it -v $1:/Worldmodels worldmodels-image
else
    docker run --rm -it -p 8888:8888 -v $1:/Worldmodels worldmodels-image
fi
