# Developing SNAKE {#developing-snake}
## Development environment {#developement-environment}

We recommend to use a virtual environment, and install SNAKE-FMRI and
its dependencies in it.


## Running tests {#running-tests}


## Writing documentation {#writing-documentation}

Documentation is available online at
<https://paquiteau.github.io/snake-fmri>

It can also be built locally :

```shell
  cd snake
  pip install -e .[doc]
  python -m sphinx docs docs_build
```

To view the html doc locally you can use :

```shell
  python -m http.server --directory docs_build 8000
```

And visit `localhost:8000` on your web browser.

