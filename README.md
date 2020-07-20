# mat2py

Some functions used in MatLab that were ported to Python. More details of the operation, please see the tutorial:

* [https://edersoncorbari.github.io/tutorials/mat2py/](https://edersoncorbari.github.io/tutorials/mat2py/)
* [https://dzone.com/articles/write-matlab-functions-to-python](https://dzone.com/articles/write-matlab-functions-to-python)

## Requirements

It is necessary only to have Python3 installed on your machine, and Pipenv, to install Pipenv see the article below:

* [https://github.com/pypa/pipenv](https://github.com/pypa/pipenv)

If you use Linux Ubuntu run:

```shell
sudo apt install python3-pip
sudo pip3 install pipenv
```

## Installing and Testing

With pipenv installed on the machine use the following commands:

```shell
git clone https://github.com/edersoncorbari/mat2py.git && cd mat2py
pipenv shell
pipenv install
```

Now just test:

```shell
python3 precision_test.py
```

Enjoy yourself! :-)
