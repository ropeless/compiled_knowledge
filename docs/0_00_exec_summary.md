# Compiled Knowledge #

_efficient inferencing with probabilistic graphical models_

## What it is ##

A _probabilistic model_ captures probabilistic relationships between a collection of random variables.
Random variables represent properties in the world (real or hypothetical) that we want to model. They are
_random_ variables in that their interactions are not necessarily deterministic, rather we only know
the probabilistic (or statistical) relationships between the variables. A probabilistic model is built for a collection
of random variables and captures the probabilistic relationships between them.

The kinds of questions asked of a probabilistic model are things like, "What is the probability of some particular
situation arising", or, "What is the most likely situation?" These questions are often asked in the context of
know or assumed conditions.

Answering probabilistic questions can be hard, especially when there are many random variables and many 
relationships in the model. _Knowledge compilation_ is a computer science technique of re-representing
a probabilistic model in an alternative structure so that answering such questions is more efficient than
working directly with the model. Compiling a probabilistic model into an alternative structure may be
computationally expensive, but it is done when the cost is paid back via the many efficient queries that
are subsequently made.

Compiled Knowledge is a software library for creating, compiling and querying discrete probabilistic graphical models.
It is provided as a Python package that is [easily installed](https://pypi.org/project/compiled_knowledge/) and
comes with [comprehensive documentation](https://compiled-knowledge.readthedocs.io/) to support modellers
and researchers.

## What it offers ##

Compiled Knowledge provides a streamlined model compilation process, with many integrated query types.
It also includes modules for efficiently sampling models and using machine learning to create models.

It is designed with flexible workflows where computer science researchers can develop and integrate their
own algorithms.

Compiled Knowledge can integrate the ACE compiler provided by the Automated Reasoning Group, University of California Los Angeles.
See http://reasoning.cs.ucla.edu/ace/ for more information about ACE.

Many pre-built example models are provided (especially standard Bayesian networks used within the research community).

Compiled Knowledge is written in Python making it very accessible to modellers and data analysts.
Internally, some components use Cython and LLVM to accelerate crucial computational elements.
It is fast, flexible and efficient.

Being [open-source](https://github.com/ropeless/compiled_knowledge), the algorithms and implementation of
Compiled Knowledge are visible to all.
The Compiled Knowledge developers are open to scrutiny and welcome feedback.


## What it costs ##

Compiled Knowledge is provided free using the [MIT Licence](https://opensource.org/license/mit).
This licence makes it easy for developers and companies to use, modify, and distribute Compiled Knowledge source code.

Many may think nothing is ever truly free. In some sense that is true. 
In this case, you need to be confident you can handle the challenges of using probabilistic models.
If you want support with the use of probabilistic models, or you would like any Compiled Knowledge customisation,
please contact the [Compiled Knowledge Team](mailto:info@compiledknowledge.org).
