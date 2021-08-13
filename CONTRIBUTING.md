# Contributing to Meerkat

We welcome contributions of all kinds: code, documentation, feedback and support. If
 you use Meerkat in your work (blogs posts, research, company) and find it
  useful, spread the word!  
  
This contribution borrows from and is heavily inspired by [Huggingface transformers](https://github.com/huggingface/transformers). 

## How to contribute

There are 4 ways you can contribute:
* Issues: raising bugs, suggesting new features
* Fixes: resolving outstanding bugs
* Features: contributing new features
* Documentation: contributing documentation or examples

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to give feedback and move your request forward.

### Bugs

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on Github under Issues).

If you didn't find anything, please use the bug issue template to file a Github issue.  


### Features

A world-class feature request addresses the following points:

1. Motivation first:
  * Is it related to a problem/frustration with the library? If so, please explain
    why. Providing a code snippet that demonstrates the problem is best.
  * Is it related to something you would need for a project? We'd love to hear
    about it!
  * Is it something you worked on and think could benefit the community?
    Awesome! Tell us what problem it solved for you.
2. Write a *full paragraph* describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you
post it.

## Contributing (Pull Requests)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
`meerkat`. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/robustness-gym/robustness-gym) by
   clicking on the 'Fork' button on the repository's page. 
   This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your Github handle>/robustness-gym.git
   $ cd robustness-gym
   $ git remote add upstream https://github.com/robustness-gym/robustness-gym.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `main` branch.

4. Meerkat manages dependencies using [`poetry`](https://python-poetry.org). 
Set up a development environment with `poetry` by running the following command in
 a virtual environment:

   ```bash
   $ pip install poetry
   $ poetry install
   ```
Note: in order to pass the full test suite (step 5), you'll need to install all extra in addition. 
```bash
   $ poetry install --extras "adversarial augmentation summarization text vision"
```
5. Develop features on your branch.

   As you work on the features, you should make sure that the test suite
   passes:

   ```bash
   $ pytest
   ```

   Meerkat relies on `black` and `isort` to format its source code
   consistently. After you make changes, autoformat them with:

   ```bash
   $ make autoformat
   ```

   Meerkat also uses `flake8` to check for coding mistakes. Quality control
    runs in CI, however you should also run the same checks with:

   ```bash
   $ make lint
   ```

   If you're modifying documents under `docs/source`, make sure to validate that
   they can still be built. This check also runs in CI. To run a local check
   make sure you have installed the documentation builder requirements, by
   running `pip install -r docs/requirements.txt` from the root of this repository
   and then run:

   ```bash
   $ make docs
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```
   
   You can use `pre-commit` to make sure you don't forget to format your code properly, 
   the dependency should already be made available by `poetry`.
   
   Just install `pre-commit` for the `robustness-gym` directory,
   
   ```bash
   $ pre-commit install
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.
   
8. We follow a one-commit-per-PR policy. Before your PR can be merged, you will have to
 `git rebase` to squash your changes into a single commit.

### Checklist

0. One commit per PR.
1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`. These
   are useful to avoid duplicated work, and to differentiate it from PRs ready
   to be merged;
4. Make sure existing tests pass;
5. Add high-coverage tests. No quality testing = no merge.
6. All public methods must have informative docstrings that work nicely with sphinx.


### Tests

An extensive test suite is included to test the library behavior. 
Library tests can be found in the 
[tests folder](https://github.com/robustness-gym/meerkat/tree/main/tests).

From the root of the
repository, here's how to run tests with `pytest` for the library:

```bash
$ make test
```

You can specify a smaller set of tests in order to test only the feature
you're working on.

Per the checklist above, all PRs should include high-coverage tests. 
To produce a code coverage report, run the following `pytest`
```
pytest --cov-report term-missing,html --cov=meerkat .
```
This will populate a directory `htmlcov` with an HTML report. 
Open `htmlcov/index.html` in a browser to view the report. 


### Style guide

For documentation strings, Meerkat follows the 
[google style](https://google.github.io/styleguide/pyguide.html).