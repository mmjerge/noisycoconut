# Contributing to NoisyCoconut

We want to make contributing to this project as easy and transparent as possible.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

## Issues

We use GitHub issues to track public bugs and feature requests. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

### Bug Reports

When filing a bug report, please include:

* A clear and descriptive title
* Steps to reproduce the issue
* Expected behavior vs actual behavior
* Your environment (Python version, PyTorch version, model used, etc.)
* Any relevant error messages or logs

### Feature Requests

We welcome feature requests! Please describe:

* The problem you're trying to solve
* Your proposed solution (if any)
* Any alternatives you've considered

## Development Setup

1. Clone your fork of the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/noisy-coconut.git
   cd noisy-coconut

    Install dependencies:

    pip install -r requirements.txt
    pip install -r requirements-dev.txt  # for development dependencies

    Run tests:

    pytest tests/

    Run linting:

    flake8 noisy_coconut/
    black --check noisy_coconut/

Code Style

    We use Black for code formatting
    We use flake8 for linting
    Please include docstrings for new functions and classes
    Follow PEP 8 conventions

License

By contributing to NoisyCoconut, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
Acknowledgments

This project builds upon the Coconut (Chain of Continuous Thought) framework by Meta Research. We are grateful for their foundational work in latent-space reasoning.

