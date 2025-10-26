# Contributing to Brain Connectivity-Based Cognitive Prediction

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version)
- Error messages or logs

### Suggesting Enhancements

Open an issue with:
- Clear description of the enhancement
- Rationale (why it's useful)
- Proposed implementation (if applicable)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit with clear messages (`git commit -m 'Add feature: ...'`)
7. Push to your fork (`git push origin feature/YourFeature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings (NumPy style)
- Add comments for complex logic
- Keep functions focused and modular

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Test on multiple platforms if possible

## Development Setup
Clone your fork
git clone https://github.com/YOUR_USERNAME/brain-connectivity-prediction.git
cd brain-connectivity-prediction

Create virtual environment
python -m venv venv
source venv/bin/activate # Linux/macOS

or
venv\Scripts\activate # Windows

Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt # If available

Run tests
python tests/test_project.py

text

## Questions?

Contact: jamil.hanouneh1997@gmail.com
