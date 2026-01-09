# Integration Tests

This directory contains integration tests for the KG pipeline.

## Running Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_config.py -v

# Run with coverage
pytest tests/integration/ --cov=src/kg --cov-report=html
```

## Test Structure

- `test_config.py` - Configuration loading and validation
- `test_dependencies.py` - Pipeline stage dependency resolution
- `conftest.py` - Shared fixtures and utilities

## Test Categories

### Configuration Tests
- Loading from YAML
- Environment variable injection
- Config overrides
- Default values

### Dependency Tests
- Automatic dependency resolution
- Execution order validation
- Stage registry functionality
