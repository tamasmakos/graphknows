// Jenkinsfile — graphknows CI
//
// Mirrors .github/workflows/ci.yml.
// Dependency graph:
//   lint → type-check
//   lint → test (3.11 and 3.13 in parallel)
//   lint → security, arch-check, dep-audit, secret-scan  (all independent)
//   lint + test → sonar  (skipped when SONAR_TOKEN credential is absent)
//
// Required Jenkins credentials (Manage Jenkins → Credentials):
//   • sonar-token  (Secret text) — optional; SonarQube stage is skipped when absent.
//
// First-time setup:
//   1. cd jenkins && docker compose up -d --build
//   2. Open http://localhost:8080, complete the wizard.
//   3. Create a new Pipeline job, set SCM to "Git" pointing at this repo,
//      or paste this file into a Pipeline script.

pipeline {
    agent any

    options {
        timeout(time: 60, unit: 'MINUTES')
        disableConcurrentBuilds(abortPrevious: true)
        timestamps()
        ansiColor('xterm')
    }

    environment {
        // Mirrors the env block in ci.yml
        PYTHONPATH          = '.'
        OPENROUTER_API_KEY  = 'ci-placeholder'
        SEARCH_SERVICE_URL  = 'http://localhost:8040'
        AGENT_SERVICE_URL   = 'http://localhost:8002'
        // Shared Python installations pre-seeded in the Jenkins Docker image
        UV_PYTHON_INSTALL_DIR = '/opt/uv-pythons'
    }

    stages {

        // ── Stage 1: Format & Lint ───────────────────────────────────────────
        // Gates type-check and test; other scans run independently.
        stage('Lint') {
            steps {
                sh '''
                    uvx ruff format --check --config ruff.toml graphknows/
                '''
                sh '''
                    uvx ruff check --no-fix --config ruff.toml graphknows/
                '''
                sh '''
                    # Ban sentence-transformers imports in production code
                    if grep -rn "from sentence_transformers\\|import sentence_transformers" \
                        --include="*.py" \
                        --exclude-dir=".venv" --exclude-dir="research" --exclude-dir="tests" \
                        graphknows/; then
                        echo "ERROR: sentence-transformers import found in production code"
                        exit 1
                    fi
                    if grep -q "sentence-transformers" pyproject.toml 2>/dev/null; then
                        echo "ERROR: sentence-transformers dep found in pyproject.toml"
                        exit 1
                    fi
                    if grep -c "sentence.transform" uv.lock 2>/dev/null | grep -qv "^0$"; then
                        echo "ERROR: sentence-transformers found in root uv.lock"
                        exit 1
                    fi
                    echo "sentence-transformers: clean"
                '''
                sh '''
                    python3 scripts/check_imports.py
                '''
            }
        }

        // ── Stage 2: All gates in parallel (all depend on lint passing) ──────
        stage('Gates') {
            parallel {

                // ── Type check ───────────────────────────────────────────────
                stage('Type Check') {
                    steps {
                        sh '''
                            uv sync --group dev --python 3.13
                            uv run --python 3.13 mypy \
                                graphknows/ \
                                --exclude "graphgen" \
                                --exclude "cli/memory" \
                                --exclude "feature_engineering" \
                                --exclude "forget" \
                                --exclude "mcp_ltm" \
                                --exclude "recall" \
                                --exclude "topic_summarizer" \
                                --strict \
                                --ignore-missing-imports \
                                --allow-untyped-decorators \
                                --no-warn-unused-ignores
                        '''
                    }
                }

                // ── Security scan (bandit) ────────────────────────────────────
                stage('Security') {
                    steps {
                        sh '''
                            uvx bandit \
                                -r graphknows/ \
                                --skip B101,B104,B105,B110 \
                                -f screen
                        '''
                    }
                }

                // ── Architecture check (tach) ─────────────────────────────────
                stage('Architecture') {
                    steps {
                        sh 'uvx tach check'
                    }
                }

                // ── Dependency vulnerability audit (pip-audit) ────────────────
                stage('Dep Audit') {
                    steps {
                        sh '''
                            uv export \
                                --no-dev --no-editable --no-hashes \
                                --format requirements-txt \
                                > /tmp/requirements-audit.txt
                            uvx pip-audit \
                                --requirement /tmp/requirements-audit.txt \
                                --progress-spinner off \
                                --ignore-vuln CVE-2025-69872 \
                                --ignore-vuln PYSEC-2025-217 \
                                --ignore-vuln CVE-2026-1839
                        '''
                    }
                }

                // ── Secret scanning (gitleaks) ────────────────────────────────
                stage('Secrets') {
                    steps {
                        sh 'gitleaks detect --source=. --verbose'
                    }
                }

                // ── Test — Python 3.11 ────────────────────────────────────────
                stage('Test Python 3.11') {
                    steps {
                        sh '''
                            UV_PROJECT_ENVIRONMENT=.venv-311 \
                            uv sync --all-packages --group dev --python 3.11
                            UV_PROJECT_ENVIRONMENT=.venv-311 \
                            uv build --package graphknows --out-dir dist/
                            UV_PROJECT_ENVIRONMENT=.venv-311 \
                            uv run --python 3.11 pytest \
                                tests/ \
                                -v --tb=short \
                                -m "not integration" \
                                --cov=graphknows \
                                --cov-config=.coveragerc \
                                --cov-report=xml:coverage-3.11.xml \
                                --cov-report=term-missing \
                                --cov-fail-under=23
                        '''
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'coverage-3.11.xml',
                                             allowEmptyArchive: true,
                                             fingerprint: true
                        }
                    }
                }

                // ── Test — Python 3.13 ────────────────────────────────────────
                stage('Test Python 3.13') {
                    steps {
                        sh '''
                            UV_PROJECT_ENVIRONMENT=.venv-313 \
                            uv sync --all-packages --group dev --python 3.13
                            UV_PROJECT_ENVIRONMENT=.venv-313 \
                            uv build --package graphknows --out-dir dist/
                            UV_PROJECT_ENVIRONMENT=.venv-313 \
                            uv run --python 3.13 pytest \
                                tests/ \
                                -v --tb=short \
                                -m "not integration" \
                                --cov=graphknows \
                                --cov-config=.coveragerc \
                                --cov-report=xml:coverage-3.13.xml \
                                --cov-report=term-missing \
                                --cov-fail-under=23
                        '''
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'coverage-3.13.xml',
                                             allowEmptyArchive: true,
                                             fingerprint: true
                        }
                    }
                }

            } // end parallel
        } // end Gates

        // ── Stage 3: SonarQube — skipped when SONAR_TOKEN env var is blank ────
        // SONAR_TOKEN is injected automatically via the .env file loaded by
        // jenkins/docker-compose.yml — no Jenkins credential needed.
        stage('SonarQube') {
            when {
                expression { env.SONAR_TOKEN?.trim() }
            }
            steps {
                sh '''
                    sonar-scanner \
                        -Dsonar.token=$SONAR_TOKEN
                '''
            }
        }

    } // end stages

    post {
        always {
            // Archive all coverage reports produced by the test matrix
            archiveArtifacts artifacts: 'coverage*.xml', allowEmptyArchive: true
        }
        failure {
            echo 'Pipeline failed — check the stage logs above.'
        }
        success {
            echo 'All checks passed.'
        }
    }
}
