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
                // The sentence-transformers ban that used to live here was
                // removed: it banned the import in production code, the
                // dependency in pyproject.toml and the entry in uv.lock, dating
                // from when embeddings were remote-only. Local sentence-
                // transformers embeddings are now the documented design (the
                // llm_free path runs with zero API keys), so the gate
                // contradicted the package it was gating and had been failing
                // for a long time — which is part of why nothing here caught the
                // dependency drift.
                //
                // The `python3 scripts/check_imports.py` step was also removed:
                // that script was added in 66619ab and later deleted without
                // updating this file, so the stage errored before reaching
                // anything useful. `tests/test_packaging.py` now covers what it
                // was for — asserting that imports match declared dependencies —
                // and runs in the Test stages below.
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
                                --ignore-vuln CVE-2026-1839 \
                                --ignore-vuln CVE-2025-3000
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
                            UV_PROJECT_ENVIRONMENT=/tmp/graphknows-venv-311 \
                            uv sync --all-packages --group dev --python 3.11
                            UV_PROJECT_ENVIRONMENT=/tmp/graphknows-venv-311 \
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
                            UV_PROJECT_ENVIRONMENT=/tmp/graphknows-venv-313 \
                            uv sync --all-packages --group dev --python 3.13
                            UV_PROJECT_ENVIRONMENT=/tmp/graphknows-venv-313 \
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

        // ── Stage 3: Image — validate the ARTIFACT, not just the source ───────
        // Every other gate installs dependencies fresh (`uv sync`), so all of
        // them pass against a correct environment while the shipped image is
        // months stale. That is exactly how six undeclared dependencies survived
        // for seven weeks: nothing ever built the image and asked whether it
        // works. This stage does.
        //
        // The build takes several minutes (it bakes ~4GB of models), so it runs
        // on main and on any change that can invalidate the image, rather than on
        // every commit. `tests/test_packaging.py` is the fast gate that catches
        // declaration drift on every commit in the Test stages above.
        stage('Image') {
            when {
                anyOf {
                    branch 'main'
                    changeset pattern: 'Dockerfile', comparator: 'GLOB'
                    changeset pattern: 'pyproject.toml', comparator: 'GLOB'
                    changeset pattern: 'uv.lock', comparator: 'GLOB'
                    changeset pattern: 'scripts/*', comparator: 'GLOB'
                }
            }
            steps {
                sh '''
                    docker build --target runtime -t graphknows-ci:${BUILD_NUMBER} .

                    # Runs the image with the network cut and asserts every
                    # dependency imports and every model loads from cache.
                    IMAGE=graphknows-ci:${BUILD_NUMBER} scripts/docker-smoke.sh
                '''
            }
            post {
                always {
                    sh 'docker rmi graphknows-ci:${BUILD_NUMBER} || true'
                }
            }
        }

        // ── Stage 4: SonarQube — skipped when SONAR_TOKEN env var is blank ────
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
