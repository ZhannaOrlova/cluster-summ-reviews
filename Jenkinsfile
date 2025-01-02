pip install kfp

pipeline {
    agent any

    environment {
        PYPI_REPO_URL = 'https://test.pypi.org/legacy/' // Replace with your repository
        PYPI_USERNAME = credentials('pypi_username')   // Jenkins credentials for PyPI
        PYPI_PASSWORD = credentials('pypi_password')   // Stored Jenkins credentials
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'dev', url: 'https://github.com/your-python-package-repo.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                ./venv/bin/pip install --upgrade pip setuptools wheel
                '''
            }
        }

        stage('Build Package') {
            steps {
                sh './venv/bin/python setup.py sdist bdist_wheel'
            }
        }

        stage('Publish Package') {
            steps {
                sh '''
                ./venv/bin/twine upload --repository-url $PYPI_REPO_URL \
                -u $PYPI_USERNAME -p $PYPI_PASSWORD dist/*
                '''
            }
        }
    }
}
