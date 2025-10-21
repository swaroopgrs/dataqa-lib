#!groovy?
@Library('julesGlobalLibrary@6.HOTFIX-20251002-2') _

buildPipeline()

def buildPipeline() {
    jules_pipelineRunner {
        // If your jules.yml sits somewhere else in your repo, please
        // use a relative path here instead of just "jules.yml"
        yml = "jules.yml"
        archivePythonArtifacts = archivePythonArtifacts()
    }
}

def archivePythonArtifacts() {
    { steps, domain, config ->
        steps.echo "Archiving Python artifacts"
        steps.junit 'test-reports/unittest.xml'
        steps.archiveArtifacts artifacts: 'test-reports/coverage.xml'
    }
}
