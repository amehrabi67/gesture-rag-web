param(
    [string]$RepoName = "gesture-rag-web",
    [switch]$Private
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is not installed. Install gh first."
}

gh auth status *> $null
if ($LASTEXITCODE -ne 0) {
    gh auth login
}

if (-not (Test-Path .git)) {
    git init
    git branch -M main
}

git add .
git commit -m "Create Gesture RAG web app" 2>$null

$visibility = if ($Private) { "--private" } else { "--public" }
gh repo create $RepoName $visibility --source . --remote origin --push

Write-Host "Pushed to GitHub. In the repository settings, enable Pages with GitHub Actions if prompted."
