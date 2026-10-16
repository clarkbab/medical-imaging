$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Resolve-Path "$ScriptDir\..\.."
$App = "$ScriptDir\marker_editor.py"

Set-Location $RepoDir

uv run streamlit run $App `
    --server.enableStaticServing true `
    --server.port 8502
