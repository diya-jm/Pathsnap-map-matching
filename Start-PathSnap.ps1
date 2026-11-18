param([ValidateSet('api','ui')][string]$Component='api',[string]$Bind='127.0.0.1',[string]$Trips='data/Student folder/student_test',[int]$Port=0)
Set-Location -LiteralPath $PSScriptRoot
$localPython=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $localPython) {
    & $localPython serve.py $Component --host $Bind --trips $Trips --port $Port
} elseif (Test-Path -LiteralPath '.packages') {
    $runtime=Get-ChildItem -Path 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.*/dependencies/python/python.exe' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (!$runtime) { throw 'Install Python 3.12 and the requirements as described in README.md.' }
    & $runtime.FullName -c "import sys,runpy; sys.path[:0]=['.packages','.']; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')" serve.py $Component --host $Bind --trips $Trips --port $Port
} else {
    throw 'Set up the virtual environment using README.md first.'
}
