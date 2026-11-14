$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$runtimeDir = Join-Path $PSScriptRoot '.runtime'
$setupDir = Join-Path $PSScriptRoot '.runtime-setup'
New-Item -ItemType Directory -Path $setupDir -Force | Out-Null
$installer = Join-Path $setupDir 'python-3.13.17-amd64.exe'
$logPath = Join-Path $setupDir 'install.log'
$downloadPath = Join-Path $setupDir 'python-installer.download'
$downloadUrl = 'https://www.python.org/ftp/python/3.13.17/python-3.13.17-amd64.exe'

function Test-PythonSignature([string] $filePath) {
    if (-not (Test-Path -LiteralPath $filePath)) { return $false }
    $signature = Get-AuthenticodeSignature -LiteralPath $filePath
    return ($signature.Status -eq 'Valid' -and
        $signature.SignerCertificate.Subject -match 'Python Software Foundation')
}

if (-not (Test-PythonSignature $installer)) {
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe --location --fail --silent --show-error --connect-timeout 10 `
            --max-time 180 --output $downloadPath $downloadUrl
        if ($LASTEXITCODE -ne 0) { throw 'Python installer download failed.' }
    } else {
        Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl `
            -OutFile $downloadPath -TimeoutSec 180
    }
    if (-not (Test-PythonSignature $downloadPath)) {
        throw 'The Python installer signature is invalid or the publisher does not match.'
    }
    Move-Item -LiteralPath $downloadPath -Destination $installer -Force
}

$arguments = @(
    '/quiet', '/log', ('"' + $logPath + '"'),
    'InstallAllUsers=0', ('TargetDir="' + $runtimeDir + '"'),
    'Include_launcher=0', 'Include_pip=0', 'Include_test=0',
    'Include_doc=0', 'Include_tcltk=1', 'PrependPath=0',
    'AssociateFiles=0', 'Shortcuts=0'
)
$process = Start-Process -FilePath $installer -ArgumentList $arguments `
    -WindowStyle Hidden -PassThru -Wait
if ($process.ExitCode -notin @(0, 3010)) {
    throw "Python setup failed with exit code $($process.ExitCode). See $logPath"
}
& (Join-Path $runtimeDir 'python.exe') -c "import tkinter; print('Python and Tkinter ready')"
if ($LASTEXITCODE -ne 0) { throw 'Python/Tkinter verification failed.' }
