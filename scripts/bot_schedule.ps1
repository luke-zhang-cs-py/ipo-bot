# The keyless bot's schedule for Windows Task Scheduler: the local alternative to .github/workflows/bot.yml.
#
#   powershell -ExecutionPolicy Bypass -File scripts\bot_schedule.ps1            register the four tasks
#   powershell -ExecutionPolicy Bypass -File scripts\bot_schedule.ps1 -Remove    remove them
#
# Slots are in UTC (a start time ending in "Z" is Task Scheduler's "synchronize across time zones"), so they do
# not move when daylight saving time changes. The tasks run as the current user, only when logged on; the bot's
# own lock stops overlapping runs, and its health report flags any slot that was missed (a sleeping laptop).
param([switch]$Remove)

$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python -ErrorAction Stop).Source
$jobs = @(
    @{ Name = "ipo-bot daily";   Job = "daily";   Start = "22:00"; Days = "Monday,Tuesday,Wednesday,Thursday,Friday"; Every = $null },
    @{ Name = "ipo-bot edgar";   Job = "edgar";   Start = "00:00"; Days = "Monday,Tuesday,Wednesday,Thursday,Friday"; Every = 3 },
    @{ Name = "ipo-bot weekly";  Job = "weekly";  Start = "06:00"; Days = "Saturday"; Every = $null },
    @{ Name = "ipo-bot monthly"; Job = "monthly"; Start = "08:00"; Days = $null; Every = $null }
)

foreach ($j in $jobs) {
    if (Get-ScheduledTask -TaskName $j.Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $j.Name -Confirm:$false
    }
    if ($Remove) { Write-Output "removed $($j.Name)"; continue }

    $action = New-ScheduledTaskAction -Execute $python -Argument "-m bot update --job $($j.Job)" -WorkingDirectory $repo
    $start = "2026-01-01T$($j.Start):00Z"
    if ($j.Days) {
        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek ($j.Days -split ",") -At "00:00"
    } else {
        $trigger = New-ScheduledTaskTrigger -Daily -At "00:00"   # replaced below by a monthly trigger on the 1st
    }
    $trigger.StartBoundary = $start
    if ($j.Every) {
        $rep = (New-ScheduledTaskTrigger -Once -At "00:00" -RepetitionInterval (New-TimeSpan -Hours $j.Every) -RepetitionDuration (New-TimeSpan -Days 1)).Repetition
        $trigger.Repetition = $rep
    }
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
    $task = Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $trigger -Settings $settings -Description "python -m bot update --job $($j.Job)"
    if (-not $j.Days) {
        # PowerShell has no monthly trigger cmdlet: set the task's trigger to the 1st of every month through COM.
        $service = New-Object -ComObject Schedule.Service
        $service.Connect()
        $def = $service.GetFolder("\").GetTask($j.Name).Definition
        $def.Triggers.Clear()
        $monthly = $def.Triggers.Create(4)        # TASK_TRIGGER_MONTHLY
        $monthly.StartBoundary = $start
        $monthly.DaysOfMonth = 1
        $monthly.MonthsOfYear = 4095              # every month
        $service.GetFolder("\").RegisterTaskDefinition($j.Name, $def, 4, $null, $null, 3) | Out-Null
    }
    Write-Output "registered $($j.Name): $start UTC"
}
