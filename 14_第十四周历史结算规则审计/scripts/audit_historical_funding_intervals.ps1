$ErrorActionPreference = 'Stop'

$outDir = Join-Path $PSScriptRoot '..\output'
$dataDir = Join-Path $PSScriptRoot '..\data'
New-Item -ItemType Directory -Force -Path $outDir, $dataDir | Out-Null

$startMs = [DateTimeOffset]::Parse('2026-04-09T00:00:00Z').ToUnixTimeMilliseconds()
$endMs = [DateTimeOffset]::Parse('2026-10-07T00:00:01Z').ToUnixTimeMilliseconds()
$chunkMs = 30 * 24 * 60 * 60 * 1000
$retrievedAt = [DateTimeOffset]::UtcNow.ToString('o')
$records = [System.Collections.Generic.List[object]]::new()
$pages = [System.Collections.Generic.List[object]]::new()
$rawPages = [System.Collections.Generic.List[object]]::new()

function Add-NormalizedRows {
    param(
        [object[]]$Rows,
        [string]$Venue,
        [string]$Asset,
        [string]$Instrument,
        [string]$TimestampField,
        [string]$SourceUrl
    )

    foreach ($row in $Rows) {
        $stamp = [int64]$row.$TimestampField
        if ($stamp -lt $script:startMs -or $stamp -gt $script:endMs) { continue }
        $utc = [DateTimeOffset]::FromUnixTimeMilliseconds($stamp).ToUniversalTime()
        $script:records.Add([PSCustomObject]@{
            venue = $Venue
            asset = $Asset
            instrument = $Instrument
            funding_time_unix_ms = $stamp
            funding_time_utc = $utc.ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
            funding_minute_unix_ms = [int64]([math]::Floor($stamp / 60000) * 60000)
            funding_time_utc_minute = $utc.ToString('yyyy-MM-ddTHH:mmZ')
            utc_hour = $utc.Hour
            utc_minute = $utc.Minute
            historical_funding_rate = $row.fundingRate
            rate_semantics = 'historical API record; not a pre-settlement forecast'
            retrieved_at_utc = $script:retrievedAt
            source_url = $SourceUrl
        })
    }
}

# Binance accepts the full six-month window in one request at this event frequency.
foreach ($asset in @(@{ name = 'BTC'; symbol = 'BTCUSDT' }, @{ name = 'ETH'; symbol = 'ETHUSDT' })) {
    $url = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=$($asset.symbol)&startTime=$startMs&endTime=$endMs&limit=1000"
    $response = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 40
    $rows = @($response)
    $rawPages.Add([PSCustomObject]@{ venue = 'Binance'; asset = $asset.name; page = 1; request_url = $url; response = $response })
    Add-NormalizedRows -Rows $rows -Venue 'Binance' -Asset $asset.name -Instrument $asset.symbol -TimestampField 'fundingTime' -SourceUrl $url
    $times = @($rows | ForEach-Object { [int64]$_.fundingTime } | Sort-Object)
    $pages.Add([PSCustomObject]@{ venue = 'Binance'; asset = $asset.name; page = 1; row_count = $rows.Count; oldest_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[0]).ToString('o'); newest_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[-1]).ToString('o'); source_url = $url })
}

# OKX's after cursor moves backward; each next page begins strictly before the prior page's oldest time.
foreach ($asset in @(@{ name = 'BTC'; instrument = 'BTC-USDT-SWAP' }, @{ name = 'ETH'; instrument = 'ETH-USDT-SWAP' })) {
    $cursor = $endMs
    for ($page = 1; $page -le 20; $page++) {
        $url = "https://www.okx.com/api/v5/public/funding-rate-history?instId=$($asset.instrument)&after=$cursor&limit=400"
        $response = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 40
        $rows = @($response.data)
        $rawPages.Add([PSCustomObject]@{ venue = 'OKX'; asset = $asset.name; page = $page; request_url = $url; response = $response })
        if ($rows.Count -eq 0) {
            $pages.Add([PSCustomObject]@{ venue = 'OKX'; asset = $asset.name; page = $page; row_count = 0; oldest_utc = ''; newest_utc = ''; source_url = $url })
            break
        }
        Add-NormalizedRows -Rows $rows -Venue 'OKX' -Asset $asset.name -Instrument $asset.instrument -TimestampField 'fundingTime' -SourceUrl $url
        $times = @($rows | ForEach-Object { [int64]$_.fundingTime } | Sort-Object)
        $pages.Add([PSCustomObject]@{ venue = 'OKX'; asset = $asset.name; page = $page; row_count = $rows.Count; oldest_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[0]).ToString('o'); newest_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[-1]).ToString('o'); source_url = $url })
        if ($times[0] -le $startMs) { break }
        if ($times[0] -ge $cursor) { throw "OKX pagination did not move backward for $($asset.instrument)." }
        $cursor = $times[0]
    }
}

# Bybit requests non-overlapping 30-day time slices, safely below the endpoint page limit.
foreach ($asset in @(@{ name = 'BTC'; symbol = 'BTCUSDT' }, @{ name = 'ETH'; symbol = 'ETHUSDT' })) {
    $sliceStart = $startMs
    $page = 0
    while ($sliceStart -le $endMs) {
        $sliceEnd = [math]::Min($sliceStart + $chunkMs, $endMs)
        $url = "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=$($asset.symbol)&startTime=$sliceStart&endTime=$sliceEnd&limit=200"
        $response = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 40
        $rows = @($response.result.list)
        $page++
        $rawPages.Add([PSCustomObject]@{ venue = 'Bybit'; asset = $asset.name; page = $page; request_url = $url; response = $response })
        if ($rows.Count -gt 0) {
            Add-NormalizedRows -Rows $rows -Venue 'Bybit' -Asset $asset.name -Instrument $asset.symbol -TimestampField 'fundingRateTimestamp' -SourceUrl $url
            $times = @($rows | ForEach-Object { [int64]$_.fundingRateTimestamp } | Sort-Object)
            $pages.Add([PSCustomObject]@{ venue = 'Bybit'; asset = $asset.name; page = $page; row_count = $rows.Count; oldest_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[0]).ToString('o'); newest_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[-1]).ToString('o'); source_url = $url })
        }
        if ($sliceEnd -ge $endMs) { break }
        $sliceStart = $sliceEnd + 1
    }
}

$uniqueRecords = $records | Sort-Object venue, asset, funding_time_unix_ms -Unique
$uniqueRecords | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $dataDir '第十四周官方历史结算记录_2026-04至10.csv')
$pages | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $dataDir '第十四周API分页抓取清单.csv')
$rawPages | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 (Join-Path $dataDir '第十四周官方API原始分页响应.json')

$coverage = foreach ($group in ($uniqueRecords | Group-Object venue, asset)) {
    $items = @($group.Group)
    $times = @($items | ForEach-Object { [int64]$_.funding_time_unix_ms } | Sort-Object -Unique)
    $gaps = for ($i = 1; $i -lt $times.Count; $i++) { [math]::Round(($times[$i] - $times[$i - 1]) / 3600000, 3) }
    $gapSummary = @($gaps | Group-Object | Sort-Object { [double]$_.Name } | ForEach-Object { '{0}h={1}' -f $_.Name, $_.Count }) -join '; '
    $nonEightHourGaps = @($gaps | Where-Object { $_ -ne 8 }).Count
    $utcHours = @($items | Group-Object utc_hour | Sort-Object { [int]$_.Name } | ForEach-Object { '{0}: {1}' -f $_.Name, $_.Count }) -join '; '
    [PSCustomObject]@{
        venue_asset = $group.Name
        record_rows = $items.Count
        unique_event_times = $times.Count
        first_event_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[0]).ToString('yyyy-MM-ddTHH:mm:ssZ')
        last_event_utc = [DateTimeOffset]::FromUnixTimeMilliseconds($times[-1]).ToString('yyyy-MM-ddTHH:mm:ssZ')
        utc_hour_counts = $utcHours
        gap_hours_distribution = $gapSummary
        non_8h_gaps = $nonEightHourGaps
    }
}
$coverage | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $outDir '第十四周平台资产结算间隔摘要.csv')

$alignment = foreach ($assetGroup in ($uniqueRecords | Group-Object asset)) {
    $slots = $assetGroup.Group | Group-Object funding_minute_unix_ms
    $venuesBySlot = foreach ($slot in $slots) {
        [PSCustomObject]@{ slot = $slot; venues = @($slot.Group.venue | Sort-Object -Unique) }
    }
    $threeVenueSlots = @($venuesBySlot | Where-Object { $_.venues.Count -eq 3 })
    $binanceBybitSlots = @($venuesBySlot | Where-Object { $_.venues -contains 'Binance' -and $_.venues -contains 'Bybit' })
    $binanceOkxSlots = @($venuesBySlot | Where-Object { $_.venues -contains 'Binance' -and $_.venues -contains 'OKX' })
    $bybitOkxSlots = @($venuesBySlot | Where-Object { $_.venues -contains 'Bybit' -and $_.venues -contains 'OKX' })
    $maxDrift = 0
    foreach ($entry in $threeVenueSlots) {
        $slot = $entry.slot
        $stamps = @($slot.Group | ForEach-Object { [int64]$_.funding_time_unix_ms })
        $drift = ($stamps | Measure-Object -Maximum).Maximum - ($stamps | Measure-Object -Minimum).Minimum
        if ($drift -gt $maxDrift) { $maxDrift = $drift }
    }
    $allThreeSlots = @($threeVenueSlots | ForEach-Object { [int64]$_.slot.Name })
    $allThreeStart = if ($allThreeSlots.Count) { [DateTimeOffset]::FromUnixTimeMilliseconds(($allThreeSlots | Measure-Object -Minimum).Minimum).ToString('yyyy-MM-ddTHH:mmZ') } else { '' }
    $allThreeEnd = if ($allThreeSlots.Count) { [DateTimeOffset]::FromUnixTimeMilliseconds(($allThreeSlots | Measure-Object -Maximum).Maximum).ToString('yyyy-MM-ddTHH:mmZ') } else { '' }
    [PSCustomObject]@{
        asset = $assetGroup.Name
        unique_minute_slots = $slots.Count
        Binance_Bybit_shared_slots = $binanceBybitSlots.Count
        Binance_Bybit_share = if ($slots.Count) { [math]::Round($binanceBybitSlots.Count / $slots.Count, 4) } else { 0 }
        Binance_OKX_shared_slots = $binanceOkxSlots.Count
        Bybit_OKX_shared_slots = $bybitOkxSlots.Count
        slots_with_all_three_venues = $threeVenueSlots.Count
        all_three_window_utc = if ($allThreeStart) { "$allThreeStart to $allThreeEnd" } else { '' }
        all_three_share_of_observed_union = if ($slots.Count) { [math]::Round($threeVenueSlots.Count / $slots.Count, 4) } else { 0 }
        max_cross_venue_timestamp_drift_ms = $maxDrift
    }
}
$alignment | Export-Csv -NoTypeInformation -Encoding utf8 (Join-Path $outDir '第十四周跨所历史结算时间匹配.csv')

Write-Output ("Unique normalized rows: {0}" -f @($uniqueRecords).Count)
Write-Output 'Venue/asset interval summary:'
$coverage | Format-Table -AutoSize | Out-String | Write-Output
Write-Output 'Cross-venue minute matching:'
$alignment | Format-Table -AutoSize | Out-String | Write-Output
